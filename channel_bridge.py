"""
channel_bridge.py — Pure-NumPy channel simulator for simulation mode.
Functional equivalent of GNU Radio's channel_model block with zero GR dependency.

Applies impairments in spec order (Section 8):
  multipath (FIR) -> CFO (phase rotation) -> SCO (resampling) -> AWGN

FIX: previous order was multipath->AWGN->SCO->CFO (wrong).
FIX: SCO fractional-clock phase now persists across apply() calls.

Reference: phase2 (2).md — Section 8
"""

import numpy as np
import zmq
import struct
import config as cfg
from scipy.signal import lfilter

# ─────────────────────────────────────────────────────────────────────────────
# Channel Impairment Engine
# ─────────────────────────────────────────────────────────────────────────────

class ChannelModel:
    """
    Configurable baseband channel impairment model.

    Parameters
    ----------
    noise_voltage : float
        AWGN noise standard deviation (linear amplitude, not dB).
    taps : array-like of complex, optional
        Multipath FIR taps. Default: single-tap (flat channel).
    cfo_hz : float
        Carrier frequency offset in Hz.
    sco_ppm : float
        Sample clock offset in parts-per-million. Positive = RX faster than TX.
    fs : float
        Sample rate (Hz).
    seed : int
        RNG seed for reproducible simulations.
    """

    def __init__(self, noise_voltage: float = 0.01,
                 taps=None,
                 cfo_hz: float = 0.0,
                 sco_ppm: float = 0.0,
                 fs: float = cfg.FS_FFT,
                 seed: int = 42):
        self.noise_voltage  = noise_voltage
        self.taps           = np.array([1.0 + 0j] if taps is None else taps,
                                       dtype=np.complex64)
        self.cfo_hz         = cfo_hz
        self.sco_ppm        = sco_ppm
        self.fs             = fs
        self._rng           = np.random.default_rng(seed)
        self._sample_count  = 0   # running absolute sample index for CFO rotation
        # SCO state: fractional clock accumulator so phase is continuous across calls
        self._sco_phase     = 0.0   # fractional output sample offset

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply all configured channel impairments in spec order:
        multipath -> CFO -> SCO -> AWGN.
        """
        x = x.astype(np.complex64)

        # 1. Multipath convolution (FIR)
        if len(self.taps) > 1:
            x = lfilter(self.taps, [1.0], x).astype(np.complex64)

        # 2. CFO — phase rotation (continuous across calls via _sample_count)
        n = np.arange(self._sample_count,
                      self._sample_count + len(x), dtype=np.float64)
        x = (x * np.exp(1j * 2 * np.pi * self.cfo_hz / self.fs * n)
             ).astype(np.complex64)
        self._sample_count += len(x)

        # 3. SCO — resample by (1 + sco_ppm*1e-6) using linear interpolation.
        #    FIX: use _sco_phase to maintain fractional-clock continuity across calls.
        if abs(self.sco_ppm) > 0:
            rate  = 1.0 + self.sco_ppm * 1e-6   # input samples per output sample
            t_in  = np.arange(len(x), dtype=np.float64)
            # Build output sample times starting from _sco_phase (carry-over).
            # Upper bound = (len-1) + rate ensures at least len(x) output points
            # for positive SCO (rate > 1). np.interp clamps t > len-1 to x[-1].
            out_pts = []
            t = self._sco_phase
            while t <= float(len(x) - 1) + rate:
                if t > float(len(x) - 1) and len(out_pts) >= len(t_in):
                    break  # already have enough samples; don't over-produce
                out_pts.append(t)
                t += rate

            if out_pts:
                t_out = np.array(out_pts, dtype=np.float64)
                x_r   = (np.interp(t_out, t_in, x.real) +
                         1j * np.interp(t_out, t_in, x.imag)).astype(np.complex64)
                x     = x_r
                # Issue 4 fix: carry-over = how far into the NEXT input block
                # the next output sample falls.  t_out[-1]+rate is the next
                # output sample time; subtract len(x_original) to get its
                # offset relative to the start of the next input block.
                self._sco_phase = float(t_out[-1]) + rate - len(t_in)
            else:
                # No output samples produced (rate > remaining samples)
                self._sco_phase = self._sco_phase - len(x)


        # 4. AWGN (added last, after all deterministic distortions)
        noise = (self._rng.standard_normal(len(x)) +
                 1j * self._rng.standard_normal(len(x)))
        x     = x + (self.noise_voltage * noise / np.sqrt(2)).astype(np.complex64)

        return x


# ─────────────────────────────────────────────────────────────────────────────
# ZMQ Bridge (simulation mode)
# ─────────────────────────────────────────────────────────────────────────────

_HEADER_FMT  = "<Qd"         # uint64 seq_num + float64 timestamp = 16 bytes
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)


def _pack_message(seq: int, ts: float, samples: np.ndarray) -> bytes:
    header = struct.pack(_HEADER_FMT, seq, ts)
    return header + samples.astype(np.complex64).tobytes()


def _unpack_message(msg: bytes):
    header  = msg[:_HEADER_SIZE]
    seq, ts = struct.unpack(_HEADER_FMT, header)
    samples = np.frombuffer(msg[_HEADER_SIZE:], dtype=np.complex64)
    return seq, ts, samples


def run_bridge(channel: ChannelModel,
               pull_addr: str = cfg.ZMQ_TX_ADDR,
               push_addr: str = cfg.ZMQ_RX_ADDR,
               ant1_push_addr: str = None):
    """
    Bridge loop: PULL from TX, apply channel impairments, PUSH to RX.

    BUG 3 FIX — Parallel antenna topology:
    If ant1_push_addr is given, the CLEAN (pre-impairment) TX samples are
    also forwarded on that socket.  rx_sim then applies its own _ch_ant1
    to those clean samples, giving the correct topology:

        TX: s -> CH_ant0 -> ZMQ_RX_ADDR  (impaired, ant0)
        TX: s -> ZMQ_ANT1_ADDR           (clean, for rx_sim to impair)

    Without ant1_push_addr, the bridge behaves exactly as before.
    """
    ctx  = zmq.Context()
    pull = ctx.socket(zmq.PULL)
    pull.connect(pull_addr)

    push = ctx.socket(zmq.PUSH)
    push.bind(push_addr)

    push_ant1 = None
    if ant1_push_addr:
        push_ant1 = ctx.socket(zmq.PUSH)
        push_ant1.bind(ant1_push_addr)
        print(f"[channel_bridge] Ant-1 clean TX forwarding on {ant1_push_addr}")

    print(f"[channel_bridge] Listening on {pull_addr}, forwarding to {push_addr}")
    print(f"  noise_voltage={channel.noise_voltage}  cfo_hz={channel.cfo_hz}"
          f"  sco_ppm={channel.sco_ppm}  taps={channel.taps}")

    try:
        while True:
            msg              = pull.recv()
            seq, ts, samples = _unpack_message(msg)
            impaired         = channel.apply(samples)
            push.send(_pack_message(seq, ts, impaired))
            # Forward CLEAN samples for ant-1 (parallel topology)
            if push_ant1 is not None:
                push_ant1.send(_pack_message(seq, ts, samples))
    except KeyboardInterrupt:
        print("[channel_bridge] Shutting down.")
    finally:
        pull.close()
        push.close()
        if push_ant1:
            push_ant1.close()
        ctx.term()



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Channel bridge (simulation mode)")
    parser.add_argument("--noise",    type=float, default=0.01,
                        help="AWGN noise voltage (linear amplitude)")
    parser.add_argument("--cfo",      type=float, default=0.0,
                        help="CFO in Hz")
    parser.add_argument("--sco",      type=float, default=0.0,
                        help="SCO in ppm")
    parser.add_argument("--taps",     type=str,   default="1+0j",
                        help="Comma-separated complex taps, e.g. '1+0j,0.3+0.1j'")
    args = parser.parse_args()

    taps = [complex(t.strip()) for t in args.taps.split(',')]
    ch   = ChannelModel(noise_voltage=args.noise,
                        taps=taps,
                        cfo_hz=args.cfo,
                        sco_ppm=args.sco)
    run_bridge(ch)
