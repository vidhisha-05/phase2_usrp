"""
tx_hardware.py — TX process: USRP B210 hardware TX.

B210-ready fixes:
  - GUARD_SAMPLES zeros (at 25 MS/s) prepended to every burst so RX Schmidl-Cox
    has a clean blank reference before the STF (eliminates false-trigger-on-noise).
  - LO lock retry loop (B210 sometimes takes 200-500 ms to lock).
  - start_of_burst / end_of_burst flags correct per UHD streaming convention.
  - CLI --guard flag to override guard length.

Reference: phase2 (2).md — Sections 9, 13, 17 Step 7
"""

import time
import numpy as np
import config as cfg
import waveform
import scrambler

try:
    import uhd
except ImportError:
    raise ImportError(
        "UHD Python bindings not found. "
        "Install via: conda install -c conda-forge uhd  or build from source.")


def setup_usrp_tx(center_freq: float = cfg.USRP_CENTER_FREQ,
                  sample_rate: float = cfg.FS_HW,
                  tx_gain: float    = cfg.USRP_TX_GAIN,
                  lo_timeout_s: float = 2.0) -> 'uhd.usrp.MultiUSRP':
    """
    Configure the USRP B210 TX chain with LO lock retry.

    IMPORTANT: Verify channel/subdev mapping on your unit first:
        python -c "import uhd; u=uhd.usrp.MultiUSRP(); print(u.get_pp_string())"
    """
    usrp = uhd.usrp.MultiUSRP()
    usrp.set_tx_rate(sample_rate)
    tune_req = uhd.libpyuhd.types.tune_request(center_freq, cfg.USRP_LO_OFFSET)
    usrp.set_tx_freq(tune_req, 0)
    usrp.set_tx_gain(tx_gain, 0)
    usrp.set_tx_antenna("TX/RX", 0)

    # LO lock with retry (B210 sometimes needs up to 500 ms)
    t0 = time.monotonic()
    while True:
        time.sleep(0.05)
        if usrp.get_tx_sensor("lo_locked", 0).to_bool():
            break
        if time.monotonic() - t0 > lo_timeout_s:
            raise RuntimeError(
                f"TX LO did not lock after {lo_timeout_s:.1f}s — "
                "check center frequency and USB connection.")

    print(f"[tx_hw] TX configured: {usrp.get_tx_freq(0)/1e6:.6f} MHz  "
          f"rate={usrp.get_tx_rate(0)/1e6:.3f} MS/s  "
          f"gain={usrp.get_tx_gain(0):.1f} dB")
    return usrp


def run_tx_hardware(payload_bytes: int  = 100,
                    modulation: str     = "BPSK",
                    packet_interval_s: float = 0.01,
                    n_packets: int      = 0,
                    guard_samples: int  = cfg.GUARD_SAMPLES):
    """
    Continuously transmit packets via USRP B210.

    Each burst layout (at 25 MS/s):
        [guard_samples zeros] + [packet IQ samples]

    The guard zeros give the RX Schmidl-Cox correlator a clean blank reference
    before the STF, eliminating false-trigger-on-noise failures. Minimum
    recommended guard_samples = cfg.GUARD_SAMPLES = 1024 (40.96 µs).

    Args:
        payload_bytes:      Payload per packet (bytes).
        modulation:         "BPSK", "QPSK", or "16QAM".
        packet_interval_s:  Inter-burst gap (seconds). 0 = continuous.
        n_packets:          0 = run forever.
        guard_samples:      Zero samples prepended at 25 MS/s. Default=1024.
    """
    usrp    = setup_usrp_tx()
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args.channels = [0]
    streamer = usrp.get_tx_stream(st_args)

    guard_hw = np.zeros(guard_samples, dtype=np.complex64)

    seq = 0
    try:
        while True:
            payload_bits = np.random.randint(0, 2, payload_bytes * 8,
                                             dtype=np.uint8)

            # Build baseband packet at 20 MS/s, resample to 25 MS/s
            pkt_bb = waveform.assemble_packet(
                payload_bits,
                modulation=modulation,
                scrambler_mod=scrambler,
                encoder_mod=scrambler,
                mapper_fn=scrambler.map_bits_to_symbols,
                idle_samples=0)   # guard added below at HW rate
            pkt_hw = waveform.resample_20to25(pkt_bb)

            # Burst = guard zeros + packet IQ (all in one send for UHD contiguity)
            burst = np.concatenate([guard_hw, pkt_hw]).astype(np.complex64)

            # First send: start_of_burst
            md = uhd.types.TXMetadata()
            md.start_of_burst = True
            md.end_of_burst   = False
            md.has_time_spec  = False
            streamer.send(burst, md)

            # End-of-burst marker (single zero sample)
            md.start_of_burst = False
            md.end_of_burst   = True
            streamer.send(np.zeros(1, dtype=np.complex64), md)

            seq += 1
            print(f"[tx_hw] TX pkt #{seq}  "
                  f"pkt_len={len(pkt_hw)}  guard={guard_samples}  "
                  f"burst={len(burst)}")

            if n_packets and seq >= n_packets:
                break
            if packet_interval_s > 0:
                time.sleep(packet_interval_s)

    except KeyboardInterrupt:
        print("[tx_hw] Keyboard interrupt — ending burst.")
        md = uhd.types.TXMetadata()
        md.start_of_burst = False
        md.end_of_burst   = True
        streamer.send(np.zeros(1, dtype=np.complex64), md)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="USRP B210 TX")
    p.add_argument("--payload",   type=int,   default=100,
                   help="Payload bytes per packet")
    p.add_argument("--mod",       type=str,   default="BPSK",
                   choices=["BPSK", "QPSK", "16QAM"])
    p.add_argument("--interval",  type=float, default=0.01,
                   help="Inter-burst gap (s). 0=continuous")
    p.add_argument("--n_packets", type=int,   default=0,
                   help="Packets to send (0=infinite)")
    p.add_argument("--guard",     type=int,   default=cfg.GUARD_SAMPLES,
                   help=f"Guard zeros at 25 MS/s (default={cfg.GUARD_SAMPLES})")
    a = p.parse_args()
    run_tx_hardware(a.payload, a.mod, a.interval, a.n_packets, a.guard)
