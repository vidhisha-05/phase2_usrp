"""
rx_hardware.py — Single-channel (1-TX / 1-RX) USRP B210 hardware RX pipeline.

Refactored & optimized for single-antenna Human Activity Recognition (HAR):
  - Single RX channel configuration ("A:A") on B210.
  - Digital CORDIC off-tuning (TuneRequest with lo_offset) to eliminate direct-conversion LO spike.
  - Continuous streaming acquisition into an 8M sample ring buffer (~320 ms headroom).
  - Lookahead carry-forward buffer (last HEADROOM samples) to eliminate boundary-straddling packet drops.
  - Integrated CSI phase sanitization (linear regression detrending) for single-antenna HAR.
  - Full packet decode with CRC-32 gate.

Three-thread pipeline:
  Acquisition Thread -> Ring Buffer -> Processing Loop -> CSI Queue -> CSILogger (HDF5)
"""

import time
import queue
import threading
import numpy as np
import config as cfg
from detector import PacketDetector
from sync import sync_packet, sanitize_csi_phase
from rx_sim import RingBuffer

try:
    import uhd
except ImportError:
    raise ImportError(
        "UHD Python bindings not found. "
        "Install via: conda install -c conda-forge uhd or build from source."
    )

# Packet window sizing (samples at baseband 20 MS/s)
_PKT_LEN = (
    cfg.STF_LEN
    + cfg.LTF_LEN
    + cfg.SIG_LEN
    + cfg.MAX_DATA_SYMS * cfg.SYMBOL_LEN
)
HEADROOM = _PKT_LEN + cfg.GUARD_SAMPLES_BB   # extra guard for window safety
BLOCK_SIZE = 4096    # UHD recv() block size (samples)
_CFO_LIMIT = 50_000  # Hz — operational gate (±50 kHz = 10× B210 TCXO drift ±5 kHz @ 2.4 GHz)
                     # T_NOI: FA=4.2%/block at 50 kHz (PASS <5%), 16.8% at 200 kHz (FAIL)


def setup_usrp_rx(center_freq: float = cfg.USRP_CENTER_FREQ,
                  sample_rate: float = cfg.FS_HW,
                  rx_gain: float = cfg.USRP_RX_GAIN,
                  subdev_spec: str = cfg.USRP_SUBDEV_SPEC,
                  lo_offset: float = cfg.USRP_LO_OFFSET,
                  lo_timeout_s: float = 2.0) -> 'uhd.usrp.MultiUSRP':
    """
    Configure single-channel USRP B210 RX with CORDIC off-tuning and LO lock retry.
    """
    usrp = uhd.usrp.MultiUSRP()
    usrp.set_rx_subdev_spec(uhd.usrp.SubdevSpec(subdev_spec))
    print(f"[rx_hw] Subdev spec configured: {usrp.get_rx_subdev_spec()}")

    # Single-channel RX (Channel 0)
    usrp.set_rx_rate(sample_rate, 0)

    # Direct-conversion LO leakage mitigation via CORDIC off-tuning
    tune_req = uhd.libpyuhd.types.tune_request(center_freq, lo_offset)
    usrp.set_rx_freq(tune_req, 0)
    usrp.set_rx_gain(rx_gain, 0)
    usrp.set_rx_antenna("RX2", 0)

    # LO lock retry loop
    t0 = time.monotonic()
    while True:
        time.sleep(0.05)
        if usrp.get_rx_sensor("lo_locked", 0).to_bool():
            break
        if time.monotonic() - t0 > lo_timeout_s:
            raise RuntimeError(
                f"RX LO did not lock after {lo_timeout_s:.1f}s. "
                "Check center frequency and USB connection."
            )

    print(
        f"[rx_hw] RX Channel 0 configured: "
        f"{usrp.get_rx_freq(0)/1e6:.6f} MHz  "
        f"rate={usrp.get_rx_rate(0)/1e6:.3f} MS/s  "
        f"gain={usrp.get_rx_gain(0):.1f} dB  "
        f"lo_offset={lo_offset/1e6:.1f} MHz"
    )
    return usrp


def run_rx_hardware(csi_queue: queue.Queue, n_packets: int = 0):
    """
    Single-channel RX processing loop over continuous UHD streaming data.

    CSI Record Schema (pushed to csi_queue):
        H_hat           : complex64 (NUM_ACTIVE,)   — Raw channel estimate
        H_sanitized     : complex64 (NUM_ACTIVE,)   — Phase-detrended channel estimate
        phase_sanitized : float64   (NUM_ACTIVE,)   — Sanitized phase (rad)
        phase_slope     : float                     — OLS phase slope (rad/SC)
        phase_intercept : float                     — OLS phase intercept (rad)
        rx_bits         : uint8 array or None       — Decoded payload bits
        crc_ok          : bool                      — CRC-32 verification flag
        cfo_hz          : float                     — Estimated total CFO (Hz)
        timestamp       : float                     — Host monotonic timestamp
        seq             : int                       — Sequential packet index
        dropped         : int                       — Cumulative ring-buffer drops
        ring_occ        : float                     — Ring buffer occupancy ratio
    """
    from waveform import make_streaming_25to20
    from sync import apply_cfo_correction, sco_correct_symbol
    import demod as demod_mod

    usrp = setup_usrp_rx()

    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args.channels = [0]
    streamer = usrp.get_rx_stream(st_args)

    ring_ch0 = RingBuffer(capacity=1 << 23)   # 8 M samples (~320 ms buffer)
    _stop = threading.Event()
    pkt_cnt = 0
    resampler_25to20 = make_streaming_25to20()

    # ── 1. Acquisition Thread ────────────────────────────────────────────────
    def _acquire():
        recv_buf = [np.empty(BLOCK_SIZE, dtype=np.complex64)]
        md = uhd.types.RXMetadata()
        cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        cmd.stream_now = True
        streamer.issue_stream_cmd(cmd)

        while not _stop.is_set():
            num_rx = streamer.recv(recv_buf, md)

            if md.error_code != uhd.types.RXMetadataErrorCode.none:
                if md.error_code == uhd.types.RXMetadataErrorCode.overflow:
                    print(
                        "[rx_hw] WARNING: UHD Overflow — "
                        "processing thread falling behind"
                    )
                else:
                    print(f"[rx_hw] UHD Stream Error: {md.strerror()}")
                continue

            if num_rx > 0:
                ring_ch0.write(recv_buf[0][:num_rx].copy())

        stop_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        streamer.issue_stream_cmd(stop_cmd)

    acq = threading.Thread(
        target=_acquire,
        daemon=True,
        name="uhd_acq"
    )
    acq.start()

    det0 = PacketDetector()

    # Carry-forward lookahead buffer (25 MS/s HW rate)
    look0_hw = np.array([], dtype=np.complex64)
    abs_start_bb = 0   # Absolute sample index counter at 20 MS/s BB rate

    def _decode_window(win_bb: np.ndarray, det_cfo: float):
        """Synchronize, extract CSI, sanitize phase, and decode packet payload."""
        res = sync_packet(
            win_bb,
            coarse_cfo_hz=det_cfo,
            n_data_symbols=0
        )

        H_hat = res['H_hat']

        if np.any(np.isnan(H_hat)):
            return (
                None,
                False,
                H_hat,
                res.get('H_sanitized', H_hat),
                res.get(
                    'phase_sanitized',
                    np.zeros(cfg.NUM_ACTIVE)
                ),
                0.0,
                0.0,
                det_cfo
            )

        lt = res['ltf_start']
        cfo_total = res['total_cfo']

        rx_c = apply_cfo_correction(
            win_bb,
            cfo_total,
            start_n=0
        )

        # Extract and decode the SIGNAL field.
        sig_body_start = lt + cfg.LTF_LEN + cfg.CP_LEN
        sig_body_end = sig_body_start + cfg.FFT_SIZE

        if sig_body_end > len(rx_c):
            return (
                None,
                False,
                H_hat,
                res['H_sanitized'],
                res['phase_sanitized'],
                res['phase_slope'],
                res['phase_intercept'],
                cfo_total
            )

        Y_sig = np.fft.fft(
            rx_c[sig_body_start:sig_body_end],
            n=cfg.FFT_SIZE
        ).astype(np.complex64)

        sig = demod_mod.parse_signal_field(
            Y_sig,
            H_hat
        )

        if not sig['valid']:
            return (
                None,
                False,
                H_hat,
                res['H_sanitized'],
                res['phase_sanitized'],
                res['phase_slope'],
                res['phase_intercept'],
                cfo_total
            )

        n_bytes = int(sig['n_payload_bytes'])
        mod = sig['modulation']
        n_sym = int(sig['n_data_syms'])

        max_payload_for_mod = cfg.MAX_PAYLOAD_BYTES_BY_MODULATION.get(mod)

        if max_payload_for_mod is None:
          return (
           None,
           False,
           H_hat,
           res['H_sanitized'],
           res['phase_sanitized'],
           res['phase_slope'],
           res['phase_intercept'],
           cfo_total
          )

        if n_bytes < 1 or n_bytes > max_payload_for_mod:
           return (
             None,
            False,
            H_hat,
            res['H_sanitized'],
            res['phase_sanitized'],
            res['phase_slope'],
            res['phase_intercept'],
            cfo_total
          )

        if n_sym < 1 or n_sym > cfg.MAX_DATA_SYMS:
            return (
                None,
                False,
                H_hat,
                res['H_sanitized'],
                res['phase_sanitized'],
                res['phase_slope'],
                res['phase_intercept'],
                cfo_total
            )
        ds = lt + cfg.LTF_LEN + cfg.SIG_LEN
        ffts = []
        b = 0.0

        for m in range(n_sym):
            s = ds + m * cfg.SYMBOL_LEN + cfg.CP_LEN
            e = s + cfg.FFT_SIZE

            if e > len(rx_c):
                break

            Y = np.fft.fft(
                rx_c[s:e],
                n=cfg.FFT_SIZE
            ).astype(np.complex64)

            Y, b, _ = sco_correct_symbol(
                Y,
                H_hat,
                symbol_idx=m,
                sco_b_accum=b
            )

            ffts.append(Y)

        if not ffts:
            return (
                None,
                False,
                H_hat,
                res['H_sanitized'],
                res['phase_sanitized'],
                res['phase_slope'],
                res['phase_intercept'],
                cfo_total
            )

        rx_bits, crc_ok = demod_mod.demodulate_packet(
            ffts,
            H_hat,
            modulation=mod,
            n_payload_bytes=n_bytes
        )

        return (
            rx_bits,
            crc_ok,
            H_hat,
            res['H_sanitized'],
            res['phase_sanitized'],
            res['phase_slope'],
            res['phase_intercept'],
            cfo_total
        )

    # ── 2. Main Processing Loop ──────────────────────────────────────────────
    try:
        while True:
            raw0 = ring_ch0.read(BLOCK_SIZE)

            if len(raw0) < BLOCK_SIZE // 4:
                time.sleep(0.0005)
                continue

            # Polyphase resample from 25 MS/s HW rate to 20 MS/s BB rate
            bb0 = resampler_25to20.process(raw0)

            # Prepend lookahead buffer from previous chunk
            buf0 = (
                np.concatenate([look0_hw, bb0])
                if len(look0_hw)
                else bb0
            ).astype(np.complex64)

            look_len = len(look0_hw)
            buf_abs_s = abs_start_bb - look_len

            # Packet detection on new baseband chunk
            dets = det0.process(bb0)
            abs_start_bb += len(bb0)

            # Process all detected candidates
            for abs_s_stream, det_cfo in dets:
                if abs(det_cfo) > _CFO_LIMIT:
                    continue

                rel = int(abs_s_stream) - int(buf_abs_s)

                if rel < 0 or rel + HEADROOM > len(buf0):
                    continue   # Boundary straddle — handled in next iteration

                win0 = buf0[rel: rel + HEADROOM]

                (
                    rx_bits,
                    crc_ok,
                    H0,
                    H_san,
                    phase_san,
                    slope,
                    intercept,
                    cfo_total
                ) = _decode_window(
                    win0,
                    det_cfo
                )

                record = {
                    'H_hat': H0,
                    'H_sanitized': H_san,
                    'phase_sanitized': phase_san,
                    'phase_slope': slope,
                    'phase_intercept': intercept,
                    'rx_bits': rx_bits,
                    'crc_ok': bool(crc_ok),
                    'cfo_hz': float(cfo_total),
                    'timestamp': time.monotonic(),
                    'seq': pkt_cnt,
                    'dropped': ring_ch0.dropped,
                    'ring_occ': ring_ch0.occupancy,
                }

                csi_queue.put(record)
                pkt_cnt += 1

                print(
                    f"[rx_hw] Pkt #{pkt_cnt:04d} | "
                    f"CRC={'PASS' if crc_ok else 'FAIL'} | "
                    f"CFO={cfo_total:+6.1f} Hz | "
                    f"Slope={slope:+.4f} | "
                    f"Drops={ring_ch0.dropped}"
                )

                if n_packets and pkt_cnt >= n_packets:
                    return

            # Update lookahead carry-forward buffer
            # (preserve last HEADROOM samples)
            look0_hw = (
                buf0[-HEADROOM:]
                if len(buf0) >= HEADROOM
                else buf0
            )

    except KeyboardInterrupt:
        print(
            "[rx_hw] Keyboard interrupt received — "
            "shutting down RX pipeline."
        )

    finally:
        _stop.set()
        acq.join(timeout=2.0)


if __name__ == "__main__":
    import argparse
    from logger import CSILogger

    parser = argparse.ArgumentParser(
        description="USRP B210 Single-Channel (1-TX / 1-RX) PHY RX Daemon"
    )

    parser.add_argument(
        "--n_packets",
        type=int,
        default=0,
        help="Number of packets to capture (0 = infinite)"
    )

    parser.add_argument(
        "--hdf5",
        type=str,
        default=cfg.HDF5_FILE_PATH,
        help="Path to output HDF5 file"
    )

    parser.add_argument(
        "--session_id",
        type=str,
        default="har_session_001",
        help="Session ID string"
    )

    args = parser.parse_args()

    csi_q = queue.Queue(maxsize=2000)

    logger = CSILogger(
        csi_q,
        args.hdf5,
        session_id=args.session_id,
        n_rx_channels=1
    )

    log_thread = threading.Thread(
        target=logger.run,
        daemon=True,
        name="hdf5_logger"
    )
    log_thread.start()

    run_rx_hardware(
        csi_q,
        args.n_packets
    )

    logger.stop()
    log_thread.join()