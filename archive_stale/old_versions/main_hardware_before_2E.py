"""
main_hardware.py -- Single-process full-duplex USRP B210 TX+RX for CSI-HAR.

F-03 FIX: tx_hardware.py and rx_hardware.py cannot run simultaneously from two
separate Python processes on one B210 - UHD raises 'device already claimed'
on the second MultiUSRP() call.

This script:
  1. Creates ONE MultiUSRP() handle for the B210.
  2. Configures TX (Port A) and RX (Port B) from that single handle.
  3. Gets a TX streamer and an RX streamer from the same usrp object.
  4. Runs the TX loop in a daemon thread.
  5. Runs the RX loop (Schmidl-Cox -> CSI -> HDF5) in the main thread.

Usage:
    python main_hardware.py [--payload 100] [--mod BPSK] [--interval 0.01]
                            [--n_packets 0] [--hdf5 csi_data.h5]
                            [--session_id har_session_001]

v7 verified parameters (config.py):
    Center freq : 2.412 GHz
    LO offset   : 8.75 MHz  (k=-56, inside lower guard. NOT 5 MHz = k=-32 active SC)
    TX gain     : 30 dB  |  RX gain : 25 dB
    TX rate     : 25 MS/s (HW) -> 20 MS/s (BB)
    RX rate     : 25 MS/s (HW) -> 20 MS/s (BB)
    CFO gate    : +/-50 kHz (FA=4.2%, PASS threshold)
"""

import argparse
import queue
import threading
import time
import numpy as np
import config as cfg
import waveform
import scrambler as scrambler_mod
from pathlib import Path

try:
    import uhd
except ImportError:
    raise ImportError(
        "UHD Python bindings not found.\n"
        "Install: conda install -c conda-forge uhd\n"
        "Or build UHD from source with Python API enabled.")

from logger import CSILogger
from detector import PacketDetector
from sync import sync_packet, apply_cfo_correction, sco_correct_symbol
from rx_hardware import RingBuffer, _CFO_LIMIT, _PKT_LEN, HEADROOM, BLOCK_SIZE
import demod as demod_mod


# =============================================================================
# 1. USRP Setup -- ONE device, TWO streamers
# =============================================================================

def setup_b210_fullduplex(
        center_freq: float = cfg.USRP_CENTER_FREQ,
        lo_offset: float   = cfg.USRP_LO_OFFSET,
        tx_gain: float     = cfg.USRP_TX_GAIN,
        rx_gain: float     = cfg.USRP_RX_GAIN,
        sample_rate: float = cfg.FS_HW,
        lo_timeout_s: float = 3.0):
    """
    Open ONE B210, configure TX (Port A) and RX (Port B), return (usrp, tx_st, rx_st).

    The B210 AD9361 chip supports simultaneous TX+RX from a single driver handle.
    Two separate processes CANNOT share one B210 -- this function is the fix.
    """
    print("[hw] Opening USRP B210 ...")
    usrp = uhd.usrp.MultiUSRP()     # ONE call, ONE device

    # TX configuration (Channel 0, Port A = TX/RX)
    usrp.set_tx_rate(sample_rate, 0)
    tune_req_tx = uhd.libpyuhd.types.tune_request(center_freq, lo_offset)
    usrp.set_tx_freq(tune_req_tx, 0)
    usrp.set_tx_gain(tx_gain, 0)
    usrp.set_tx_antenna("TX/RX", 0)

    # RX configuration (Channel 0, Port B = RX2)
    usrp.set_rx_subdev_spec(uhd.usrp.SubdevSpec(cfg.USRP_SUBDEV_SPEC))
    usrp.set_rx_rate(sample_rate, 0)
    tune_req_rx = uhd.libpyuhd.types.tune_request(center_freq, lo_offset)
    usrp.set_rx_freq(tune_req_rx, 0)
    usrp.set_rx_gain(rx_gain, 0)
    usrp.set_rx_antenna("RX2", 0)

    # LO lock -- wait for BOTH TX and RX to lock
    print("[hw] Waiting for TX+RX LO lock ...")
    t0 = time.monotonic()
    while True:
        time.sleep(0.05)
        tx_locked = usrp.get_tx_sensor("lo_locked", 0).to_bool()
        rx_locked = usrp.get_rx_sensor("lo_locked", 0).to_bool()
        if tx_locked and rx_locked:
            break
        if time.monotonic() - t0 > lo_timeout_s:
            raise RuntimeError(
                f"LO did not lock after {lo_timeout_s:.1f}s "
                f"(TX locked={tx_locked}, RX locked={rx_locked}). "
                "Check center frequency and USB 3.0 connection.")

    lo_k = -int(lo_offset / cfg.SUBCARRIER_SPACING)
    print(f"[hw] TX freq={usrp.get_tx_freq(0)/1e6:.6f} MHz  "
          f"gain={usrp.get_tx_gain(0):.1f} dB  rate={usrp.get_tx_rate(0)/1e6:.3f} MS/s")
    print(f"[hw] RX freq={usrp.get_rx_freq(0)/1e6:.6f} MHz  "
          f"gain={usrp.get_rx_gain(0):.1f} dB  rate={usrp.get_rx_rate(0)/1e6:.3f} MS/s")
    print(f"[hw] LO offset={lo_offset/1e6:.3f} MHz -> k={lo_k} (in lower guard: {lo_k not in cfg.ACTIVE_SUBCARRIERS})")

    # TX streamer
    tx_args = uhd.usrp.StreamArgs("fc32", "sc16")
    tx_args.channels = [0]
    tx_streamer = usrp.get_tx_stream(tx_args)

    # RX streamer
    rx_args = uhd.usrp.StreamArgs("fc32", "sc16")
    rx_args.channels = [0]
    rx_streamer = usrp.get_rx_stream(rx_args)

    return usrp, tx_streamer, rx_streamer


# =============================================================================
# 2. TX Thread
# =============================================================================

def tx_loop(tx_streamer,
            payload_bytes: int = 100,
            modulation: str = "BPSK",
            packet_interval_s: float = 0.01,
            n_packets: int = 0,
            stop_event: threading.Event = None):
    """
    TX loop: assemble OFDM packet, prepend guard zeros, send via UHD streamer.
    Runs in a daemon thread.
    """
    guard_hw = np.zeros(cfg.GUARD_SAMPLES, dtype=np.complex64)
    rng = np.random.default_rng(0)
    seq = 0

    try:
        while not (stop_event and stop_event.is_set()):
            payload_bits = rng.integers(0, 2, payload_bytes * 8, dtype=np.uint8)

            pkt_bb = waveform.assemble_packet(
                payload_bits,
                modulation=modulation,
                scrambler_mod=scrambler_mod,
                encoder_mod=scrambler_mod,
                mapper_fn=scrambler_mod.map_bits_to_symbols,
                idle_samples=0)
            pkt_hw = waveform.resample_20to25(pkt_bb)
            burst  = np.concatenate([guard_hw, pkt_hw]).astype(np.complex64)

            md = uhd.types.TXMetadata()
            md.start_of_burst = True
            md.end_of_burst   = False
            md.has_time_spec  = False
            tx_streamer.send(burst, md)

            md.start_of_burst = False
            md.end_of_burst   = True
            tx_streamer.send(np.zeros(1, dtype=np.complex64), md)

            seq += 1
            if seq % 20 == 0:
                print(f"[tx] Sent {seq} packets")

            if n_packets and seq >= n_packets:
                break
            if packet_interval_s > 0:
                time.sleep(packet_interval_s)

    except KeyboardInterrupt:
        pass

    # Clean EOB
    md = uhd.types.TXMetadata()
    md.start_of_burst = False
    md.end_of_burst   = True
    tx_streamer.send(np.zeros(1, dtype=np.complex64), md)
    print(f"[tx] TX loop finished after {seq} packets.")


# =============================================================================
# 3. RX Loop (main thread)
# =============================================================================

def rx_loop(rx_streamer, csi_queue: queue.Queue, n_packets: int = 0):
    """
    RX: continuous UHD stream -> ring buffer -> Schmidl-Cox -> CSI -> HDF5 queue.
    Uses the pre-configured rx_streamer from the shared B210 handle (F-03 fix).
    """
    from waveform import make_streaming_25to20

    ring         = RingBuffer(capacity=1 << 23)
    _stop        = threading.Event()
    pkt_cnt      = 0
    resampler_25to20 = make_streaming_25to20()

    def _acquire():
        recv_buf = [np.empty(BLOCK_SIZE, dtype=np.complex64)]
        md       = uhd.types.RXMetadata()
        cmd      = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        cmd.stream_now = True
        rx_streamer.issue_stream_cmd(cmd)
        while not _stop.is_set():
            n = rx_streamer.recv(recv_buf, md)
            if md.error_code != uhd.types.RXMetadataErrorCode.none:
                if md.error_code == uhd.types.RXMetadataErrorCode.overflow:
                    print("[rx] WARNING: UHD overflow")
                continue
            if n > 0:
                ring.write(recv_buf[0][:n].copy())
        rx_streamer.issue_stream_cmd(
            uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))

    acq = threading.Thread(target=_acquire, daemon=True, name="uhd_rx_acq")
    acq.start()

    det          = PacketDetector()
    look_hw      = np.array([], dtype=np.complex64)
    abs_start_bb = 0

    def _decode(win_bb, det_cfo, n_bytes=100, mod='BPSK'):
        res   = sync_packet(win_bb, coarse_cfo_hz=det_cfo, n_data_symbols=0)
        H_hat = res['H_hat']
        if np.any(np.isnan(H_hat)):
            return (None, False, H_hat,
                    res.get('H_sanitized', H_hat),
                    res.get('phase_sanitized', np.zeros(cfg.NUM_ACTIVE)),
                    0.0, 0.0, det_cfo)
        lt        = res['ltf_start']
        cfo_total = res['total_cfo']
        rx_c      = apply_cfo_correction(win_bb, cfo_total, start_n=0)
        n_sym     = demod_mod.n_data_syms_for_payload(n_bytes, mod)
        ds        = lt + cfg.LTF_LEN + cfg.SIG_LEN
        ffts, b   = [], 0.0
        for m in range(n_sym):
            s = ds + m * cfg.SYMBOL_LEN + cfg.CP_LEN
            e = s + cfg.FFT_SIZE
            if e > len(rx_c):
                break
            Y = np.fft.fft(rx_c[s:e], n=cfg.FFT_SIZE).astype(np.complex64)
            Y, b, _ = sco_correct_symbol(Y, H_hat, symbol_idx=m, sco_b_accum=b)
            ffts.append(Y)
        if not ffts:
            return (None, False, H_hat, res['H_sanitized'],
                    res['phase_sanitized'], res['phase_slope'],
                    res['phase_intercept'], cfo_total)
        rx_bits, crc_ok = demod_mod.demodulate_packet(
            ffts, H_hat, modulation=mod, n_payload_bytes=n_bytes)
        return (rx_bits, crc_ok, H_hat, res['H_sanitized'],
                res['phase_sanitized'], res['phase_slope'],
                res['phase_intercept'], cfo_total)

    print("[rx] RX loop started. Waiting for packets ...")
    try:
        while True:
            raw = ring.read(BLOCK_SIZE)
            if len(raw) < BLOCK_SIZE // 4:
                time.sleep(0.0005)
                continue

            bb  = resampler_25to20.process(raw)

            # Absolute BB coordinate of the first sample in the combined
            # [look-ahead history | current chunk] buffer.
            look_len = len(look_hw)
            buf = (np.concatenate([look_hw, bb]) if look_len else bb).astype(np.complex64)
            buf_abs_s = abs_start_bb - look_len

            dets = det.process(bb)
            abs_start_bb += len(bb)

            for abs_s, det_cfo in dets:
                if abs(det_cfo) > _CFO_LIMIT:
                    continue
                rel = int(abs_s) - int(buf_abs_s)
                if rel < 0 or rel + HEADROOM > len(buf):
                    continue
                win = buf[rel: rel + HEADROOM]
                (rx_bits, crc_ok, H0, H_san, phase_san,
                 slope, intercept, cfo_total) = _decode(win, det_cfo)

                record = {
                    'H_hat':           H0,
                    'H_sanitized':     H_san,
                    'phase_sanitized': phase_san,
                    'phase_slope':     slope,
                    'phase_intercept': intercept,
                    'rx_bits':         rx_bits,
                    'crc_ok':          bool(crc_ok),
                    'cfo_hz':          float(cfo_total),
                    'timestamp':       time.monotonic(),
                    'seq':             pkt_cnt,
                    'dropped':         ring.dropped,
                    'ring_occ':        ring.occupancy,
                }
                csi_queue.put(record)
                pkt_cnt += 1
                print(f"[rx] Pkt #{pkt_cnt:04d} | CRC={'PASS' if crc_ok else 'FAIL'} | "
                      f"CFO={cfo_total:+.1f} Hz | ring={ring.occupancy:.3f}")

                if n_packets and pkt_cnt >= n_packets:
                    return

            look_hw = buf[-HEADROOM:] if len(buf) >= HEADROOM else buf

    except KeyboardInterrupt:
        print("[rx] Keyboard interrupt.")
    finally:
        _stop.set()
        acq.join(timeout=2.0)
        print(f"[rx] Done. Total packets decoded: {pkt_cnt}")


# =============================================================================
# 4. Inline Pre-Flight Gate (fixes missing validate_pipeline import in main_rx.py)
# =============================================================================

def _preflight_gate() -> bool:
    """Run stage 1 simulation loopback before allowing hardware start."""
    print("[preflight] Running Stage 1 PHY primitive checks ...")
    import subprocess, sys
    project_dir = Path(__file__).resolve().parent

    result = subprocess.run(
        [sys.executable, 'validate_stage1.py'],
        capture_output=True,
        timeout=30,
        cwd=str(project_dir))
    ok = (result.returncode == 0)
    status = "PASS" if ok else "FAIL"
    print(f"[preflight] Stage 1 {status}")
    if not ok:
        out = result.stdout.decode(errors='replace')
        err = result.stderr.decode(errors='replace')
        print(out[-2000:] if len(out) > 2000 else out)
        print(err[-500:] if len(err) > 500 else err)
    return ok


# =============================================================================
# 5. Entry Point
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="128-pt OFDM PHY -- Full-Duplex B210 TX+RX (single process, F-03 fix)")
    p.add_argument("--payload",       type=int,   default=100)
    p.add_argument("--mod",           type=str,   default="BPSK",
                   choices=["BPSK", "QPSK", "16QAM"])
    p.add_argument("--interval",      type=float, default=0.01,
                   help="Inter-TX-burst gap (s)")
    p.add_argument("--n_packets",     type=int,   default=0,
                   help="Stop after N decoded packets. 0=infinite.")
    p.add_argument("--hdf5",          type=str,   default=cfg.HDF5_FILE_PATH)
    p.add_argument("--session_id",    type=str,   default="har_session_001")
    p.add_argument("--skip_preflight", action="store_true",
                   help="Skip simulation pre-flight (not recommended)")
    a = p.parse_args()

    if not a.skip_preflight:
        if not _preflight_gate():
            print("[main] Pre-flight FAIL. Fix simulation before connecting hardware.")
            return

    # ONE MultiUSRP() call for both TX and RX (F-03 fix)
    usrp, tx_streamer, rx_streamer = setup_b210_fullduplex()

    # Logger thread
    csi_q  = queue.Queue(maxsize=2000)
    logger = CSILogger(csi_q,
                       hdf5_path=a.hdf5,
                       session_id=a.session_id,
                       n_rx_channels=cfg.NUM_RX_CHANNELS)  # F-26: use cfg, not hardcode 2
    log_t  = threading.Thread(target=logger.run, daemon=True, name="hdf5_logger")
    log_t.start()

    # TX daemon thread
    stop_ev = threading.Event()
    tx_t = threading.Thread(
        target=tx_loop,
        args=(tx_streamer, a.payload, a.mod, a.interval, 0, stop_ev),
        daemon=True, name="tx_thread")
    tx_t.start()
    print("[main] TX thread started. Starting RX ...")

    # RX in main thread (blocks until n_packets or Ctrl+C)
    try:
        rx_loop(rx_streamer, csi_q, n_packets=a.n_packets)
    finally:
        stop_ev.set()
        tx_t.join(timeout=3.0)
        logger.stop()
        log_t.join(timeout=10.0)
        print("[main] Session complete. HDF5 saved.")


if __name__ == "__main__":
    main()
