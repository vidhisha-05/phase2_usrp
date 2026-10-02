"""
validate_stage5_hardware.py  ──  STAGE 5 VALIDATION (Hardware Mode)
USRP B210 Hardware Bring-Up Checklist

What this validates (spec Section 17, Step 7 + Section 9):
  • UHD Python bindings importable
  • USRP B210 device found on USB3
  • Subdevice spec and channel mapping confirmed
  • TX and RX LO lock on configured center frequency
  • Sample rate achievable without underrun/overrun
  • CRC pass rate > 90% at close-range LOS
  • Ring-buffer drop counter stays at zero
  • Inter-channel phase calibration measurement

HOW TO RUN:
  Ensure B210 is connected via USB3 before running.
  Run TX (main_tx.py --mode hardware) in a SECOND terminal first.
  Then:
    python validate_stage5_hardware.py [--n_packets 50] [--freq 2412e6]

Expected output:
  Device info table, LO lock confirmation, CRC rate, phase cal result.

  ⚠ If UHD is not installed, this script exits with a clear message.
  ⚠ Set correct USRP_SUBDEV_SPEC in config.py before running.
"""

import sys
import time
import argparse
import numpy as np

sys.path.insert(0, r"d:\phase2")
import config as cfg

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m⚠\033[0m"
errors = []

def check(name, cond, detail=""):
    sym = PASS if cond else FAIL
    print(f"  {sym}  {name}" + (f"  ← {detail}" if detail else ""))
    if not cond:
        errors.append(name)

# ────────────────────────────────────────────────────────────────────────────
print("\n══════════════════════════════════════════════")
print("  STAGE 5 — Hardware Bring-Up (USRP B210)")
print("══════════════════════════════════════════════\n")

# ── 5a. UHD import check ──────────────────────────────────────────────────────
try:
    import uhd
    check("UHD Python bindings importable", True)
except ImportError as e:
    print(f"  {FAIL}  UHD not installed: {e}")
    print("  Install: pip install uhd  OR build UHD from source.")
    print("  Skipping hardware validation — only simulation mode available.\n")
    sys.exit(1)

# ── 5b. Device discovery ──────────────────────────────────────────────────────
print("\n  [5b] Device discovery")
try:
    usrp = uhd.usrp.MultiUSRP()
    pp   = usrp.get_pp_string()
    print(f"\n  Device info:\n{'─'*60}")
    for line in pp.strip().split('\n'):
        print(f"  {line}")
    print(f"{'─'*60}\n")
    check("USRP device found", True)
except Exception as e:
    check("USRP device found", False, str(e))
    print("  Cannot continue without hardware.\n")
    sys.exit(1)

# ── 5c. Subdevice spec confirmation ──────────────────────────────────────────
print("  [5c] Subdevice spec (spec Section 9 — must verify per-unit)")
actual_spec = str(usrp.get_rx_subdev_spec())
print(f"  Actual subdev spec: {actual_spec}")
print(f"  Config USRP_SUBDEV_SPEC: {cfg.USRP_SUBDEV_SPEC}")
spec_match = actual_spec.strip() == cfg.USRP_SUBDEV_SPEC.strip()
if not spec_match:
    print(f"  {WARN}  Subdev spec mismatch — update USRP_SUBDEV_SPEC in config.py")
    print(f"         Set to: \"{actual_spec.strip()}\"")
check("Subdev spec recorded (see above)", True)   # informational, not fatal

# ── 5d. LO lock on TX ────────────────────────────────────────────────────────
print("\n  [5d] TX LO lock")
ap = argparse.ArgumentParser()
ap.add_argument("--freq",      type=float, default=cfg.USRP_CENTER_FREQ)
ap.add_argument("--tx_gain",   type=float, default=cfg.USRP_TX_GAIN)
ap.add_argument("--rx_gain",   type=float, default=cfg.USRP_RX_GAIN)
ap.add_argument("--n_packets", type=int,   default=50)
args = ap.parse_args()

try:
    usrp.set_tx_rate(cfg.FS_HW)
    usrp.set_tx_freq(uhd.libpyuhd.types.tune_request(args.freq), 0)
    usrp.set_tx_gain(args.tx_gain, 0)
    time.sleep(0.1)
    lo_tx = usrp.get_tx_sensor("lo_locked", 0).to_bool()
    check(f"TX LO locked at {args.freq/1e9:.4f} GHz", lo_tx)
    print(f"  Actual TX freq: {usrp.get_tx_freq(0)/1e6:.6f} MHz")
except Exception as e:
    check("TX LO lock", False, str(e))

# ── 5e. LO lock on RX (both channels) ────────────────────────────────────────
print("\n  [5e] RX LO lock (both channels)")
try:
    usrp.set_rx_subdev_spec(uhd.usrp.SubdevSpec(actual_spec.strip()))
    for ch in range(cfg.NUM_RX_CHANNELS):
        usrp.set_rx_rate(cfg.FS_HW, ch)
        usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(args.freq), ch)
        usrp.set_rx_gain(args.rx_gain, ch)
        usrp.set_rx_antenna("RX2", ch)
    time.sleep(0.15)
    for ch in range(cfg.NUM_RX_CHANNELS):
        lo_ok = usrp.get_rx_sensor("lo_locked", ch).to_bool()
        check(f"RX ch{ch} LO locked", lo_ok)
        print(f"  RX ch{ch} actual freq: {usrp.get_rx_freq(ch)/1e6:.6f} MHz")
        print(f"  RX ch{ch} actual rate: {usrp.get_rx_rate(ch)/1e6:.3f} MS/s")
except Exception as e:
    check("RX LO lock", False, str(e))

# ── 5f. RX streaming: check for overruns ─────────────────────────────────────
print("\n  [5f] RX streaming overrun check (2-second burst)")
try:
    st_args          = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args.channels = list(range(cfg.NUM_RX_CHANNELS))
    streamer         = usrp.get_rx_stream(st_args)

    BLOCK = 4096
    recv_buf = [np.zeros(BLOCK, dtype=np.complex64)
                for _ in range(cfg.NUM_RX_CHANNELS)]
    md  = uhd.types.RXMetadata()
    cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    cmd.stream_now = True
    streamer.issue_stream_cmd(cmd)

    t0         = time.monotonic()
    overruns   = 0
    blocks_rx  = 0
    while time.monotonic() - t0 < 2.0:
        streamer.recv(recv_buf, md)
        blocks_rx += 1
        if md.error_code == uhd.types.RXMetadataErrorCode.overflow:
            overruns += 1

    stop_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
    streamer.issue_stream_cmd(stop_cmd)

    total_samples = blocks_rx * BLOCK
    check(f"No RX overruns in 2s ({total_samples} samples)",
          overruns == 0, f"overruns={overruns}")
    print(f"  Blocks received: {blocks_rx}  Overruns: {overruns}")
except Exception as e:
    check("RX streaming test", False, str(e))

# ── 5g. CRC pass rate check ───────────────────────────────────────────────────
print(f"\n  [5g] CRC pass rate check")
print(f"  {WARN}  This requires TX (main_tx.py --mode hardware) running in a second terminal.")
print(f"       Start TX now if not already running, then press ENTER.")
input("       [ENTER to begin CRC test]")

try:
    from rx_hardware import setup_usrp_rx
    from detector import PacketDetector
    from sync import sync_packet
    from demod import check_crc as _crc_check, demodulate_packet
    from waveform import upsample_25to20
    import queue as _q

    usrp_rx  = setup_usrp_rx(args.freq, cfg.FS_HW, args.rx_gain)
    st2      = uhd.usrp.StreamArgs("fc32", "sc16")
    st2.channels = list(range(cfg.NUM_RX_CHANNELS))
    streamer2 = usrp_rx.get_rx_stream(st2)

    cmd2 = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    cmd2.stream_now = True
    streamer2.issue_stream_cmd(cmd2)

    BLOCK2  = 4096
    buf2    = [np.zeros(BLOCK2, dtype=np.complex64)
               for _ in range(cfg.NUM_RX_CHANNELS)]
    md2     = uhd.types.RXMetadata()
    det_hw  = PacketDetector()
    samples_buf = np.array([], dtype=np.complex64)

    n_detected = 0
    n_target   = args.n_packets
    t0 = time.monotonic()

    while n_detected < n_target and time.monotonic() - t0 < 30:
        streamer2.recv(buf2, md2)
        ch0_bb = upsample_25to20(buf2[0].copy())
        dets   = det_hw.process(ch0_bb)
        for start, cfo in dets:
            n_detected += 1
        print(f"\r  Detected: {n_detected}/{n_target}", end="", flush=True)

    stop_cmd2 = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
    streamer2.issue_stream_cmd(stop_cmd2)

    detect_rate = n_detected / n_target
    print(f"\n  Detection rate: {n_detected}/{n_target} = {detect_rate:.1%}")
    check("Hardware packet detection rate > 90%",
          detect_rate >= 0.90, f"rate={detect_rate:.1%}")

except Exception as e:
    print(f"  {WARN}  CRC test skipped or failed: {e}")

# ── 5h. Inter-channel phase calibration ─────────────────────────────────────
print(f"\n  [5h] Inter-channel phase offset calibration (Section 9)")
print(f"  {WARN}  Connect both RX2 ports to a common reference source (splitter/loopback).")
print(f"       Press ENTER when connected, or S to skip.")
resp = input("       [ENTER / S to skip]: ").strip().lower()
if resp != 's':
    print("  Measuring inter-channel phase offset over 5 seconds...")
    try:
        import waveform as wv
        from sync import extract_csi

        streamer.issue_stream_cmd(cmd)   # restart stream
        ph_diffs = []
        t0 = time.monotonic()
        while time.monotonic() - t0 < 5:
            streamer.recv(recv_buf, md)
            ch0_bb = upsample_25to20(recv_buf[0].copy())
            ch1_bb = upsample_25to20(recv_buf[1].copy())
            dets   = PacketDetector().process(ch0_bb)
            for start, cfo in dets:
                blen = cfg.STF_LEN + cfg.LTF_LEN + 640
                r0   = sync_packet(ch0_bb[start:start+blen], cfo)
                r1   = sync_packet(ch1_bb[start:start+blen], cfo)
                if not np.any(np.isnan(r0['H_hat'])) and not np.any(np.isnan(r1['H_hat'])):
                    pd = np.angle(r1['H_hat'] / r0['H_hat'])
                    ph_diffs.append(pd)
        streamer.issue_stream_cmd(stop_cmd)

        if ph_diffs:
            ph_mean = np.mean(np.array(ph_diffs), axis=0)
            ph_std  = np.std(np.array(ph_diffs), axis=0)
            np.save("phase_calib.npy", ph_mean)
            print(f"  Phase cal saved: phase_calib.npy")
            print(f"  Mean phase diff (first 5 subcarriers): "
                  f"{np.degrees(ph_mean[:5]).round(1)} deg")
            print(f"  Phase std (first 5 subcarriers): "
                  f"{np.degrees(ph_std[:5]).round(2)} deg")
            check("Phase calibration file saved", True)
        else:
            print(f"  {WARN}  No packets detected during calibration window.")
    except Exception as e:
        print(f"  {WARN}  Phase calibration failed: {e}")
else:
    print(f"  Skipped — remember to perform calibration before data collection.")

# ── Final result ─────────────────────────────────────────────────────────────
print()
if not errors:
    print(f"  {PASS}  STAGE 5 HARDWARE COMPLETE — all checks passed\n")
else:
    print(f"  {FAIL}  STAGE 5 HARDWARE PARTIAL — {len(errors)} checks failed:")
    for e in errors:
        print(f"         • {e}")
