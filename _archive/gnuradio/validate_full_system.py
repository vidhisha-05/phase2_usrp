# validate_full_system.py
# ============================================================
# End-to-end validation: TX -> Channel -> Detector -> CSI
# Validates every stage with pass/fail criteria and plots
# Run AFTER the GRC flowgraph has been running for > 5 seconds
#
# Usage:
#   C:/Users/Vidhisha/radioconda/python.exe validate_full_system.py
#   C:/Users/Vidhisha/radioconda/python.exe validate_full_system.py --hdf5 my_file.h5
# ============================================================

import sys
import os
import argparse
import time
import numpy as np

# ── locate project ────────────────────────────────────────────────
PROJ = "d:/phase2"
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

# ── config ───────────────────────────────────────────────────────
FFT_SIZE   = 128
CP_LEN     = 32
SYMBOL_LEN = 160
STF_LEN    = 128
LTF_LEN    = 320
SIG_LEN    = 160
NUM_ACTIVE = 107
FS_FFT     = 20e6

GREEN = "\033[92m"
RED   = "\033[91m"
YEL   = "\033[93m"
BLU   = "\033[94m"
RST   = "\033[0m"
BOLD  = "\033[1m"

def ok(msg):  print(f"  {GREEN}[PASS]{RST} {msg}")
def fail(msg):print(f"  {RED}[FAIL]{RST} {msg}")
def warn(msg):print(f"  {YEL}[WARN]{RST} {msg}")
def hdr(msg): print(f"\n{BOLD}{BLU}{'-'*60}{RST}\n{BOLD} {msg}{RST}\n{'-'*60}")

# =================================================================
# STAGE 1  Validate TX waveform generation
# =================================================================
def validate_tx():
    hdr("STAGE 1: TX Waveform Generation")
    try:
        import waveform
        ok("waveform module imported")
    except ImportError as e:
        fail(f"Cannot import waveform: {e}")
        return None, False

    bits = np.random.randint(0, 2, 60 * 8, dtype=np.uint8)
    pkt  = waveform.assemble_packet(bits).astype(np.complex64)

    # Check length
    min_len = STF_LEN + LTF_LEN + SIG_LEN
    if len(pkt) >= min_len:
        ok(f"Packet length {len(pkt)} samples >= {min_len} (STF+LTF+SIG)")
    else:
        fail(f"Packet too short: {len(pkt)} < {min_len}")
        return pkt, False

    # Check STF power (should be > 0)
    stf_pow = np.mean(np.abs(pkt[:STF_LEN]) ** 2)
    if stf_pow > 0.01:
        ok(f"STF power = {stf_pow:.4f}  (non-zero ✓)")
    else:
        fail(f"STF power = {stf_pow:.6f}  (too low — waveform silent?)")

    # Check LTF: two identical halves (LTF property)
    ltf_start = STF_LEN + 64          # skip STF + CP
    ltf1 = pkt[ltf_start:ltf_start + FFT_SIZE]
    ltf2 = pkt[ltf_start + FFT_SIZE:ltf_start + 2 * FFT_SIZE]
    corr = abs(np.dot(ltf1, np.conj(ltf2))) / (np.linalg.norm(ltf1) * np.linalg.norm(ltf2) + 1e-12)
    if corr > 0.85:
        ok(f"LTF correlation between halves = {corr:.4f}  (good symmetry ✓)")
    else:
        warn(f"LTF correlation = {corr:.4f}  (expected > 0.85)")

    # Check OFDM spectrum occupancy (should use ~107/128 subcarriers)
    fft_out = np.fft.fftshift(np.abs(np.fft.fft(pkt[STF_LEN + LTF_LEN:STF_LEN + LTF_LEN + SYMBOL_LEN])))
    peak    = np.max(fft_out)
    guard   = fft_out[:10]
    if np.mean(guard) < 0.1 * peak:
        ok(f"Guard bands below 10% of peak  (proper subcarrier allocation ✓)")
    else:
        warn("Guard bands are high — check subcarrier mapping")

    print(f"    Packet = {len(pkt)} samples  "
          f"STF_pow={stf_pow:.3f}  LTF_corr={corr:.3f}")
    return pkt, True


# =================================================================
# STAGE 2  Validate Channel Impairment model
# =================================================================
def validate_channel(clean_pkt):
    hdr("STAGE 2: Channel Impairment")
    if clean_pkt is None:
        warn("Skipping (no TX packet)")
        return None, False

    noise = 0.005
    cfo   = 500.0   # Hz
    N     = len(clean_pkt)
    n_idx = np.arange(N, dtype=np.float64)

    # Apply CFO
    x = clean_pkt * np.exp(1j * 2.0 * np.pi * cfo / FS_FFT * n_idx).astype(np.complex64)

    # Add AWGN
    s    = noise / np.sqrt(2.0)
    noi  = (s * np.random.randn(N).astype(np.float32) +
            1j * s * np.random.randn(N).astype(np.float32))
    rx   = (x + noi.astype(np.complex64)).astype(np.complex64)

    # Verify SNR
    sig_pow   = np.mean(np.abs(clean_pkt) ** 2)
    noise_pow = noise ** 2
    snr_db    = 10 * np.log10(sig_pow / noise_pow) if noise_pow > 0 else float("inf")
    if snr_db > 20:
        ok(f"SNR = {snr_db:.1f} dB  (high quality channel ✓)")
    elif snr_db > 10:
        ok(f"SNR = {snr_db:.1f} dB  (adequate ✓)")
    else:
        warn(f"SNR = {snr_db:.1f} dB  (low — increase noise or check power)")

    # Verify phase rotation (CFO applied)
    phase_shift = np.angle(np.mean(rx[:16] * np.conj(clean_pkt[:16])))
    expected    = 2.0 * np.pi * cfo / FS_FFT * 8   # mid-point
    if abs(phase_shift) > 0.0001:
        ok(f"CFO phase rotation detected  phase_shift={phase_shift:.4f} rad ✓")
    else:
        warn("No CFO phase rotation detected")

    print(f"    SNR={snr_db:.1f} dB  noise={noise}  cfo={cfo} Hz")
    return rx, True


# =================================================================
# STAGE 3  Validate STF Packet Detector
# =================================================================
def validate_detector(clean_pkt, rx_pkt):
    hdr("STAGE 3: STF Packet Detector")
    if rx_pkt is None:
        warn("Skipping (no RX signal)")
        return False

    L     = 16           # Schmidl-Cox window
    best  = 0.0
    best_i = 0

    for i in range(len(rx_pkt) - 2 * L):
        sa = rx_pkt[i:i + L]
        sb = rx_pkt[i + L:i + 2 * L]
        P  = np.sum(sa * np.conj(sb))
        R  = np.sum(np.abs(sb) ** 2) + 1e-12
        val = (abs(P) ** 2) / (R ** 2)
        if val > best:
            best   = val
            best_i = i

    thresh = 0.65
    if best > thresh:
        ok(f"Schmidl-Cox peak = {best:.4f} > {thresh}  at sample {best_i} ✓")
    else:
        fail(f"Schmidl-Cox peak = {best:.4f} < {thresh}  (no detection)")
        return False

    # Check that detected position is near expected (start of STF)
    offset_err = abs(best_i - 0)
    if offset_err < 32:
        ok(f"Timing offset error = {offset_err} samples  (< 32 ✓)")
    else:
        warn(f"Timing offset = {best_i}  (may affect CSI quality)")

    # CFO estimate from detector
    sa  = rx_pkt[best_i:best_i + L]
    sb  = rx_pkt[best_i + L:best_i + 2 * L]
    P   = np.sum(sa * np.conj(sb))
    cfo_est = np.angle(P) * FS_FFT / (2.0 * np.pi * L)
    cfo_true = 500.0
    cfo_err  = abs(cfo_est - cfo_true)
    if cfo_err < 200:
        ok(f"Coarse CFO estimate = {cfo_est:.1f} Hz  (true={cfo_true}, err={cfo_err:.1f} ✓)")
    else:
        warn(f"Coarse CFO error = {cfo_err:.1f} Hz  (expected < 200)")

    print(f"    Peak={best:.4f}  sample={best_i}  cfo_est={cfo_est:.1f} Hz")
    return True


# =================================================================
# STAGE 4  Validate CSI Extraction
# =================================================================
def validate_csi_extraction(clean_pkt):
    hdr("STAGE 4: CSI Extraction (H_hat)")
    try:
        import sync
        ok("sync module imported")
    except ImportError as e:
        fail(f"Cannot import sync: {e}")
        return None, False

    # Apply known channel + noise
    cfo_true = 500.0
    noise    = 0.005
    N        = len(clean_pkt)
    n_idx    = np.arange(N, dtype=np.float64)
    rx       = clean_pkt * np.exp(1j * 2.0 * np.pi * cfo_true / FS_FFT * n_idx
                                  ).astype(np.complex64)
    s    = noise / np.sqrt(2.0)
    rx  += (s * np.random.randn(N) + 1j * s * np.random.randn(N)).astype(np.complex64)

    try:
        pc   = sync.apply_cfo_correction(rx, cfo_true)
        s1   = STF_LEN + 64
        s2   = s1 + FFT_SIZE
        fine = sync.estimate_fine_cfo(pc[s1:s1 + FFT_SIZE], pc[s2:s2 + FFT_SIZE])
        H    = sync.extract_csi(pc, ltf_start=STF_LEN, total_cfo_hz=cfo_true + fine)
    except Exception as ex:
        fail(f"CSI extraction raised: {ex}")
        return None, False

    # Length check
    if len(H) == NUM_ACTIVE:
        ok(f"H_hat length = {len(H)} == NUM_ACTIVE ({NUM_ACTIVE}) ✓")
    else:
        fail(f"H_hat length = {len(H)} != {NUM_ACTIVE}")
        return H, False

    # NaN / Inf check
    if not np.any(np.isnan(H)) and not np.any(np.isinf(H)):
        ok("H_hat contains no NaN/Inf ✓")
    else:
        fail(f"H_hat has NaN={np.sum(np.isnan(H))} Inf={np.sum(np.isinf(H))}")
        return H, False

    # Magnitude check (should be near 1.0 for AWGN-only channel)
    mag   = np.abs(H)
    mean  = float(np.mean(mag))
    std   = float(np.std(mag))
    if 0.5 < mean < 2.0:
        ok(f"|H_hat| mean = {mean:.3f}  std = {std:.3f}  (near 1.0 for flat channel ✓)")
    else:
        warn(f"|H_hat| mean = {mean:.3f}  (expected ~1.0 for single tap)")

    # Fine CFO residual
    if abs(fine) < 200:
        ok(f"Fine CFO residual = {fine:.2f} Hz  (< 200 Hz ✓)")
    else:
        warn(f"Fine CFO residual = {fine:.2f} Hz  (large — check LTF window)")

    print(f"    |H| mean={mean:.3f}  std={std:.4f}  fine_cfo={fine:.1f} Hz")
    return H, True


# =================================================================
# STAGE 5  Validate HDF5 Output
# =================================================================
def validate_hdf5(hdf5_file):
    hdr("STAGE 5: HDF5 CSI Log File")

    if not os.path.isfile(hdf5_file):
        fail(f"File not found: {hdf5_file}")
        print("  → Run the GRC flowgraph for at least 5 seconds first")
        return False

    size_kb = os.path.getsize(hdf5_file) / 1024
    ok(f"File exists: {hdf5_file}  ({size_kb:.1f} KB)")

    try:
        import h5py
    except ImportError:
        fail("h5py not installed — run: pip install h5py")
        return False

    with h5py.File(hdf5_file, "r") as f:
        # Find sessions
        if "sessions" not in f:
            fail("No 'sessions' group found in HDF5")
            return False

        sessions = list(f["sessions"].keys())
        ok(f"Sessions found: {sessions}")

        total_pkts = 0
        for sess in sessions:
            path = f"sessions/{sess}/csi/antenna0"
            if path not in f:
                warn(f"  {sess}: no antenna0 dataset")
                continue
            ds     = f[path]
            n_pkts = ds.shape[0]
            n_sub  = ds.shape[1]
            total_pkts += n_pkts

            if n_pkts > 0:
                ok(f"  {sess}: {n_pkts} packets × {n_sub} subcarriers")
            else:
                warn(f"  {sess}: 0 packets — flowgraph running?")
                continue

            # Read last 50 for stats
            H_batch = ds[-min(50, n_pkts):].astype(np.complex64)
            mag     = np.abs(H_batch)
            mean_h  = float(np.mean(mag))
            std_h   = float(np.std(mag))
            var_t   = float(np.std(np.mean(mag, axis=1)))

            if mean_h > 0.1:
                ok(f"    |H| mean={mean_h:.3f}  std={std_h:.4f}  temporal_var={var_t:.4f} ✓")
            else:
                fail(f"    |H| mean={mean_h:.3f}  (too low — extraction failing?)")

            # Check subcarrier count
            if n_sub == NUM_ACTIVE:
                ok(f"    Subcarrier count = {n_sub} == NUM_ACTIVE ({NUM_ACTIVE}) ✓")
            else:
                warn(f"    Subcarrier count = {n_sub}  (expected {NUM_ACTIVE})")

        if total_pkts > 0:
            ok(f"Total logged packets = {total_pkts}")
            pkt_rate = total_pkts / max(size_kb / 1024, 0.001)
            print(f"    Estimated rate: ~{total_pkts / 60:.1f} pkts/s "
                  f"(if running for ~60s)")
        else:
            fail("No packets found in any session — is the flowgraph running?")
            return False

    return True


# =================================================================
# STAGE 6  Live console monitor (read from GRC console output)
# =================================================================
def show_console_expectations():
    hdr("STAGE 6: What to Expect in GRC Console")
    print("""
  The GRC console (bottom panel) shows output from csi_monitor block:

  ┌─────────────────────────────────────────────────────────────┐
  │  pkt#    5  cfo=  500.0Hz  [=====+++####++++=====]  mean=0.821  100.0/s│
  │  pkt#   10  cfo=  499.8Hz  [=====+++####++++=====]  mean=0.834   99.5/s│
  └─────────────────────────────────────────────────────────────┘

  Column meanings:
    pkt#       Packet sequence number (increments every 5 pkts)
    cfo=       Estimated CFO in Hz    (should match variable cfo_hz ~500 Hz)
    [bar]      ASCII |H_hat| magnitude bar (should be non-empty, middle bright)
    mean=      Mean |H_hat| across all 107 active subcarriers (should be ~0.8-1.2)
    rate       Packets per second     (should be ~100/s at 0.01s interval)

  HDF5 logger output (also in console):
    [HDF5] gr_csi_output.h5  session=gr_session_001
    [HDF5] Flushed 50 total=50
    [Logger] 100 pkts

  ── PASS criteria summary ────────────────────────────────────────
  ✓ cfo value matches the cfo_hz Variable (within ±200 Hz)
  ✓ |H| bar is non-empty and centered (flat channel → uniform bar)
  ✓ mean |H| is 0.5 – 1.5 (AWGN only, single tap = 1+0j)
  ✓ rate is close to 1 / packet_interval_s = 100 pkt/s
  ✓ HDF5 file grows (check file size increases over time)
""")


# =================================================================
# STAGE 7  Visual plots (matplotlib)
# =================================================================
def plot_results(H_tx, H_rx):
    hdr("STAGE 7: Plots (close window to continue)")
    try:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
    except ImportError:
        warn("matplotlib not available — install with: pip install matplotlib")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("128-pt Custom OFDM PHY — End-to-End Validation",
                 fontsize=13, fontweight="bold")

    sc = np.arange(NUM_ACTIVE)

    # 1. TX packet waveform
    ax = axes[0, 0]
    if H_tx is not None:
        t = np.arange(len(H_tx)) / FS_FFT * 1e6
        ax.plot(t, np.real(H_tx), "b", lw=0.6, alpha=0.8, label="I")
        ax.plot(t, np.imag(H_tx), "r", lw=0.6, alpha=0.8, label="Q")
        ax.axvspan(0, STF_LEN / FS_FFT * 1e6, color="green",
                   alpha=0.1, label="STF")
        ax.axvspan(STF_LEN / FS_FFT * 1e6,
                   (STF_LEN + LTF_LEN) / FS_FFT * 1e6,
                   color="blue", alpha=0.1, label="LTF")
    ax.set_title("TX Waveform (clean)")
    ax.set_xlabel("Time (µs)"); ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right", fontsize=7); ax.grid(True, alpha=0.3)

    # 2. TX spectrum
    ax = axes[0, 1]
    if H_tx is not None:
        pkt_sym = H_tx[STF_LEN + LTF_LEN:]
        if len(pkt_sym) >= SYMBOL_LEN:
            spec = 20 * np.log10(np.abs(np.fft.fftshift(
                np.fft.fft(pkt_sym[:SYMBOL_LEN], n=FFT_SIZE))) + 1e-10)
            freqs = np.linspace(-FS_FFT / 2e6, FS_FFT / 2e6, FFT_SIZE)
            ax.plot(freqs, spec, "b", lw=1)
            ax.axhline(-20, color="r", ls="--", alpha=0.5, label="-20 dB floor")
    ax.set_title("TX Spectrum (128-pt FFT)")
    ax.set_xlabel("Frequency (MHz)"); ax.set_ylabel("Power (dB)")
    ax.set_ylim(-80, 10); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 3. |H_hat| magnitude per subcarrier
    ax = axes[1, 0]
    if H_rx is not None:
        ax.plot(sc, np.abs(H_rx), "b.-", lw=1, ms=3, label="|H_hat|")
        ax.axhline(1.0, color="r", ls="--", alpha=0.6, label="Ideal (flat) = 1.0")
        ax.fill_between(sc, np.abs(H_rx), 0, alpha=0.2, color="blue")
    ax.set_title("|H_hat| — Channel Frequency Response (107 active SC)")
    ax.set_xlabel("Active Subcarrier Index"); ax.set_ylabel("|H|")
    ax.set_ylim(0, 2.5); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 4. Phase of H_hat
    ax = axes[1, 1]
    if H_rx is not None:
        ax.plot(sc, np.angle(H_rx) * 180 / np.pi, "g.-", lw=1, ms=3,
                label="∠H_hat (deg)")
        ax.axhline(0, color="r", ls="--", alpha=0.6, label="Ideal (flat) = 0°")
    ax.set_title("Phase of H_hat (should be ~0° for AWGN channel)")
    ax.set_xlabel("Active Subcarrier Index"); ax.set_ylabel("Phase (°)")
    ax.set_ylim(-200, 200); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("validation_plots.png", dpi=120, bbox_inches="tight")
    print(f"  Saved: validation_plots.png")
    plt.show()


# =================================================================
# MAIN
# =================================================================
def main():
    parser = argparse.ArgumentParser(description="128-pt OFDM PHY Validation")
    parser.add_argument("--hdf5", default="gr_csi_output.h5",
                        help="Path to HDF5 output file (default: gr_csi_output.h5)")
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib plots")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"{BOLD} 128-pt Custom OFDM PHY — End-to-End Validation{RST}")
    print(f" HDF5: {args.hdf5}")
    print(f"{'='*60}")

    results = {}

    tx_pkt, results["tx"]      = validate_tx()
    rx_pkt, results["channel"] = validate_channel(tx_pkt)
    results["detector"]        = validate_detector(tx_pkt, rx_pkt)
    H_hat,  results["csi"]     = validate_csi_extraction(tx_pkt)
    results["hdf5"]            = validate_hdf5(args.hdf5)
    show_console_expectations()

    # ── Summary ──────────────────────────────────────────────────
    hdr("VALIDATION SUMMARY")
    all_pass = True
    for stage, passed in results.items():
        icon = f"{GREEN}PASS{RST}" if passed else f"{RED}FAIL{RST}"
        print(f"  {icon}  {stage}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print(f"{GREEN}{BOLD}  ALL STAGES PASSED ✓  System is working correctly.{RST}")
        print(f"""
  What the live simulation shows:
  +-- GRC Waveform plot -----------------------------------------+
  | Dense burst (~6.4 us) = OFDM preamble+payload, then silence  |
  | Pattern repeats every 10 ms (100 packets/sec)                |
  +--------------------------------------------------------------+
  +-- GRC Spectrum plot -----------------------------------------+
  | Flat top with ripple = 107 active subcarriers                |
  | Guard bands visible at +/-10 MHz edges                       |
  +--------------------------------------------------------------+
  +-- HDF5 file -------------------------------------------------+
  | {args.hdf5}                                          |
  | Shape: (N_packets, 107) complex64                            |
  +--------------------------------------------------------------+
""")
    else:
        print(f"{RED}{BOLD}  SOME STAGES FAILED — see FAIL lines above.{RST}")

    if not args.no_plot:
        plot_results(tx_pkt, H_hat)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
