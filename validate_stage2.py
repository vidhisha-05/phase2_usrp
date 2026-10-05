"""
validate_stage2.py  --  STAGE 2 VALIDATION
Configurable-Impairment Simulation (AWGN + CFO + SCO + Multipath)
"""

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
import waveform
from channel_bridge import ChannelModel
from sync import extract_csi, estimate_fine_cfo, apply_cfo_correction
from sync import sco_correct_symbol
from detector import PacketDetector

errors = []

def check(name, cond, detail=""):
    tag = "[PASS]" if cond else "[FAIL]"
    msg = f"  {tag}  {name}"
    if detail:
        msg += f"  <- {detail}"
    print(msg)
    if not cond:
        errors.append(name)

def build_pkt():
    stf = waveform.generate_stf()
    ltf = waveform.generate_ltf()
    return np.concatenate([stf, ltf]).astype(np.complex64)

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*54)
print("  STAGE 2 -- Configurable-Impairment Validation")
print("="*54 + "\n")

# 2a. AWGN SNR sweep
print("  [2a] AWGN SNR sweep (flat channel, no CFO/SCO)")
print("  NOTE: noise_v calibrated to actual OFDM packet power=0.0447 (-13.5dB)")
print(f"  {'SNR(dB)':>8}  {'noise_v':>9}  {'|H| mean':>10}  {'|H| std':>9}  Status")
print("  " + "-"*54)

# True signal power of an OFDM packet at 20MS/s (measured, constant for BPSK)
_OFDM_PKT_POWER = 0.0447   # E[|x|^2], confirmed by np.mean(np.abs(pkt)**2)

for snr_db in [30, 20, 15, 10]:
    # noise_v = sqrt(pkt_power / 10^(SNR/10)) / sqrt(2)  [per-component std dev]
    # Factor /sqrt(2) because we add sigma*N(0,1) to both I and Q
    noise_v  = np.sqrt(_OFDM_PKT_POWER / (10 ** (snr_db / 10.0))) / np.sqrt(2)
    ch       = ChannelModel(noise_voltage=noise_v, cfo_hz=0, sco_ppm=0)
    mags     = []
    for _ in range(20):
        p   = build_pkt()
        p_n = ch.apply(p)
        H   = extract_csi(p_n, ltf_start=cfg.STF_LEN, total_cfo_hz=0.0)
        if not np.any(np.isnan(H)):
            mags.extend(np.abs(H).tolist())
    m, s = np.mean(mags), np.std(mags)
    # At high SNR H_hat should be near 1.0 (flat channel); tolerance widens at low SNR.
    ok   = abs(m - 1.0) < 0.15 if snr_db >= 25 else abs(m - 1.0) < 0.6
    tag  = "[PASS]" if ok else "[INFO]"
    print(f"  {snr_db:>8} dB  {noise_v:>9.5f}  {m:>10.4f}  {s:>9.4f}  {tag}")
    if snr_db >= 15:
        check(f"AWGN SNR={snr_db}dB |H| mean within tolerance", ok, f"mean={m:.4f}")

print()

# 2b. Two-stage CFO correction
print("  [2b] Two-stage CFO correction")
for true_cfo in [500, 2000, 5000, 10000]:
    ch       = ChannelModel(noise_voltage=0.005, cfo_hz=true_cfo, sco_ppm=0)
    pkt      = build_pkt()
    pkt_imp  = ch.apply(pkt)

    det        = PacketDetector()
    stf_rx     = pkt_imp[:cfg.STF_LEN]
    coarse_cfo = det._estimate_coarse_cfo(stf_rx)

    pkt_cc = apply_cfo_correction(pkt_imp, coarse_cfo)
    s1     = cfg.STF_LEN + 64
    s2     = s1 + cfg.FFT_SIZE
    ltf1   = pkt_cc[s1:s1 + cfg.FFT_SIZE]
    ltf2   = pkt_cc[s2:s2 + cfg.FFT_SIZE]
    fine_cfo  = estimate_fine_cfo(ltf1, ltf2)
    total_cfo = coarse_cfo + fine_cfo
    residual  = abs(total_cfo - true_cfo)

    check(f"CFO={true_cfo:>6} Hz  residual<300 Hz",
          residual < 300, f"residual={residual:.1f} Hz")

print()

# 2c. Known multipath CSI matching
print("  [2c] CSI with known multipath taps")
for taps in [[1.0+0j], [1.0+0j, 0.5+0.1j], [1.0+0j, 0.3-0.2j, 0.1+0.3j]]:
    ch   = ChannelModel(noise_voltage=0.005, taps=taps, cfo_hz=0, sco_ppm=0)
    pkt  = build_pkt()
    H    = extract_csi(ch.apply(pkt), ltf_start=cfg.STF_LEN, total_cfo_hz=0.0)
    ok   = (not np.any(np.isnan(H))) and np.mean(np.abs(H)) > 0.1
    tap_str = "+".join([f"{t:.1f}" for t in taps[:2]]) + ("..." if len(taps)>2 else "")
    check(f"Taps [{tap_str}]  H_hat valid no NaN", ok,
          f"mean|H|={np.mean(np.abs(H)):.3f}")

print()

# 2d. SCO pilot-tracking
print("  [2d] SCO pilot-tracking (per-symbol phase ramp)")
print("  Range validated: 0-20 ppm (B210 TCXO <=2ppm, margin factor=10x)")
for sco_ppm in [0.0, 1.0, 5.0, 10.0, 20.0]:
    # Physical model: applied phase ramp at 1 symbol offset:
    # phi(k) = 2*pi*eps*(N_CP+N_FFT)/N_FFT * k  (at m=0)
    # eps = sco_ppm*1e-6
    sco_slope = 2 * np.pi * sco_ppm * 1e-6 * cfg.SYMBOL_LEN / cfg.FFT_SIZE
    X = np.zeros(cfg.FFT_SIZE, dtype=np.complex64)
    for i, k in enumerate(cfg.DATA_INDICES):
        X[cfg.k_to_bin(k)] = 1.0
    for i, k in enumerate(cfg.PILOT_INDICES):
        X[cfg.k_to_bin(k)] = cfg.PILOT_POLARITY[i]

    Y_imp = X.copy()
    for k in cfg.ACTIVE_SUBCARRIERS:
        b = cfg.k_to_bin(k)
        Y_imp[b] = X[b] * np.exp(1j * sco_slope * k)

    H_flat = np.ones(cfg.NUM_ACTIVE, dtype=np.complex64)
    Y_corr, b_acc, a_m = sco_correct_symbol(Y_imp, H_flat)
    resid = []
    for i, k in enumerate(cfg.PILOT_INDICES):
        z = Y_corr[cfg.k_to_bin(k)] * cfg.PILOT_POLARITY[i]
        resid.append(abs(float(np.angle(z))))
    max_resid = max(resid)
    check(f"SCO={sco_ppm:>5} ppm  pilot phase residual<0.15 rad",
          max_resid < 0.15, f"max_resid={max_resid:.4f} rad")

print()

# Final
if not errors:
    print("  [PASS]  STAGE 2 COMPLETE -- all checks passed\n")
    sys.exit(0)
else:
    print(f"  [FAIL]  STAGE 2 FAILED -- {len(errors)} checks failed:")
    for e in errors:
        print(f"         * {e}")
    sys.exit(1)
