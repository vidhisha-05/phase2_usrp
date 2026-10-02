"""
validate_stage4.py  --  STAGE 4 VALIDATION
2-Channel Alignment + Resampler + Phase Stability
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg
import waveform
from channel_bridge import ChannelModel
from sync import sync_packet
from waveform import upsample_25to20, downsample_20to25
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

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*54)
print("  STAGE 4 -- Resampler + 2-Channel Alignment")
print("="*54 + "\n")

# 4a. Resampler round-trip
print("  [4a] Rational resampler 20->25->20 MS/s round-trip")
for n_samples in [160, 1280, 4096]:
    x    = (np.random.randn(n_samples) + 1j*np.random.randn(n_samples)
            ).astype(np.complex64)
    y25  = downsample_20to25(x)
    y20  = upsample_25to20(y25)
    ratio = len(y20) / n_samples
    # Group-delay strip is proportionally larger for short buffers
    tol   = 0.78 if n_samples <= 160 else 0.85
    ok    = tol < ratio < 1.15
    check(f"Resample {n_samples}->25->20  ratio in [{tol:.2f},1.15]",
          ok, f"ratio={ratio:.3f}  in={n_samples} out={len(y20)}")

print()
print(f"  Resampler alignment latency: 25->20 = {waveform.make_streaming_25to20().latency_samples} output samples; "
      f"20->25 = {waveform.make_streaming_20to25().latency_samples} output samples")
print()

# 4b. STF periodicity preserved through resampler
print("  [4b] Resampling preserves STF 16-sample periodicity")
stf_20  = waveform.generate_stf()
stf_25  = downsample_20to25(stf_20)
stf_20b = upsample_25to20(stf_25)
n_use  = min(len(stf_20b) - 32, 64)   # ensure both slices fit
acorr  = []
for d in range(1, 32):
    seg_a = stf_20b[:n_use]
    seg_b = stf_20b[d:d + n_use]
    if len(seg_a) == len(seg_b) and len(seg_a) > 0:
        acorr.append(abs(np.vdot(seg_b, seg_a)))
peak_lag = int(np.argmax(acorr)) + 1
check("STF autocorr peak at lag=16 after resample",
      peak_lag == 16, f"peak at lag={peak_lag}")
print()

# 4c. 2-channel CSI alignment (direct extraction, no streaming detector)
print("  [4c] 2-channel CSI packet-index alignment")
N_PKTS = 20
ch0 = ChannelModel(noise_voltage=0.01, cfo_hz=200,  sco_ppm=0, seed=42)
ch1 = ChannelModel(noise_voltage=0.01, cfo_hz=200,  sco_ppm=0,
                    taps=[1.0+0j, 0.2-0.1j], seed=99)

from sync import extract_csi, estimate_fine_cfo, apply_cfo_correction

det_c   = PacketDetector()
h0_list = []
h1_list = []

for _ in range(N_PKTS):
    stf = waveform.generate_stf()
    ltf = waveform.generate_ltf()
    pkt = np.concatenate([stf, ltf]).astype(np.complex64)
    p0  = ch0.apply(pkt.copy())
    p1  = ch1.apply(pkt.copy())

    # Use direct CFO extraction (same as Stage 3 proven approach)
    coarse_cfo = det_c._estimate_coarse_cfo(p0[:cfg.STF_LEN])
    p0_cc = apply_cfo_correction(p0, coarse_cfo)
    p1_cc = apply_cfo_correction(p1, coarse_cfo)

    s1 = cfg.STF_LEN + 64
    s2 = s1 + cfg.FFT_SIZE
    if s2 + cfg.FFT_SIZE > len(p0_cc):
        continue

    fine_cfo  = estimate_fine_cfo(p0_cc[s1:s1+cfg.FFT_SIZE], p0_cc[s2:s2+cfg.FFT_SIZE])
    total_cfo = coarse_cfo + fine_cfo

    H0 = extract_csi(p0_cc, ltf_start=cfg.STF_LEN, total_cfo_hz=total_cfo)
    H1 = extract_csi(p1_cc, ltf_start=cfg.STF_LEN, total_cfo_hz=total_cfo)

    if (not np.any(np.isnan(H0)) and not np.any(np.isnan(H1)) and
            np.mean(np.abs(H0)) > 0.1 and np.mean(np.abs(H1)) > 0.1):
        h0_list.append(H0)
        h1_list.append(H1)

check(f"At least 15/{N_PKTS} packets aligned on both channels",
      len(h0_list) >= 15, f"aligned={len(h0_list)}")

if len(h0_list) >= 2:
    H0   = np.array(h0_list)
    H1   = np.array(h1_list)
    check("H0 and H1 shapes match", H0.shape == H1.shape,
          f"{H0.shape} vs {H1.shape}")
    corr = float(np.corrcoef(np.abs(H0).ravel(), np.abs(H1).ravel())[0, 1])
    check("Cross-channel |H| correlation < 0.99",
          corr < 0.99, f"corr={corr:.4f}")
    print(f"  Cross-channel correlation: {corr:.4f}"
          f"  (lower = more decorrelated)")
print()

# 4d. Phase stability (static scene)
print("  [4d] Phase stability across packets (static scene)")
N_STABLE = 30
ch_s   = ChannelModel(noise_voltage=0.005, cfo_hz=0, sco_ppm=0, seed=7)
det_s  = PacketDetector()
phases = []

for _ in range(N_STABLE):
    pkt   = np.concatenate([waveform.generate_stf(),
                              waveform.generate_ltf()]).astype(np.complex64)
    p_imp = ch_s.apply(pkt)
    dets  = det_s.process(p_imp)
    if dets:
        start, cfo = dets[0]
        buf  = p_imp[max(0, start):start + cfg.STF_LEN + cfg.LTF_LEN + 640]
        r    = sync_packet(buf, cfo)
        if not np.any(np.isnan(r['H_hat'])):
            phases.append(float(np.angle(r['H_hat'][0])))

if len(phases) >= 10:
    unwrapped   = np.unwrap(np.array(phases))
    phase_std   = float(np.std(unwrapped))
    phase_drift = float(np.max(unwrapped) - np.min(unwrapped))
    check("Phase std < 0.3 rad (static scene)",
          phase_std < 0.3, f"std={phase_std:.4f} rad")
    check("Phase drift < 1.0 rad over 30 pkts",
          phase_drift < 1.0, f"drift={phase_drift:.4f} rad")
    print(f"  Phase: std={phase_std:.4f} rad  drift={phase_drift:.4f} rad"
          f"  n={len(phases)} pkts")
    print("  NOTE: Re-measure phase stability on real hardware per spec Sec 7.3.")
print()

# Final
if not errors:
    print("  [PASS]  STAGE 4 COMPLETE -- all checks passed\n")
    sys.exit(0)
else:
    print(f"  [FAIL]  STAGE 4 FAILED -- {len(errors)} checks failed:")
    for e in errors:
        print(f"         * {e}")
    sys.exit(1)
