from pathlib import Path
import sys
import csv

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Use the CURRENT project directory, not the historical d:\phase2 path.
# ---------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import config as cfg
import waveform
from channel_bridge import ChannelModel
from sync import extract_csi


OUT_DIR = PROJECT_DIR / "evaluation"
FIG_DIR = OUT_DIR / "figures"
SIM_DIR = OUT_DIR / "simulations"

FIG_DIR.mkdir(parents=True, exist_ok=True)
SIM_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Same packet construction used by validate_stage2.py
# ---------------------------------------------------------------------
def build_pkt():
    stf = waveform.generate_stf()
    ltf = waveform.generate_ltf()
    return np.concatenate([stf, ltf]).astype(np.complex64)


# Same calibrated packet power used by validate_stage2.py.
_OFDM_PKT_POWER = 0.0447


# Same SNR sweep and 20 trials used by Stage 2.
SNR_VALUES = [30, 20, 15, 10]
N_TRIALS = 20


results = []


print("\n=== AWGN CSI EVIDENCE GENERATION ===")
print(f"[INFO] Project directory : {PROJECT_DIR}")
print(f"[INFO] Trials per SNR    : {N_TRIALS}")
print()


for snr_db in SNR_VALUES:

    # Same noise calibration as validate_stage2.py:
    #
    # noise_v =
    # sqrt(P_signal / 10^(SNR/10)) / sqrt(2)
    #
    # The sqrt(2) converts total complex noise power to the
    # per-component Gaussian standard deviation used by ChannelModel.
    noise_v = (
        np.sqrt(
            _OFDM_PKT_POWER /
            (10 ** (snr_db / 10.0))
        )
        / np.sqrt(2)
    )

    ch = ChannelModel(
        noise_voltage=noise_v,
        cfo_hz=0,
        sco_ppm=0,
    )

    mags = []

    for trial in range(N_TRIALS):

        pkt = build_pkt()
        pkt_noisy = ch.apply(pkt)

        H = extract_csi(
            pkt_noisy,
            ltf_start=cfg.STF_LEN,
            total_cfo_hz=0.0,
        )

        if np.any(np.isnan(H)):
            print(
                f"[WARN] SNR={snr_db} dB "
                f"trial={trial + 1}: NaN CSI"
            )
            continue

        mags.extend(np.abs(H).tolist())

    if not mags:
        raise RuntimeError(
            f"No valid CSI samples for SNR={snr_db} dB"
        )

    mags = np.asarray(mags, dtype=float)

    mean_mag = float(np.mean(mags))
    std_mag = float(np.std(mags))

    results.append(
        {
            "SNR_dB": snr_db,
            "noise_voltage": noise_v,
            "trials": N_TRIALS,
            "valid_CSI_samples": len(mags),
            "mean_abs_H": mean_mag,
            "std_abs_H": std_mag,
            "abs_H_error_from_1": abs(mean_mag - 1.0),
        }
    )

    print(
        f"SNR={snr_db:>2} dB | "
        f"noise_v={noise_v:.6f} | "
        f"mean|H|={mean_mag:.6f} | "
        f"std|H|={std_mag:.6f}"
    )


# ---------------------------------------------------------------------
# Save raw numerical evidence.
# ---------------------------------------------------------------------
csv_path = SIM_DIR / "awgn_csi_sweep.csv"

with csv_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    fieldnames = [
        "SNR_dB",
        "noise_voltage",
        "trials",
        "valid_CSI_samples",
        "mean_abs_H",
        "std_abs_H",
        "abs_H_error_from_1",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(results)


# ---------------------------------------------------------------------
# Plot mean |H| with ±1 standard-deviation error bars.
# ---------------------------------------------------------------------
x = np.array(
    [r["SNR_dB"] for r in results],
    dtype=float,
)

mean_h = np.array(
    [r["mean_abs_H"] for r in results],
    dtype=float,
)

std_h = np.array(
    [r["std_abs_H"] for r in results],
    dtype=float,
)


fig, ax = plt.subplots(figsize=(8, 5))


ax.errorbar(
    x,
    mean_h,
    yerr=std_h,
    marker="o",
    linewidth=1.5,
    capsize=4,
    label="Mean |H| ± 1σ",
)


ax.axhline(
    1.0,
    linestyle="--",
    linewidth=1.2,
    label="Ideal flat-channel |H| = 1",
)


ax.set_xlabel("SNR (dB)")
ax.set_ylabel("Estimated channel magnitude |H|")
ax.set_title("CSI Magnitude Robustness under AWGN")


ax.set_xticks(SNR_VALUES)
ax.grid(True, alpha=0.25)
ax.legend()


fig.tight_layout()

fig_path = FIG_DIR / "fig_awgn_csi_robustness.png"

fig.savefig(
    fig_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


print()
print(f"[PASS] Raw results : {csv_path}")
print(f"[PASS] Figure      : {fig_path}")
print()
print("=== AWGN CSI EVIDENCE COMPLETE ===")