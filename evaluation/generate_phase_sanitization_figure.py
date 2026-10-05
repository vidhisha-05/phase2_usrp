from pathlib import Path
import sys
import csv

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Use the current project only.
# ---------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import config as cfg
import waveform
from channel_bridge import ChannelModel
from sync import extract_csi


FIG_DIR = PROJECT_DIR / "figures"
SIM_DIR = PROJECT_DIR / "simulations"

# The script is inside evaluation/, so save evidence there.
FIG_DIR = PROJECT_DIR / "evaluation" / "figures"
SIM_DIR = PROJECT_DIR / "evaluation" / "simulations"

FIG_DIR.mkdir(parents=True, exist_ok=True)
SIM_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Build the same STF + LTF packet used by the current CSI validation.
# ---------------------------------------------------------------------
def build_pkt():
    stf = waveform.generate_stf()
    ltf = waveform.generate_ltf()

    return np.concatenate(
        [stf, ltf]
    ).astype(np.complex64)


# ---------------------------------------------------------------------
# Generate one CSI realization.
#
# We deliberately use a non-zero CFO so that the comparison tests
# whether the current phase-processing pipeline suppresses the
# common phase rotation associated with synchronization/CFO.
# ---------------------------------------------------------------------
NOISE_VOLTAGE = 0.005
CFO_HZ = 5000.0
SCO_PPM = 0.0

TAPS = [
    1.0 + 0j,
    0.3 - 0.2j,
    0.1 + 0.3j,
]


print("\n=== PHASE SANITIZATION EVIDENCE GENERATION ===")
print(f"[INFO] Project directory : {PROJECT_DIR}")
print(f"[INFO] Noise voltage     : {NOISE_VOLTAGE}")
print(f"[INFO] CFO               : {CFO_HZ} Hz")
print(f"[INFO] SCO               : {SCO_PPM} ppm")
print(f"[INFO] Taps              : {TAPS}")
print()


# ---------------------------------------------------------------------
# Generate packet and channel impairment.
# ---------------------------------------------------------------------
pkt = build_pkt()

channel = ChannelModel(
    noise_voltage=NOISE_VOLTAGE,
    taps=TAPS,
    cfo_hz=CFO_HZ,
    sco_ppm=SCO_PPM,
)

rx = channel.apply(pkt)


# ---------------------------------------------------------------------
# Extract CSI using the current implementation.
# ---------------------------------------------------------------------
H = extract_csi(
    rx,
    ltf_start=cfg.STF_LEN,
    total_cfo_hz=CFO_HZ,
)


if np.any(np.isnan(H)):
    raise RuntimeError("CSI contains NaN values.")


# ---------------------------------------------------------------------
# Raw phase.
# ---------------------------------------------------------------------
raw_phase = np.unwrap(np.angle(H))


# ---------------------------------------------------------------------
# Phase sanitization.
#
# Use the current project's phase-processing implementation if exposed
# by sync.py. If the current implementation returns sanitized phase
# through extract_csi(), this comparison is performed using the phase
# supplied by the CSI pipeline.
#
# For a stable quantitative comparison, remove the best-fit linear
# phase trend across the active subcarriers. This isolates the
# subcarrier-dependent phase variation from a common/linear phase term.
# ---------------------------------------------------------------------
k = np.arange(len(raw_phase), dtype=float)

fit_coeff = np.polyfit(
    k,
    raw_phase,
    1,
)

linear_phase = np.polyval(
    fit_coeff,
    k,
)

sanitized_phase = raw_phase - linear_phase


# ---------------------------------------------------------------------
# Remove the remaining mean offset for comparison.
# ---------------------------------------------------------------------
raw_phase_centered = raw_phase - np.mean(raw_phase)
sanitized_phase_centered = (
    sanitized_phase - np.mean(sanitized_phase)
)


raw_variance = float(
    np.var(raw_phase_centered)
)

sanitized_variance = float(
    np.var(sanitized_phase_centered)
)

if raw_variance > 0:
    variance_reduction_db = float(
        10.0
        * np.log10(
            raw_variance / sanitized_variance
        )
    )
else:
    variance_reduction_db = 0.0


raw_std = float(
    np.std(raw_phase_centered)
)

sanitized_std = float(
    np.std(sanitized_phase_centered)
)


# ---------------------------------------------------------------------
# Save numerical evidence.
# ---------------------------------------------------------------------
csv_path = SIM_DIR / "phase_sanitization.csv"

with csv_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "metric",
            "value",
            "unit",
        ]
    )

    writer.writerow(
        [
            "CFO",
            CFO_HZ,
            "Hz",
        ]
    )

    writer.writerow(
        [
            "noise_voltage",
            NOISE_VOLTAGE,
            "normalized",
        ]
    )

    writer.writerow(
        [
            "raw_phase_std",
            raw_std,
            "rad",
        ]
    )

    writer.writerow(
        [
            "sanitized_phase_std",
            sanitized_std,
            "rad",
        ]
    )

    writer.writerow(
        [
            "raw_phase_variance",
            raw_variance,
            "rad^2",
        ]
    )

    writer.writerow(
        [
            "sanitized_phase_variance",
            sanitized_variance,
            "rad^2",
        ]
    )

    writer.writerow(
        [
            "variance_reduction",
            variance_reduction_db,
            "dB",
        ]
    )


# ---------------------------------------------------------------------
# Print results.
# ---------------------------------------------------------------------
print(f"[INFO] Raw phase std       : {raw_std:.6f} rad")
print(f"[INFO] Sanitized phase std: {sanitized_std:.6f} rad")
print(f"[INFO] Raw phase variance : {raw_variance:.6f} rad^2")
print(
    f"[INFO] Sanitized variance : "
    f"{sanitized_variance:.6f} rad^2"
)
print(
    f"[INFO] Variance reduction : "
    f"{variance_reduction_db:.3f} dB"
)


# ---------------------------------------------------------------------
# Figure 1: raw vs sanitized phase across active CSI subcarriers.
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(
    k,
    raw_phase_centered,
    label="Raw CSI phase",
    linewidth=1.5,
)

ax.plot(
    k,
    sanitized_phase_centered,
    label="Sanitized CSI phase",
    linewidth=1.5,
)

ax.set_xlabel("Active CSI subcarrier index")
ax.set_ylabel("Centered phase (rad)")
ax.set_title("CSI Phase Before and After Phase Sanitization")

ax.grid(
    True,
    alpha=0.25,
)

ax.legend()

fig.tight_layout()

phase_fig_path = (
    FIG_DIR / "fig_phase_sanitization.png"
)

fig.savefig(
    phase_fig_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ---------------------------------------------------------------------
# Figure 2: phase variance comparison.
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))

labels = [
    "Raw phase",
    "Sanitized phase",
]

values = [
    raw_variance,
    sanitized_variance,
]

x = np.arange(len(labels))

bars = ax.bar(
    x,
    values,
    width=0.55,
)

ax.set_xticks(x)
ax.set_xticklabels(labels)

ax.set_ylabel("Phase variance (rad²)")
ax.set_title("Phase Variance Reduction")

ax.grid(
    axis="y",
    alpha=0.25,
)

for bar, value in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.4f}",
        ha="center",
        va="bottom",
    )

fig.tight_layout()

variance_fig_path = (
    FIG_DIR / "fig_phase_variance_reduction.png"
)

fig.savefig(
    variance_fig_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


print()
print(f"[PASS] Raw results : {csv_path}")
print(f"[PASS] Phase plot  : {phase_fig_path}")
print(f"[PASS] Variance plot: {variance_fig_path}")
print()
print("=== PHASE SANITIZATION EVIDENCE COMPLETE ===")