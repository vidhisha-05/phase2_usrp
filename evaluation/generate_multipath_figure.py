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


FIG_DIR = PROJECT_DIR / "evaluation" / "figures"
SIM_DIR = PROJECT_DIR / "evaluation" / "simulations"

FIG_DIR.mkdir(parents=True, exist_ok=True)
SIM_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Same packet construction used by validate_stage2.py
# ---------------------------------------------------------------------
def build_pkt():
    stf = waveform.generate_stf()
    ltf = waveform.generate_ltf()
    return np.concatenate([stf, ltf]).astype(np.complex64)


# ---------------------------------------------------------------------
# Exact multipath profiles used by the current Stage-2 validator
# ---------------------------------------------------------------------
TAP_CASES = [
    ("Single tap", [1.0 + 0j]),
    ("2-tap", [1.0 + 0j, 0.5 + 0.1j]),
    ("3-tap", [1.0 + 0j, 0.3 - 0.2j, 0.1 + 0.3j]),
]

NOISE_VOLTAGE = 0.005


results = []

print("\n=== MULTIPATH CSI EVIDENCE GENERATION ===")
print(f"[INFO] Project directory : {PROJECT_DIR}")
print(f"[INFO] Noise voltage     : {NOISE_VOLTAGE}")
print()


for case_name, taps in TAP_CASES:

    # Same ChannelModel configuration as validate_stage2.py
    ch = ChannelModel(
        noise_voltage=NOISE_VOLTAGE,
        taps=taps,
        cfo_hz=0,
        sco_ppm=0,
    )

    pkt = build_pkt()

    H = extract_csi(
        ch.apply(pkt),
        ltf_start=cfg.STF_LEN,
        total_cfo_hz=0.0,
    )

    if np.any(np.isnan(H)):
        raise RuntimeError(
            f"NaN CSI detected for case: {case_name}"
        )

    magnitude = np.abs(H)

    mean_mag = float(np.mean(magnitude))
    std_mag = float(np.std(magnitude))
    min_mag = float(np.min(magnitude))
    max_mag = float(np.max(magnitude))

    tap_string = "; ".join(
        f"{t.real:+.3f}{t.imag:+.3f}j"
        for t in taps
    )

    results.append(
        {
            "case": case_name,
            "taps": tap_string,
            "num_taps": len(taps),
            "mean_abs_H": mean_mag,
            "std_abs_H": std_mag,
            "min_abs_H": min_mag,
            "max_abs_H": max_mag,
            "nan_present": False,
        }
    )

    print(
        f"{case_name:>10} | "
        f"taps={tap_string} | "
        f"mean|H|={mean_mag:.6f} | "
        f"std={std_mag:.6f} | "
        f"min={min_mag:.6f} | "
        f"max={max_mag:.6f}"
    )


# ---------------------------------------------------------------------
# Save raw numerical evidence
# ---------------------------------------------------------------------
csv_path = SIM_DIR / "multipath_csi_profiles.csv"

with csv_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    fieldnames = [
        "case",
        "taps",
        "num_taps",
        "mean_abs_H",
        "std_abs_H",
        "min_abs_H",
        "max_abs_H",
        "nan_present",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(results)


# ---------------------------------------------------------------------
# Plot mean |H|
#
# NOTE:
# std_abs_H is the spread across the 106 active subcarriers in
# one CSI realization. It is therefore NOT plotted as an error bar.
# ---------------------------------------------------------------------
labels = [
    r["case"]
    for r in results
]

means = np.array(
    [r["mean_abs_H"] for r in results],
    dtype=float,
)


fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(len(labels))

bars = ax.bar(
    x,
    means,
    width=0.55,
)

ax.set_xticks(x)
ax.set_xticklabels(labels)

ax.set_xlabel("Channel profile")
ax.set_ylabel("Mean estimated channel magnitude |H|")
ax.set_title("CSI Magnitude under Tested Multipath Profiles")

ax.grid(
    axis="y",
    alpha=0.25,
)

for bar, value in zip(bars, means):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.3f}",
        ha="center",
        va="bottom",
    )

fig.tight_layout()


fig_path = FIG_DIR / "fig_multipath_csi.png"

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
print("=== MULTIPATH CSI EVIDENCE COMPLETE ===")