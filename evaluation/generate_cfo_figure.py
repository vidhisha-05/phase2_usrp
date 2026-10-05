from pathlib import Path
import sys
import csv

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Current project only
# ---------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import config as cfg
import waveform
from channel_bridge import ChannelModel
from sync import estimate_fine_cfo, apply_cfo_correction
from detector import PacketDetector


FIG_DIR = PROJECT_DIR / "evaluation" / "figures"
SIM_DIR = PROJECT_DIR / "evaluation" / "simulations"

FIG_DIR.mkdir(parents=True, exist_ok=True)
SIM_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Same packet construction as validate_stage2.py
# ---------------------------------------------------------------------
def build_pkt():
    stf = waveform.generate_stf()
    ltf = waveform.generate_ltf()
    return np.concatenate([stf, ltf]).astype(np.complex64)


# ---------------------------------------------------------------------
# Same CFO test conditions as validate_stage2.py
# ---------------------------------------------------------------------
TRUE_CFO_VALUES = [500, 2000, 5000, 10000]

NOISE_VOLTAGE = 0.005
MAX_ALLOWED_RESIDUAL = 300.0


results = []


print("\n=== CFO CORRECTION EVIDENCE GENERATION ===")
print(f"[INFO] Project directory : {PROJECT_DIR}")
print(f"[INFO] Noise voltage     : {NOISE_VOLTAGE}")
print(f"[INFO] Allowed residual  : < {MAX_ALLOWED_RESIDUAL} Hz")
print()


for true_cfo in TRUE_CFO_VALUES:

    # Same impairment model as Stage 2.
    ch = ChannelModel(
        noise_voltage=NOISE_VOLTAGE,
        cfo_hz=true_cfo,
        sco_ppm=0,
    )

    pkt = build_pkt()
    pkt_imp = ch.apply(pkt)

    # Same detector-based coarse CFO estimate.
    det = PacketDetector()

    stf_rx = pkt_imp[:cfg.STF_LEN]

    coarse_cfo = det._estimate_coarse_cfo(stf_rx)

    # Same coarse correction.
    pkt_cc = apply_cfo_correction(
        pkt_imp,
        coarse_cfo,
    )

    # Same LTF locations as validate_stage2.py.
    s1 = cfg.STF_LEN + 64
    s2 = s1 + cfg.FFT_SIZE

    ltf1 = pkt_cc[
        s1:s1 + cfg.FFT_SIZE
    ]

    ltf2 = pkt_cc[
        s2:s2 + cfg.FFT_SIZE
    ]

    # Same fine estimator.
    fine_cfo = estimate_fine_cfo(
        ltf1,
        ltf2,
    )

    total_cfo = coarse_cfo + fine_cfo

    residual = abs(
        total_cfo - true_cfo
    )

    passed = residual < MAX_ALLOWED_RESIDUAL

    results.append(
        {
            "true_cfo_hz": true_cfo,
            "coarse_cfo_hz": coarse_cfo,
            "fine_cfo_hz": fine_cfo,
            "estimated_total_cfo_hz": total_cfo,
            "residual_hz": residual,
            "threshold_hz": MAX_ALLOWED_RESIDUAL,
            "pass": passed,
        }
    )

    print(
        f"CFO={true_cfo:>6} Hz | "
        f"coarse={coarse_cfo:>10.3f} Hz | "
        f"fine={fine_cfo:>10.3f} Hz | "
        f"total={total_cfo:>10.3f} Hz | "
        f"residual={residual:>8.3f} Hz | "
        f"{'PASS' if passed else 'FAIL'}"
    )


# ---------------------------------------------------------------------
# Save raw numerical evidence.
# ---------------------------------------------------------------------
csv_path = SIM_DIR / "cfo_correction_sweep.csv"

with csv_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    fieldnames = [
        "true_cfo_hz",
        "coarse_cfo_hz",
        "fine_cfo_hz",
        "estimated_total_cfo_hz",
        "residual_hz",
        "threshold_hz",
        "pass",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(results)


# ---------------------------------------------------------------------
# Plot residual CFO versus injected CFO.
# ---------------------------------------------------------------------
x = np.array(
    [r["true_cfo_hz"] for r in results],
    dtype=float,
)

residuals = np.array(
    [r["residual_hz"] for r in results],
    dtype=float,
)


fig, ax = plt.subplots(figsize=(8, 5))


ax.plot(
    x,
    residuals,
    marker="o",
    linewidth=1.5,
    label="Residual CFO",
)


ax.axhline(
    MAX_ALLOWED_RESIDUAL,
    linestyle="--",
    linewidth=1.2,
    label="Validation limit (300 Hz)",
)


ax.set_xlabel("Injected CFO (Hz)")
ax.set_ylabel("Residual CFO after correction (Hz)")
ax.set_title("Two-Stage CFO Correction")


ax.set_xticks(TRUE_CFO_VALUES)
ax.grid(True, alpha=0.25)
ax.legend()


fig.tight_layout()


fig_path = FIG_DIR / "fig_cfo_correction.png"

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
print("=== CFO EVIDENCE COMPLETE ===")