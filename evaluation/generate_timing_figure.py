from pathlib import Path
import csv
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parent.parent
TABLE = PROJECT_DIR / "evaluation" / "tables" / "packet_timing.csv"
OUT = PROJECT_DIR / "evaluation" / "figures" / "fig_packet_timing.png"


rows = []
with TABLE.open("r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)


labels = [
    f'{r["Modulation"]} {r["Payload_B"]}B'
    for r in rows
]

burst_us = [
    float(r["Burst_us"])
    for r in rows
]

period_us = [
    float(r["Packet_period_us"])
    for r in rows
]

occupancy = [
    float(r["Occupancy_percent"])
    for r in rows
]


fig, ax = plt.subplots(figsize=(10, 5.5))

x = range(len(labels))

ax.bar(x, burst_us, label="TX burst duration")
ax.axhline(
    period_us[0],
    linestyle="--",
    linewidth=1.5,
    label="5 ms packet period",
)

ax.set_xticks(list(x))
ax.set_xticklabels(labels, rotation=25, ha="right")
ax.set_xlabel("Packet configuration")
ax.set_ylabel("Time (µs)")
ax.set_title("Packet Burst Duration within 5 ms Transmission Period")
ax.grid(axis="y", alpha=0.25)

for i, (duration, occ) in enumerate(zip(burst_us, occupancy)):
    ax.text(
        i,
        duration,
        f"{duration:.2f} µs\n({occ:.2f}%)",
        ha="center",
        va="bottom",
        fontsize=8,
    )

ax.legend()

fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"[PASS] Figure written: {OUT}")