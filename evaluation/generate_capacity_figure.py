from pathlib import Path
import csv
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parent.parent
TABLE = PROJECT_DIR / "evaluation" / "tables" / "packet_capacity.csv"
OUT = PROJECT_DIR / "evaluation" / "figures" / "fig_packet_capacity.png"


rows = []
with TABLE.open("r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)


modulations = [r["Modulation"] for r in rows]
payloads = [int(r["Derived_max_payload_B"]) for r in rows]


fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(modulations, payloads)

ax.set_xlabel("Modulation")
ax.set_ylabel("Maximum payload (bytes)")
ax.set_title("Maximum Supported Payload by Modulation")
ax.grid(axis="y", alpha=0.25)

for bar, value in zip(bars, payloads):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value,
        f"{value} B",
        ha="center",
        va="bottom",
    )

fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"[PASS] Figure written: {OUT}")