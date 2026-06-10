import csv
from pathlib import Path

import matplotlib.pyplot as plt

LEDGER = Path("output/validation_ledger.csv")
OUT_CSV = Path("output/stage2_q1_q2_curve.csv")
OUT_PNG = Path("output/stage2_q1_q2_curve.png")
METHODS = {
    "Q2 CReFF-only official encoder train-size240": "Q2: direct Stage2",
    "Q1 Stage2-on-Stage1 train-size240": "Q1: Stage1 -> Stage2",
}

rows = []
with LEDGER.open(newline="") as f:
    for row in csv.DictReader(f):
        if row["method"] in METHODS:
            rows.append({
                "method": METHODS[row["method"]],
                "step": int(row["train_step"]),
                "J": float(row["J"]),
                "F": float(row["F"]),
                "J_and_F": float(row["J_and_F"]),
            })
rows.sort(key=lambda row: (row["method"], row["step"]))
if not rows:
    raise RuntimeError("No Q1/Q2 rows found in validation ledger")

with OUT_CSV.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["method", "step", "J", "F", "J_and_F"])
    writer.writeheader()
    writer.writerows(rows)

fig, ax = plt.subplots(figsize=(7.2, 4.6))
styles = {
    "Q1: Stage1 -> Stage2": ("#1976d2", "o"),
    "Q2: direct Stage2": ("#d1495b", "s"),
}
for method in sorted({row["method"] for row in rows}):
    points = [row for row in rows if row["method"] == method]
    color, marker = styles[method]
    ax.plot([row["step"] for row in points], [row["J_and_F"] for row in points],
            label=method, color=color, marker=marker, linewidth=2.2, markersize=6)
ax.set_xlabel("Stage 2 training step")
ax.set_ylabel("DAVIS-2017 full-val J&F")
ax.set_title("Stage 1 improves Stage 2 convergence and accuracy")
ax.grid(True, alpha=0.25)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=180)
print(f"saved {OUT_CSV}")
print(f"saved {OUT_PNG}")
