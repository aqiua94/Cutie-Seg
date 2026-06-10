import argparse
import csv
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--metrics", type=Path, required=True)
parser.add_argument("--minimum", type=float, required=True)
parser.add_argument("--label", required=True)
args = parser.parse_args()
with args.metrics.open(newline="") as f:
    row = next(row for row in csv.DictReader(f) if row["video"] == "global")
value = float(row["J&F"])
print(f"{args.label}: J&F={value:.6f}; minimum={args.minimum:.6f}; delta={value-args.minimum:+.6f}")
if value <= args.minimum:
    raise SystemExit(f"gate failed: {args.label} did not exceed {args.minimum:.6f}")
