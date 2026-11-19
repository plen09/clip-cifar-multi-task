#!/usr/bin/env python3
"""Aggregate completed test results across seeds (mean ± sample standard deviation)."""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from src.utils import write_json

METHOD_ORDER = ["fine_only", "coarse_only", "equal", "uncertainty", "gradnorm", "fixed"]
METRICS = [
    "test_fine_acc_pct", "test_coarse_acc_pct", "test_mean_acc_pct",
    "best_validation_score_pct", "final_weight_fine", "final_weight_coarse",
]


def summarize_runs(runs_dir, out_dir):
    runs_dir, out_dir = Path(runs_dir), Path(out_dir)
    grouped = defaultdict(list)
    for path in runs_dir.glob("*/seed_*/results.json"):
        with path.open(encoding="utf-8") as handle:
            item = json.load(handle)
        grouped[item["method"]].append(item)

    if not grouped:
        print(f"No completed results found in {runs_dir}.")
        return []

    rows = []
    for method in sorted(grouped, key=lambda m: (METHOD_ORDER.index(m) if m in METHOD_ORDER else 999, m)):
        items = sorted(grouped[method], key=lambda i: i["seed"])
        row = {"method": method, "n_seeds": len(items),
               "seeds": ",".join(str(i["seed"]) for i in items)}
        for key in METRICS:
            values = [float(i[key]) for i in items if i.get(key) is not None]
            row[f"{key}_mean"] = statistics.mean(values) if values else None
            row[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else (
                0.0 if len(values) == 1 else None
            )
        rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(out_dir / "summary.json", rows)
    print("\nSummary (test set, accuracy %; sample SD):")
    for row in rows:
        def fmt(metric):
            avg = row[f"{metric}_mean"]
            sd = row[f"{metric}_std"]
            if avg is None:
                return "N/A"
            return f"{avg:.2f} +/- {sd:.2f}"
        print(f"  {row['method']:12s} n={row['n_seeds']}: "
              f"fine {fmt('test_fine_acc_pct')} | "
              f"coarse {fmt('test_coarse_acc_pct')}")
    print("Saved:", out_dir / "summary.csv", "and", out_dir / "summary.json")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/clean"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/clean"))
    args = parser.parse_args(argv)
    return summarize_runs(args.runs_dir, args.out_dir)


if __name__ == "__main__":
    main()
