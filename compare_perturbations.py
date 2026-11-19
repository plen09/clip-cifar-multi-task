#!/usr/bin/env python3
"""Combine perturbation experiments, compare each condition to its own Equal baseline."""
import argparse
import csv
from pathlib import Path


def read_summary(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def aggregate(root, baseline=None):
    root = Path(root)
    baseline = Path(baseline) if baseline else None
    groups = []
    if baseline and baseline.is_file():
        groups.append(("existing_clean", read_summary(baseline)))
    for path in sorted(root.glob("*/summary.csv")):
        groups.append((path.parent.name, read_summary(path)))
    if not groups:
        print("No summaries found. Run run_perturbations.py first.")
        return []
    out = []
    metrics = ["fine", "coarse", "mean"]
    for scenario, rows in groups:
        equal = next((r for r in rows if r["method"] == "equal"), None)
        for row in rows:
            record = {"scenario": scenario, "method": row["method"],
                      "n_seeds": row["n_seeds"]}
            for metric in metrics:
                column = f"test_{metric}_acc_pct_mean"
                std_col = f"test_{metric}_acc_pct_std"
                value = row.get(column, "")
                record[f"{metric}_acc_pct_mean"] = value
                record[f"{metric}_acc_pct_std"] = row.get(std_col, "")
                if value not in (None, "") and equal and equal.get(column) not in (None, ""):
                    record[f"{metric}_delta_vs_equal_pp"] = round(float(value) - float(equal[column]), 5)
                else:
                    record[f"{metric}_delta_vs_equal_pp"] = ""
            out.append(record)
    root.mkdir(exist_ok=True, parents=True)
    destination = root / "overview.csv"
    with destination.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    print(f"Saved {destination} ({len(out)} rows; accuracy differences in percentage points).")
    print("Compare each method to Equal Weight *within the same scenario*.")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=Path("results/perturbations"))
    p.add_argument("--baseline", type=Path, default=Path("results/clean/summary.csv"))
    args = p.parse_args(argv)
    return aggregate(args.results_dir, args.baseline)


if __name__ == "__main__":
    main()
