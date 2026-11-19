#!/usr/bin/env python3
"""Summarize heterogeneous multitask experiments and seed-paired deltas."""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from src.utils import write_json

METHODS = ["fine_only", "aux_only", "equal", "uncertainty", "gradnorm", "pcgrad", "fixed"]
METRICS = [
    "test_fine_acc_pct", "test_aux_acc_pct", "test_mean_acc_pct",
    "best_validation_score_pct", "final_weight_fine", "final_weight_aux",
    "grad_cosine_all_epochs_mean", "grad_conflict_rate_all_epochs",
]


def mean_std(values):
    values = [float(v) for v in values if v is not None]
    return ((statistics.mean(values) if values else None),
            (statistics.stdev(values) if len(values) > 1
             else 0.0 if values else None))


def summarize(runs_dir, out_dir):
    runs_dir, out_dir = Path(runs_dir), Path(out_dir)
    grouped = defaultdict(dict)
    scenario = None
    for f in runs_dir.glob("*/seed_*/results.json"):
        item = json.loads(f.read_text(encoding="utf-8"))
        if scenario is None:
            scenario = item["scenario"]
        if item["scenario"] != scenario:
            raise ValueError("Mixed scenarios in a single runs-dir; separate rotation and color runs")
        grouped[item["method"]][item["seed"]] = item
    if not grouped:
        print(f"No completed results in {runs_dir}")
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    pairs = []
    for method in sorted(grouped, key=lambda m: (METHODS.index(m) if m in METHODS else 999, m)):
        items = grouped[method]
        row = {"scenario": scenario, "method": method, "n_seeds": len(items),
               "seeds": ",".join(str(s) for s in sorted(items))}
        for key in METRICS:
            avg, std = mean_std([item.get(key) for item in items.values()])
            row[key+"_mean"] = avg
            row[key+"_std"] = std
        for metric in ("test_fine_acc_pct", "test_aux_acc_pct", "test_mean_acc_pct"):
            delta = []
            for seed in sorted(items):
                target, equal = items[seed], grouped.get("equal", {}).get(seed)
                if equal is None or target.get(metric) is None or equal.get(metric) is None:
                    continue
                diff = target[metric] - equal[metric]
                delta.append(diff)
                pairs.append({"method": method, "seed": seed, "metric": metric,
                              "method_value": target[metric], "equal_value": equal[metric],
                              "delta_vs_equal_pp": diff})
            row["delta_vs_equal_"+metric+"_mean_pp"], row["delta_vs_equal_"+metric+"_std_pp"] = mean_std(delta)
        rows.append(row)

    path = out_dir / "summary.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "paired_differences.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "seed", "metric", "method_value", "equal_value", "delta_vs_equal_pp"])
        writer.writeheader()
        writer.writerows(pairs)
    write_json(out_dir / "summary.json", rows)
    print("\nScenario:", scenario, "| test accuracy % ± sample SD (single seed = no SD inference)")
    for row in rows:
        def value(key):
            v = row.get(key+"_mean")
            s = row.get(key+"_std")
            return f"{v:.2f}±{s:.2f}" if v is not None else "—"
        print(f"  {row['method']:12s} n={row['n_seeds']} fine={value('test_fine_acc_pct')} "
              f"aux={value('test_aux_acc_pct')} joint={value('test_mean_acc_pct')} "
              f"conflict={value('grad_conflict_rate_all_epochs')}")
    print(f"Saved {path} and {out_dir / 'paired_differences.csv'}")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    return summarize(args.runs_dir, args.out_dir)


if __name__ == "__main__":
    main()
