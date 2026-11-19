#!/usr/bin/env python3
"""Plot accuracy, learned task weights, and shared-gradient conflict diagnostics."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from summarize_hetero import METHODS


def read_logs(runs_dir):
    collected = defaultdict(list)
    for file in Path(runs_dir).glob("*/seed_*/metrics.csv"):
        with file.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [{k: (float(v) if v not in (None, "") else float("nan"))
                     for k, v in line.items()} for line in reader]
        if rows:
            collected[file.parent.parent.name].append(rows)
    return collected


def draw_metric(logs, metric, title, ylabel, filename, allowed=None):
    plt.figure(figsize=(8.5, 5.1))
    plotted = 0
    for method in METHODS:
        if method not in logs or (allowed is not None and method not in allowed):
            continue
        run_rows = logs[method]
        min_epochs = min(len(run) for run in run_rows)
        matrix = np.asarray([[run[i].get(metric, float("nan")) for i in range(min_epochs)]
                             for run in run_rows], dtype=float)
        if not np.isfinite(matrix).all():
            continue
        xs = np.arange(1, min_epochs+1)
        mu = matrix.mean(axis=0)
        sd = matrix.std(axis=0, ddof=1) if len(matrix) > 1 else np.zeros_like(mu)
        line = plt.plot(xs, mu, linewidth=1.8, label=f"{method} (n={len(matrix)})")[0]
        if len(matrix) > 1:
            plt.fill_between(xs, mu-sd, mu+sd, alpha=0.14, color=line.get_color())
        plotted += 1
    if metric == "grad_cosine":
        plt.axhline(0, linewidth=1.0, linestyle="--", color="gray", alpha=0.65)
    if metric == "grad_conflict_rate":
        plt.ylim(0, 1)
    if plotted == 0:
        plt.close()
        return False
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend(fontsize=9)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(filename, dpi=180)
    plt.close()
    print("Saved", filename)
    return True


def create_plots(runs_dir, out_dir):
    logs = read_logs(runs_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = [
        ("val_fine_acc", "Fine Classification · Validation Accuracy", "Accuracy (%)", "validation_fine.png", None),
        ("val_aux_acc", "Auxiliary Task · Validation Accuracy", "Accuracy (%)", "validation_aux.png", None),
        ("grad_cosine", "Shared-Parameter Gradient Cosine", "Cosine similarity", "gradient_cosine.png", None),
        ("grad_conflict_rate", "Frequency of Negative Shared Gradients", "Fraction of probed batches", "gradient_conflict_rate.png", None),
        ("raw_grad_norm_fine", "Raw Fine Shared Gradient Norm", "L2 norm", "raw_grad_fine.png", None),
        ("raw_grad_norm_aux", "Raw Auxiliary Shared Gradient Norm", "L2 norm", "raw_grad_aux.png", None),
        ("weight_fine", "Effective Fine Loss Weight", "Weight", "weight_fine.png", ["equal", "uncertainty", "gradnorm", "fixed"]),
        ("weight_aux", "Effective Auxiliary Loss Weight", "Weight", "weight_aux.png", ["equal", "uncertainty", "gradnorm", "fixed"]),
        ("gradnorm_r_fine", "GradNorm · Fine Relative Inverse Rate", "Inverse training rate", "gradnorm_rate_fine.png", ["gradnorm"]),
        ("gradnorm_r_aux", "GradNorm · Auxiliary Relative Inverse Rate", "Inverse training rate", "gradnorm_rate_aux.png", ["gradnorm"]),
    ]
    n = 0
    for key, title, ylabel, filename, methods in pairs:
        n += int(draw_metric(logs, key, title, ylabel, out_dir / filename, allowed=methods))
    return n


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    return create_plots(args.runs_dir, args.out_dir)


if __name__ == "__main__":
    main()
