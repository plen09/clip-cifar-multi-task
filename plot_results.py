#!/usr/bin/env python3
"""Publication-friendly English-labelled figures from saved epoch CSV logs."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless / remote SSH
import matplotlib.pyplot as plt
import numpy as np

from summarize import METHOD_ORDER


def read_runs(runs_dir):
    data = defaultdict(list)
    for path in Path(runs_dir).glob("*/seed_*/metrics.csv"):
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue
        converted = []
        for row in rows:
            out = {}
            for k, v in row.items():
                if k == "epoch":
                    out[k] = int(v)
                else:
                    out[k] = float(v) if v not in ("", None) else float("nan")
            converted.append(out)
        data[path.parent.parent.name].append(converted)
    return data


def plot_metric(data, methods, metric, title, ylabel, destination):
    plt.figure(figsize=(8.4, 5.0))
    n_lines = 0
    for method in methods:
        sequences = data.get(method, [])
        if not sequences:
            continue
        count = min(len(seq) for seq in sequences)
        if count < 1:
            continue
        x = np.arange(1, count + 1)
        matrix = np.asarray([[s[i].get(metric, np.nan) for i in range(count)]
                             for s in sequences], dtype=float)
        if not np.isfinite(matrix).any():
            continue
        # Avoid numpy nanmean warnings for an entirely untrained task.
        if not np.isfinite(matrix).all():
            continue
        mean = matrix.mean(axis=0)
        sd = matrix.std(axis=0, ddof=1) if len(sequences) > 1 else np.zeros(count)
        line = plt.plot(x, mean, linewidth=1.9, label=f"{method} (n={len(sequences)})")[0]
        if len(sequences) > 1:
            plt.fill_between(x, mean - sd, mean + sd, color=line.get_color(), alpha=0.14)
        n_lines += 1
    if n_lines == 0:
        plt.close()
        return False
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.2)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(destination, dpi=180)
    plt.close()
    print("Saved", destination)
    return True


def make_plots(runs_dir, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = read_runs(runs_dir)
    methods = [m for m in METHOD_ORDER if m in data]
    if not methods:
        print("No training CSV logs found; nothing to plot.")
        return 0
    created = 0
    specs = [
        (methods, "val_fine_acc", "CIFAR-100 Fine Classification", "Validation accuracy (%)", "validation_fine.png"),
        (methods, "val_coarse_acc", "CIFAR-100 Coarse Classification", "Validation accuracy (%)", "validation_coarse.png"),
        (["uncertainty", "gradnorm", "equal", "fixed"], "weight_fine",
         "Fine-task Loss Multiplier", "Effective weight", "weight_fine.png"),
        (["uncertainty", "gradnorm", "equal", "fixed"], "weight_coarse",
         "Coarse-task Loss Multiplier", "Effective weight", "weight_coarse.png"),
        (["gradnorm"], "gradnorm_G_fine",
         "GradNorm: Fine-task Shared Gradient Norm", "Gradient L2 norm", "gradnorm_fine.png"),
        (["gradnorm"], "gradnorm_G_coarse",
         "GradNorm: Coarse-task Shared Gradient Norm", "Gradient L2 norm", "gradnorm_coarse.png"),
        (["gradnorm"], "gradnorm_r_fine",
         "GradNorm: Fine-task Relative Inverse Rate", "Inverse rate", "gradnorm_rate_fine.png"),
    ]
    for ms, key, title, ylabel, filename in specs:
        created += int(plot_metric(data, ms, key, title, ylabel, out_dir / filename))
    if "gradnorm" in data:
        sequences = data["gradnorm"]
        n = min(len(seq) for seq in sequences)
        plt.figure(figsize=(8.4, 5))
        for key, label, ls in [
            ("gradnorm_G_fine", "Fine observed", "-"),
            ("gradnorm_target_fine", "Fine target", "--"),
            ("gradnorm_G_coarse", "Coarse observed", "-"),
            ("gradnorm_target_coarse", "Coarse target", "--"),
        ]:
            values = np.array([[s[i].get(key, float("nan")) for i in range(n)]
                               for s in sequences], dtype=float)
            if np.isfinite(values).all():
                plt.plot(np.arange(1, n + 1), values.mean(axis=0),
                         label=label, linestyle=ls, linewidth=1.8)
        if plt.gca().has_data():
            plt.title("GradNorm: Observed vs Target Shared Gradient Norms")
            plt.xlabel("Epoch")
            plt.ylabel("Gradient L2 norm")
            plt.grid(alpha=0.2)
            plt.legend()
            plt.tight_layout()
            dest = out_dir / "gradnorm_observed_vs_target.png"
            plt.savefig(dest, dpi=180)
            print("Saved", dest)
            created += 1
        plt.close()
    return created


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/clean"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/clean/figures"))
    args = parser.parse_args(argv)
    return make_plots(args.runs_dir, args.out_dir)


if __name__ == "__main__":
    main()
