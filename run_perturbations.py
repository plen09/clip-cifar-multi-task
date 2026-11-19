#!/usr/bin/env python3
"""Run controlled CLIP FEATURE/LABEL noise experiments with the existing cache.

By default: 4 scenarios x 3 multitask methods x 3 seeds x 30 epochs.
Validation and test always stay clean. Baseline 'results/clean/summary.csv' is optional.
Scenario outputs never overwrite the existing runs/ or results/ folders.
"""
import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCENARIOS = {
    "gaussian03": ("Gaussian feature noise, σ=0.3", ["--feature-noise", "gaussian", "--feature-noise-level", "0.3"]),
    "gaussian07": ("Gaussian feature noise, σ=0.7", ["--feature-noise", "gaussian", "--feature-noise-level", "0.7"]),
    "dropout20": ("Feature dropout, p=0.2", ["--feature-noise", "dropout", "--feature-noise-level", "0.2"]),
    "coarse20": ("Coarse training labels flipped with p=0.2", ["--label-noise", "coarse", "--label-noise-rate", "0.2"]),
    "coarse40": ("Coarse training labels flipped with p=0.4", ["--label-noise", "coarse", "--label-noise-rate", "0.4"]),
    "fine20": ("Fine training labels flipped with p=0.2", ["--label-noise", "fine", "--label-noise-rate", "0.2"]),
    "lowdata10": ("Train on 10% of 45k (4500), original 5k val/10k test", ["--train-fraction", "0.1"]),
    "clean": ("Clean control (no perturbation)", []),
}
RECOMMENDED = ["gaussian07", "coarse20", "fine20", "lowdata10"]
METHODS = ["equal", "uncertainty", "gradnorm", "fixed", "fine_only", "coarse_only"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS), default=RECOMMENDED)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS[:3])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--fixed-fine", type=float, default=1.0)
    parser.add_argument("--fixed-coarse", type=float, default=1.0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--base-runs-dir", type=Path, default=Path("runs/perturbations"))
    parser.add_argument("--base-results-dir", type=Path, default=Path("results/perturbations"))
    parser.add_argument("--baseline-summary", type=Path, default=Path("results/clean/summary.csv"))
    parser.add_argument("--quick", action="store_true", help="Use 1 seed, 2 epochs, quick dirs.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip a completed run within scenario.")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly rerun and replace results.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    if not args.scenarios or not args.methods or not args.seeds:
        parser.error("At least one scenario, method, and seed are required.")
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("epochs and batch-size must be positive.")
    if args.overwrite and args.skip_existing:
        parser.error("Cannot combine --overwrite and --skip-existing.")
    if args.quick:
        args.epochs = 2
        args.seeds = [0]
        if args.base_runs_dir == Path("runs/perturbations"):
            args.base_runs_dir = Path("runs/perturbations_quick")
        if args.base_results_dir == Path("results/perturbations"):
            args.base_results_dir = Path("results/perturbations_quick")
    if not args.dry_run and not all((args.features_dir / file).is_file() for file in ("train.pt", "test.pt")):
        parser.error("Existing feature cache missing; verify data/features/train.pt and test.pt.")

    for scenario in args.scenarios:
        desc, flags = SCENARIOS[scenario]
        print(f"\n=== {scenario}: {desc} ===", flush=True)
        cmd = [
            sys.executable, str(ROOT / "run_all.py"),
            "--features-dir", str(args.features_dir),
            "--runs-dir", str(args.base_runs_dir / scenario),
            "--results-dir", str(args.base_results_dir / scenario),
            "--methods", *args.methods, "--seeds", *[str(s) for s in args.seeds],
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size), "--device", args.device,
            "--fixed-fine", str(args.fixed_fine), "--fixed-coarse", str(args.fixed_coarse),
            *flags,
        ]
        if args.skip_existing:
            cmd.append("--skip-existing")
        if args.overwrite:
            cmd.append("--overwrite")
        if args.no_plots:
            cmd.append("--no-plots")
        if args.dry_run:
            cmd.append("--dry-run")
        print(">", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT)
    cmd = [
        sys.executable, str(ROOT / "compare_perturbations.py"),
        "--results-dir", str(args.base_results_dir),
        "--baseline", str(args.baseline_summary),
    ]
    if not args.dry_run:
        subprocess.run(cmd, check=True, cwd=ROOT)
    else:
        print("Would aggregate:", " ".join(cmd))


if __name__ == "__main__":
    main()
