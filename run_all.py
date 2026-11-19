#!/usr/bin/env python3
"""Run 5 methods x 3 seeds, aggregate results, and draw figures."""
import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_METHODS = ["fine_only", "coarse_only", "equal", "uncertainty", "gradnorm"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/clean"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/clean"))
    parser.add_argument("--methods", nargs="+", choices=DEFAULT_METHODS + ["fixed"],
                        default=DEFAULT_METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--weight-lr", type=float, default=0.01)
    parser.add_argument("--fixed-fine", type=float, default=1.0)
    parser.add_argument("--fixed-coarse", type=float, default=1.0)
    parser.add_argument("--feature-noise", choices=["none", "gaussian", "dropout"], default="none")
    parser.add_argument("--feature-noise-level", type=float, default=0.0)
    parser.add_argument("--label-noise", choices=["none", "fine", "coarse"], default="none")
    parser.add_argument("--label-noise-rate", type=float, default=0.0)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--corruption-seed", type=int, default=1729)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--quick", action="store_true",
                        help="1 seed x 2 epochs; uses runs/clean_quick/ and results/clean_quick/ by default.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print all commands without executing them.")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    if args.quick:
        args.seeds = [0]
        args.epochs = 2
        if args.runs_dir == Path("runs/clean"):
            args.runs_dir = Path("runs/clean_quick")
        if args.results_dir == Path("results/clean"):
            args.results_dir = Path("results/clean_quick")
    if not (args.features_dir / "train.pt").is_file() or not (args.features_dir / "test.pt").is_file():
        if not args.dry_run:
            parser.error("Missing cached features. First run: python prepare_features.py")
    if args.skip_existing and args.overwrite:
        parser.error("--skip-existing and --overwrite are mutually exclusive.")

    for method in args.methods:
        for seed in args.seeds:
            result = args.runs_dir / method / f"seed_{seed}" / "results.json"
            if args.skip_existing and result.exists():
                print("SKIP existing", result, flush=True)
                continue
            cmd = [
                sys.executable, str(ROOT / "train.py"),
                "--features-dir", str(args.features_dir),
                "--runs-dir", str(args.runs_dir),
                "--method", method, "--seed", str(seed),
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--alpha", str(args.alpha),
                "--weight-lr", str(args.weight_lr),
                "--device", args.device,
                "--fixed-fine", str(args.fixed_fine),
                "--fixed-coarse", str(args.fixed_coarse),
                "--feature-noise", args.feature_noise,
                "--feature-noise-level", str(args.feature_noise_level),
                "--label-noise", args.label_noise,
                "--label-noise-rate", str(args.label_noise_rate),
                "--train-fraction", str(args.train_fraction),
                "--corruption-seed", str(args.corruption_seed),
            ]
            if args.overwrite:
                cmd.append("--overwrite")
            print("\n>", " ".join(cmd), flush=True)
            if not args.dry_run:
                subprocess.run(cmd, check=True, cwd=ROOT)
    for script, argv2 in [
        ("summarize.py", ["--runs-dir", str(args.runs_dir),
                          "--out-dir", str(args.results_dir)]),
        ("plot_results.py", ["--runs-dir", str(args.runs_dir),
                             "--out-dir", str(args.results_dir / "figures")]),
    ]:
        if script == "plot_results.py" and args.no_plots:
            continue
        cmd = [sys.executable, str(ROOT / script), *argv2]
        print("\n>", " ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True, cwd=ROOT)
    print("\nDone.")


if __name__ == "__main__":
    main()
