#!/usr/bin/env python3
"""Run seed-matched heterogenous multitask comparisons, then summarize & plot."""
import argparse
import subprocess
import sys
from pathlib import Path

from train_hetero import METHODS

ROOT = Path(__file__).resolve().parent
DEFAULT_METHODS = ["fine_only", "aux_only", "equal", "uncertainty", "gradnorm", "pcgrad"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", choices=["rotation", "color"], default="rotation")
    p.add_argument("--features-dir", type=Path, default=None)
    p.add_argument("--runs-dir", type=Path, default=None)
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=DEFAULT_METHODS)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--alpha", type=float, default=1.5)
    p.add_argument("--weight-lr", type=float, default=0.01)
    p.add_argument("--fixed-fine", type=float, default=1.0)
    p.add_argument("--fixed-aux", type=float, default=1.0)
    p.add_argument("--diagnose-every", type=int, default=10)
    p.add_argument("--train-fraction", type=float, default=1.0)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--split-seed", type=int, default=2026)
    p.add_argument("--quick", action="store_true", help="One seed x two epochs in separate output dirs")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args(argv)
    if args.features_dir is None:
        args.features_dir = Path(f"data/hetero_{args.scenario}")
    if args.quick:
        args.seeds, args.epochs = [0], 2
    if args.runs_dir is None:
        args.runs_dir = Path("runs") / (args.scenario + ("_quick" if args.quick else ""))
    if args.results_dir is None:
        args.results_dir = Path("results") / (args.scenario + ("_quick" if args.quick else ""))
    if args.skip_existing and args.overwrite:
        p.error("Cannot combine --skip-existing and --overwrite")
    if not args.dry_run and not all((args.features_dir / name).is_file()
                                    for name in ("train.pt", "test.pt")):
        p.error(f"Missing {args.features_dir}/train.pt or test.pt: "
                "run prepare_rotation_features.py or prepare_color_features.py first")
    for method in args.methods:
        for seed in args.seeds:
            output = args.runs_dir / method / f"seed_{seed}" / "results.json"
            if args.skip_existing and output.is_file():
                print("SKIP already completed:", output, flush=True)
                continue
            cmd = [sys.executable, str(ROOT / "train_hetero.py"),
                   "--scenario", args.scenario,
                   "--features-dir", str(args.features_dir),
                   "--runs-dir", str(args.runs_dir),
                   "--method", method, "--seed", str(seed),
                   "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
                   "--lr", str(args.lr), "--alpha", str(args.alpha),
                   "--weight-lr", str(args.weight_lr),
                   "--fixed-fine", str(args.fixed_fine),
                   "--fixed-aux", str(args.fixed_aux),
                   "--diagnose-every", str(args.diagnose_every),
                   "--train-fraction", str(args.train_fraction),
                   "--split-seed", str(args.split_seed), "--device", args.device]
            if args.overwrite:
                cmd.append("--overwrite")
            print("\n>", " ".join(cmd), flush=True)
            if not args.dry_run:
                subprocess.run(cmd, cwd=ROOT, check=True)
    for script, options in [
        ("summarize_hetero.py", ["--runs-dir", str(args.runs_dir), "--out-dir", str(args.results_dir)]),
        ("plot_hetero.py", ["--runs-dir", str(args.runs_dir), "--out-dir", str(args.results_dir / "figures")]),
    ]:
        if args.no_plots and script == "plot_hetero.py":
            continue
        cmd = [sys.executable, str(ROOT / script), *options]
        print("\n>", " ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=ROOT, check=True)
    print("\nDone; results:", args.results_dir, flush=True)


if __name__ == "__main__":
    main()
