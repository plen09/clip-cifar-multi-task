#!/usr/bin/env python3
"""Train the multi-task head on a fixed, frozen CLIP feature cache.

Evaluation protocol:
- Same *fine-stratified* 45k/5k train/validation split for all runs.
- Select checkpoint by VAL task accuracy, never by test accuracy.
- Evaluate the untouched 10k test partition once, at the end of each run.
- Independent seeds: 0,1,2 by default in run_all.py.
"""
import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.balancing import GradNormBalancer, UncertaintyWeighting
from src.data import read_feature_file, stratified_split, tensor_dataset
from src.model import MultiTaskMLP
from src.perturb import corrupt_training_labels, perturb_embeddings, subsample_train_ids
from src.utils import device_from_string, seed_everything, write_json


CSV_COLUMNS = [
    "epoch", "train_fine_loss", "train_coarse_loss", "val_fine_loss",
    "val_coarse_loss", "val_fine_acc", "val_coarse_acc", "val_mean_acc",
    "weight_fine", "weight_coarse", "gradnorm_G_fine", "gradnorm_G_coarse",
    "gradnorm_r_fine", "gradnorm_r_coarse", "gradnorm_target_fine",
    "gradnorm_target_coarse", "elapsed_seconds",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/clean"))
    parser.add_argument("--method", required=True,
                        choices=["fine_only", "coarse_only", "equal",
                                 "uncertainty", "gradnorm", "fixed"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--weight-lr", type=float, default=0.01,
                        help="GradNorm task-weight optimizer LR (default 0.01).")
    parser.add_argument("--alpha", type=float, default=1.5,
                        help="GradNorm alpha; 1.5 is the paper's NYUv2 setting.")
    parser.add_argument("--fixed-fine", type=float, default=1.0)
    parser.add_argument("--fixed-coarse", type=float, default=1.0)
    parser.add_argument("--feature-noise", choices=["none", "gaussian", "dropout"], default="none",
                        help="Apply fresh stochastic perturbations to TRAIN embeddings only.")
    parser.add_argument("--feature-noise-level", type=float, default=0.0,
                        help="Gaussian relative L2 magnitude, or dropout fraction.")
    parser.add_argument("--label-noise", choices=["none", "fine", "coarse"], default="none",
                        help="Randomly flip TRAIN labels for one task; val/test remain clean.")
    parser.add_argument("--label-noise-rate", type=float, default=0.0)
    parser.add_argument("--train-fraction", type=float, default=1.0,
                        help="Fraction of original 45k TRAIN examples, class-stratified.")
    parser.add_argument("--corruption-seed", type=int, default=1729,
                        help="Same corruption across methods within each run seed.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.epochs <= 0 or args.batch_size <= 0 or args.hidden_dim <= 0:
        parser.error("epochs, batch-size, hidden-dim must be positive.")
    if args.lr <= 0 or args.weight_lr <= 0 or args.weight_decay < 0:
        parser.error("Learning rates must be positive; weight-decay must be nonnegative.")
    if args.method == "fixed" and (args.fixed_fine < 0 or args.fixed_coarse < 0
                                    or args.fixed_fine + args.fixed_coarse <= 0):
        parser.error("Fixed task weights must be nonnegative and not both zero.")
    if args.alpha < 0 or args.num_workers < 0:
        parser.error("alpha and num-workers must be nonnegative.")
    if not (0 < args.train_fraction <= 1):
        parser.error("train-fraction must be in (0, 1].")
    if not (0 <= args.label_noise_rate <= 1):
        parser.error("label-noise-rate must be between 0 and 1.")
    if args.label_noise == "none" and args.label_noise_rate != 0:
        parser.error("Set --label-noise fine/coarse when label-noise-rate > 0.")
    if args.label_noise != "none" and args.label_noise_rate == 0:
        parser.error("Set --label-noise-rate > 0 when label-noise is enabled.")
    if args.feature_noise == "none" and args.feature_noise_level != 0:
        parser.error("Set --feature-noise gaussian/dropout when level > 0.")
    if args.feature_noise != "none" and args.feature_noise_level <= 0:
        parser.error("Set --feature-noise-level > 0 when feature noise is enabled.")
    if args.feature_noise == "gaussian" and args.feature_noise_level > 5:
        parser.error("Gaussian relative noise above 5 is not supported.")
    if args.feature_noise == "dropout" and args.feature_noise_level >= 1:
        parser.error("Dropout rate must be less than 1.")
    return args


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    sums = torch.zeros(2, dtype=torch.float64)
    correct = torch.zeros(2, dtype=torch.float64)
    count = 0
    for features, fine, coarse in loader:
        features = features.to(device, non_blocking=True)
        fine = fine.to(device, non_blocking=True)
        coarse = coarse.to(device, non_blocking=True)
        fine_logits, coarse_logits = model(features)
        logits = (fine_logits, coarse_logits)
        targets = (fine, coarse)
        for i in range(2):
            sums[i] += float(F.cross_entropy(logits[i], targets[i], reduction="sum").item())
            correct[i] += int((logits[i].argmax(dim=-1) == targets[i]).sum().item())
        count += len(features)
    return {
        "fine_loss": float(sums[0] / count),
        "coarse_loss": float(sums[1] / count),
        "fine_acc": float(correct[0] / count * 100),
        "coarse_acc": float(correct[1] / count * 100),
    }


def active_tasks(method):
    if method == "fine_only":
        return (True, False)
    if method == "coarse_only":
        return (False, True)
    return (True, True)


def checkpoint_score(metrics, active):
    acc = [metrics["fine_acc"], metrics["coarse_acc"]]
    return sum(x for x, flag in zip(acc, active) if flag) / sum(active)


def reported_metrics(metrics, active):
    output = dict(metrics)
    if not active[0]:
        output["fine_acc"], output["fine_loss"] = None, None
    if not active[1]:
        output["coarse_acc"], output["coarse_loss"] = None, None
    output["mean_acc"] = (
        (output["fine_acc"] + output["coarse_acc"]) / 2.0
        if all(active) else None
    )
    return output


def run(args):
    device = device_from_string(args.device)
    seed_everything(args.seed)
    active = active_tasks(args.method)
    train_payload = read_feature_file(args.features_dir / "train.pt")
    test_payload = read_feature_file(args.features_dir / "test.pt")
    if train_payload["meta"].get("normalized") != test_payload["meta"].get("normalized"):
        raise ValueError("Train/test features have inconsistent normalization.")
    if train_payload["meta"].get("backbone") != test_payload["meta"].get("backbone"):
        raise ValueError("Train/test features were generated from different backbones.")
    if train_payload["meta"].get("checkpoint_sha256") != test_payload["meta"].get("checkpoint_sha256"):
        raise ValueError("Train/test features came from different local checkpoint files.")
    train_ids, val_ids = stratified_split(
        train_payload["fine"], val_fraction=0.1, seed=args.split_seed
    )
    train_ids = subsample_train_ids(
        train_ids, train_payload["fine"], args.train_fraction, seed=args.split_seed + 71
    )
    train_ds = tensor_dataset(train_payload, train_ids)
    train_ds, num_corrupted = corrupt_training_labels(
        train_ds, args.label_noise, args.label_noise_rate,
        seed=args.corruption_seed + args.seed,
    )
    val_ds = tensor_dataset(train_payload, val_ids)
    test_ds = tensor_dataset(test_payload)
    gen = torch.Generator().manual_seed(args.seed)
    # Separate RNG so noise is identical for paired methods at a given seed.
    # It is never consumed by network initialization or minibatch shuffling.
    noise_gen = torch.Generator(device=device).manual_seed(
        args.corruption_seed + args.seed + 1_000_003
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_ds, shuffle=True, generator=gen, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    run_dir = args.runs_dir / args.method / f"seed_{args.seed}"
    csv_path = run_dir / "metrics.csv"
    if csv_path.exists() and not args.overwrite:
        raise FileExistsError(f"{csv_path} exists. Use --overwrite to rerun deliberately.")
    run_dir.mkdir(parents=True, exist_ok=True)

    model = MultiTaskMLP(hidden_dim=args.hidden_dim).to(device)
    balancer = None
    if args.method == "uncertainty":
        balancer = UncertaintyWeighting(n_tasks=2).to(device)
    if args.method == "gradnorm":
        balancer = GradNormBalancer(n_tasks=2).to(device)

    if args.method == "uncertainty":
        net_optimizer = torch.optim.AdamW([
            {"params": model.parameters(), "weight_decay": args.weight_decay},
            {"params": balancer.parameters(), "weight_decay": 0.0},
        ], lr=args.lr)
    else:
        net_optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    weight_optimizer = (
        torch.optim.Adam([balancer.weights_param], lr=args.weight_lr)
        if args.method == "gradnorm" else None
    )

    l0 = None  # Per-task, unweighted losses on the FIRST training minibatch.
    best_val = float("-inf")
    best_epoch = None
    checkpoint_file = run_dir / "best.pt"
    elapsed = time.perf_counter()

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()

    print(f"RUN {args.method} seed={args.seed} on {device}; "
          f"train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}; "
          f"feature_noise={args.feature_noise}:{args.feature_noise_level}; "
          f"label_noise={args.label_noise}:{args.label_noise_rate} "
          f"({num_corrupted} changed); train_fraction={args.train_fraction}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        num_samples = 0
        loss_sums = torch.zeros(2, dtype=torch.float64)
        weight_sums = torch.zeros(2, dtype=torch.float64)
        grad_sums = torch.zeros(2, dtype=torch.float64)
        rate_sums = torch.zeros(2, dtype=torch.float64)
        target_sums = torch.zeros(2, dtype=torch.float64)
        grad_batches = 0

        for features, fine, coarse in train_loader:
            features = features.to(device, non_blocking=True)
            features = perturb_embeddings(
                features, args.feature_noise, args.feature_noise_level, noise_gen
            )
            fine = fine.to(device, non_blocking=True)
            coarse = coarse.to(device, non_blocking=True)
            fine_logits, coarse_logits = model(features)
            losses = [
                F.cross_entropy(fine_logits, fine),
                F.cross_entropy(coarse_logits, coarse),
            ]
            loss_vec = torch.stack(losses)
            if not torch.isfinite(loss_vec).all():
                raise FloatingPointError("Nonfinite task loss. Check features and LR.")

            if args.method == "fine_only":
                weights = torch.tensor([1.0, 0.0], device=device)
                total = loss_vec[0]
            elif args.method == "coarse_only":
                weights = torch.tensor([0.0, 1.0], device=device)
                total = loss_vec[1]
            elif args.method == "equal":
                weights = torch.ones(2, device=device)
                total = loss_vec.sum()
            elif args.method == "fixed":
                weights = torch.tensor([args.fixed_fine, args.fixed_coarse], device=device)
                total = (weights * loss_vec).sum()
            elif args.method == "uncertainty":
                weights = balancer.weights()
                total = balancer(losses)
            else:
                assert args.method == "gradnorm"
                if l0 is None:
                    l0 = loss_vec.detach().clone()
                # Task-weight snapshot controls ONLY the network loss gradient.
                weights = balancer.weights()
                grad_loss, diagnostic = balancer.grad_loss(
                    losses, model.shared_parameters(), l0, alpha=args.alpha
                )
                grad_sums += torch.tensor(diagnostic["weighted_norms"], dtype=torch.float64)
                rate_sums += torch.tensor(diagnostic["inverse_rates"], dtype=torch.float64)
                target_sums += torch.tensor(diagnostic["targets"], dtype=torch.float64)
                grad_batches += 1
                total = (weights * loss_vec).sum()

            net_optimizer.zero_grad(set_to_none=True)
            total.backward()
            if not torch.isfinite(total.detach()):
                raise FloatingPointError("Nonfinite weighted training loss.")
            net_optimizer.step()

            if args.method == "gradnorm":
                # The auxiliary graph uses detached raw gradient norms, so
                # no second-order gradients or network weight updates occur here.
                weight_optimizer.zero_grad(set_to_none=True)
                grad_loss.backward()
                weight_optimizer.step()
                balancer.normalize_()

            batch_n = len(features)
            num_samples += batch_n
            loss_sums += loss_vec.detach().cpu().double() * batch_n
            weight_sums += weights.detach().cpu().double() * batch_n

        val = evaluate(model, val_loader, device)
        score = checkpoint_score(val, active)
        train_losses = (loss_sums / num_samples).tolist()
        mean_weights = (weight_sums / num_samples).tolist()
        reported_val = reported_metrics(val, active)
        row = {
            "epoch": epoch,
            "train_fine_loss": train_losses[0] if active[0] else None,
            "train_coarse_loss": train_losses[1] if active[1] else None,
            "val_fine_loss": reported_val["fine_loss"],
            "val_coarse_loss": reported_val["coarse_loss"],
            "val_fine_acc": reported_val["fine_acc"],
            "val_coarse_acc": reported_val["coarse_acc"],
            "val_mean_acc": reported_val["mean_acc"],
            "weight_fine": mean_weights[0],
            "weight_coarse": mean_weights[1],
            "gradnorm_G_fine": float(grad_sums[0] / grad_batches) if grad_batches else None,
            "gradnorm_G_coarse": float(grad_sums[1] / grad_batches) if grad_batches else None,
            "gradnorm_r_fine": float(rate_sums[0] / grad_batches) if grad_batches else None,
            "gradnorm_r_coarse": float(rate_sums[1] / grad_batches) if grad_batches else None,
            "gradnorm_target_fine": float(target_sums[0] / grad_batches) if grad_batches else None,
            "gradnorm_target_coarse": float(target_sums[1] / grad_batches) if grad_batches else None,
            "elapsed_seconds": round(time.perf_counter() - elapsed, 2),
        }
        with csv_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
            writer.writerow(row)

        if score > best_val:
            best_val = score
            best_epoch = epoch
            torch.save({
                "model": model.state_dict(),
                "balancer": balancer.state_dict() if balancer is not None else None,
                "initial_losses": l0.detach().cpu() if l0 is not None else None,
                "method": args.method,
                "seed": args.seed,
                "epoch": epoch,
                "val_score": score,
                "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            }, checkpoint_file)
        print(f"  epoch {epoch:02d}/{args.epochs} | "
              f"val selected acc {score:.2f}% | "
              f"task weights {mean_weights[0]:.3f}/{mean_weights[1]:.3f}")

    # IMPORTANT: Test is evaluated only after the validation-selected checkpoint.
    state = torch.load(checkpoint_file, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    if balancer is not None:
        balancer.load_state_dict(state["balancer"])
    test = reported_metrics(evaluate(model, test_loader, device), active)
    if balancer is not None:
        final_weights = balancer.weights().cpu().tolist()
    elif args.method == "fine_only":
        final_weights = [1.0, 0.0]
    elif args.method == "coarse_only":
        final_weights = [0.0, 1.0]
    elif args.method == "fixed":
        final_weights = [args.fixed_fine, args.fixed_coarse]
    else:
        final_weights = [1.0, 1.0]

    result = {
        "method": args.method,
        "seed": args.seed,
        "backbone": train_payload["meta"]["backbone"],
        "normalized_features": train_payload["meta"]["normalized"],
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "split_seed": args.split_seed,
        "perturbation": {
            "feature_noise": args.feature_noise,
            "feature_noise_level": args.feature_noise_level,
            "label_noise": args.label_noise,
            "label_noise_rate": args.label_noise_rate,
            "num_changed_train_labels": num_corrupted,
            "train_fraction": args.train_fraction,
            "corruption_seed": args.corruption_seed,
            "clean_validation": True,
            "clean_test": True,
        },
        "best_epoch": best_epoch,
        "best_validation_score_pct": best_val,
        "test_fine_acc_pct": test["fine_acc"],
        "test_coarse_acc_pct": test["coarse_acc"],
        "test_mean_acc_pct": test["mean_acc"],
        "test_fine_ce": test["fine_loss"],
        "test_coarse_ce": test["coarse_loss"],
        "final_weight_fine": final_weights[0],
        "final_weight_coarse": final_weights[1],
        "initial_losses": (
            state["initial_losses"].tolist()
            if state["initial_losses"] is not None else None
        ),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": str(torch.__version__),
        "config": vars(args),
        "note": "Methods and losses transferred to frozen CLIP+CIFAR100; not original-paper numerical reproduction.",
    }
    write_json(run_dir / "results.json", result)
    print(f"TEST {args.method} seed={args.seed} | fine={test['fine_acc']} "
          f"| coarse={test['coarse_acc']} | best_epoch={best_epoch}")
    print("Outputs:", run_dir)
    return result


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
