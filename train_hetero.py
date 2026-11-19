#!/usr/bin/env python3
"""Frozen-CLIP heterogenous multitask training with cosine/conflict diagnostics.

Train: CIFAR-100 fine classes + 4-way rotation (primary), or fine + color (control).
Rotation assignment covers all 4 angles per image over 4 epochs.
Validation selection only. Test once using validation-selected checkpoint.
"""
import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.balancing import GradNormBalancer, UncertaintyWeighting
from src.data import stratified_split
from src.hetero import (HeteroMLP, cosine_and_norms, pcgrad_combine,
                        read_hetero_cache, rotation_indices, subset_dataset, task_grads)
from src.utils import device_from_string, seed_everything, write_json


METHODS = ("fine_only", "aux_only", "equal", "uncertainty", "gradnorm", "pcgrad", "fixed")
CSV_FIELDS = [
    "epoch", "train_fine_ce", "train_aux_ce", "val_fine_ce", "val_aux_ce",
    "val_fine_acc", "val_aux_acc", "val_mean_acc", "val_checkpoint_score",
    "weight_fine", "weight_aux", "raw_grad_norm_fine", "raw_grad_norm_aux",
    "grad_cosine", "grad_conflict_rate", "grad_diagnostic_batches",
    "gradnorm_G_fine", "gradnorm_G_aux", "gradnorm_target_fine",
    "gradnorm_target_aux", "gradnorm_r_fine", "gradnorm_r_aux", "elapsed_seconds",
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", choices=["rotation", "color"], default="rotation")
    p.add_argument("--features-dir", type=Path, default=None)
    p.add_argument("--runs-dir", type=Path, default=None)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split-seed", type=int, default=2026)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--weight-lr", type=float, default=0.01)
    p.add_argument("--alpha", type=float, default=1.5)
    p.add_argument("--fixed-fine", type=float, default=1.0)
    p.add_argument("--fixed-aux", type=float, default=1.0)
    p.add_argument("--diagnose-every", type=int, default=10,
                   help="Probe shared gradient angle every N training batches. PCGrad probes all.")
    p.add_argument("--train-fraction", type=float, default=1.0)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    if args.features_dir is None:
        args.features_dir = Path(f"data/hetero_{args.scenario}")
    if args.runs_dir is None:
        args.runs_dir = Path("runs") / args.scenario
    if any(v <= 0 for v in (args.epochs, args.batch_size, args.hidden_dim,
                            args.lr, args.weight_lr, args.diagnose_every)):
        p.error("Positive epochs/batch-size/hidden-dim/lr/weight-lr/diagnose-every required")
    if args.alpha < 0 or args.weight_decay < 0 or not (0 < args.train_fraction <= 1):
        p.error("alpha, weight-decay must be nonnegative; train-fraction must be (0,1]")
    if args.fixed_fine < 0 or args.fixed_aux < 0 or args.fixed_fine + args.fixed_aux == 0:
        p.error("Fixed task weights must be nonnegative and not both zero")
    return args


def train_features(features, ids, scenario, epoch, seed):
    if scenario == "rotation":
        angles = rotation_indices(ids, epoch, seed)
        indices = torch.arange(len(ids), device=features.device)
        x = features[indices, angles, :]
        return x, angles
    return features, None


@torch.inference_mode()
def evaluate(model, loader, scenario, device):
    model.eval()
    fine_ce, aux_ce = 0.0, 0.0
    fine_correct, aux_correct = 0, 0
    fine_count, aux_count = 0, 0
    for x, fine, aux, _ in loader:
        x = x.to(device, non_blocking=True)
        fine = fine.to(device, non_blocking=True)
        if scenario == "rotation":
            fine_logits, _ = model(x[:, 0, :])
            bs = len(x)
            aux_logits = model(x.reshape(-1, 512))[1]
            aux_targets = torch.arange(4, device=device, dtype=torch.long).repeat(bs)
        else:
            fine_logits, aux_logits = model(x)
            aux_targets = aux.to(device, non_blocking=True)
        fine_ce += float(F.cross_entropy(fine_logits, fine, reduction="sum").item())
        aux_ce += float(F.cross_entropy(aux_logits, aux_targets, reduction="sum").item())
        fine_correct += int((fine_logits.argmax(1) == fine).sum().item())
        aux_correct += int((aux_logits.argmax(1) == aux_targets).sum().item())
        fine_count += len(fine)
        aux_count += len(aux_targets)
    return {
        "fine_acc": 100.0 * fine_correct / fine_count,
        "aux_acc": 100.0 * aux_correct / aux_count,
        "fine_ce": fine_ce / fine_count,
        "aux_ce": aux_ce / aux_count,
    }


def active_tasks(method):
    return method != "aux_only", method != "fine_only"


def checkpoint_score(metrics, method):
    if method == "fine_only":
        return metrics["fine_acc"]
    if method == "aux_only":
        return metrics["aux_acc"]
    return (metrics["fine_acc"] + metrics["aux_acc"]) / 2.0


def run(args):
    seed_everything(args.seed)
    device = device_from_string(args.device)
    train = read_hetero_cache(args.features_dir / "train.pt", args.scenario)
    test = read_hetero_cache(args.features_dir / "test.pt", args.scenario)
    if train["meta"].get("checkpoint_sha256") != test["meta"].get("checkpoint_sha256"):
        raise ValueError("Train/test features originate from different CLIP checkpoints")
    if train["meta"].get("normalized") != test["meta"].get("normalized"):
        raise ValueError("Train/test feature normalization differs")
    if args.scenario == "color" and train["meta"].get("quantile_train_split_seed") != args.split_seed:
        raise ValueError("Color quantile thresholds were fitted for a different split seed")
    train_ids, val_ids = stratified_split(train["fine"], seed=args.split_seed)
    if args.train_fraction < 1:
        from src.perturb import subsample_train_ids
        train_ids = subsample_train_ids(train_ids, train["fine"], args.train_fraction,
                                        seed=args.split_seed + 71)
    dataset_train = subset_dataset(train, train_ids)
    dataset_val = subset_dataset(train, val_ids)
    dataset_test = subset_dataset(test)
    data_args = {"batch_size": args.batch_size, "num_workers": 0,
                 "pin_memory": device.type == "cuda"}
    generator = torch.Generator().manual_seed(args.seed)
    loader_train = DataLoader(dataset_train, shuffle=True, generator=generator, **data_args)
    loader_val = DataLoader(dataset_val, shuffle=False, **data_args)
    loader_test = DataLoader(dataset_test, shuffle=False, **data_args)

    run_dir = args.runs_dir / args.method / f"seed_{args.seed}"
    metrics_path = run_dir / "metrics.csv"
    if metrics_path.exists() and not args.overwrite:
        raise FileExistsError(f"Experiment output exists: {metrics_path}; choose --overwrite if needed")
    run_dir.mkdir(parents=True, exist_ok=True)
    model = HeteroMLP(hidden_dim=args.hidden_dim).to(device)
    balancer = None
    if args.method == "uncertainty":
        balancer = UncertaintyWeighting(n_tasks=2).to(device)
    elif args.method == "gradnorm":
        balancer = GradNormBalancer(n_tasks=2).to(device)
    if args.method == "uncertainty":
        opt_net = torch.optim.AdamW([
            {"params": model.parameters(), "weight_decay": args.weight_decay},
            {"params": balancer.parameters(), "weight_decay": 0},
        ], lr=args.lr)
    else:
        opt_net = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_weights = (torch.optim.Adam([balancer.weights_param], lr=args.weight_lr)
                   if args.method == "gradnorm" else None)
    shared = model.shared_parameters()
    started = time.perf_counter()
    best_score, best_epoch = float("-inf"), None
    checkpoint = run_dir / "best.pt"
    initial_losses = None
    diagnostics_total, diagnostics_conflict = 0, 0
    cosine_sum_total = 0.0
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()
    print(f"[HETERO] scenario={args.scenario} method={args.method} seed={args.seed} "
          f"device={device} train={len(dataset_train)} val={len(dataset_val)} "
          f"test={len(dataset_test)}", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = torch.zeros(2, dtype=torch.float64)
        weight_sum = torch.zeros(2, dtype=torch.float64)
        count = 0
        diagnostics = {"cos": 0.0, "negative": 0, "norm_fine": 0.0, "norm_aux": 0.0, "count": 0}
        gradnorm_diag = {key: torch.zeros(2, dtype=torch.float64)
                         for key in ("weighted_norms", "targets", "inverse_rates")}
        gradnorm_batches = 0

        for batch_i, (x, fine, aux, ids) in enumerate(loader_train):
            x = x.to(device, non_blocking=True)
            fine = fine.to(device, non_blocking=True)
            aux = aux.to(device, non_blocking=True)
            ids = ids.to(device, non_blocking=True)
            x, rotating = train_features(x, ids, args.scenario, epoch, args.seed)
            if rotating is not None:
                aux = rotating
            logits_fine, logits_aux = model(x)
            losses = (F.cross_entropy(logits_fine, fine), F.cross_entropy(logits_aux, aux))
            v = torch.stack(losses)
            if not torch.isfinite(v).all():
                raise FloatingPointError("Nonfinite per-task loss. Check LR/features.")

            if args.method == "fine_only":
                w = torch.tensor([1.0, 0.0], device=device)
                total = losses[0]
            elif args.method == "aux_only":
                w = torch.tensor([0.0, 1.0], device=device)
                total = losses[1]
            elif args.method == "equal" or args.method == "pcgrad":
                w = torch.ones(2, device=device)
                total = v.sum()
            elif args.method == "fixed":
                w = torch.tensor([args.fixed_fine, args.fixed_aux], device=device)
                total = (w * v).sum()
            elif args.method == "uncertainty":
                w = balancer.weights()
                total = balancer(losses)
            elif args.method == "gradnorm":
                if initial_losses is None:
                    initial_losses = v.detach().clone()
                w = balancer.weights()
                gradloss, gdiag = balancer.grad_loss(losses, shared, initial_losses, alpha=args.alpha)
                for key in gradnorm_diag:
                    gradnorm_diag[key] += torch.tensor(gdiag[key], dtype=torch.float64)
                gradnorm_batches += 1
                total = (w * v).sum()
            else:
                raise AssertionError(args.method)

            grads = None
            # Measuring angles of raw gradients (before task weighting) isolates
            # gradient direction from UW/GradNorm's positive scalar multipliers.
            if all(active_tasks(args.method)) and (args.method == "pcgrad"
                 or batch_i % args.diagnose_every == 0):
                grads = task_grads(losses, shared)
                cosine, n1, n2, _ = cosine_and_norms(*grads)
                cos = float(cosine.detach().item())
                diagnostics["cos"] += cos
                diagnostics["negative"] += int(cos < 0)
                diagnostics["norm_fine"] += float(n1.detach().item())
                diagnostics["norm_aux"] += float(n2.detach().item())
                diagnostics["count"] += 1

            opt_net.zero_grad(set_to_none=True)
            total.backward()
            if args.method == "pcgrad":
                if grads is None:
                    raise AssertionError("PCGrad requires per-task shared gradients")
                merged, _ = pcgrad_combine(*grads)
                for param, updated in zip(shared, merged):
                    param.grad = updated.to(dtype=param.dtype).clone()
                # task-specific fine and aux heads still receive ordinary gradients
            opt_net.step()
            if args.method == "gradnorm":
                opt_weights.zero_grad(set_to_none=True)
                gradloss.backward()
                opt_weights.step()
                balancer.normalize_()

            n = len(x)
            count += n
            loss_sum += v.detach().double().cpu() * n
            weight_sum += w.detach().double().cpu() * n

        val = evaluate(model, loader_val, args.scenario, device)
        score = checkpoint_score(val, args.method)
        diag_num = diagnostics["count"]
        if diag_num:
            diagnostics_total += diag_num
            diagnostics_conflict += diagnostics["negative"]
            cosine_sum_total += diagnostics["cos"]
        means = (loss_sum / count).tolist()
        wmeans = (weight_sum / count).tolist()
        def average_diag(key):
            return diagnostics[key] / diag_num if diag_num else None
        def gradnorm_avg(key, idx):
            return float(gradnorm_diag[key][idx] / gradnorm_batches) if gradnorm_batches else None
        row = {
            "epoch": epoch, "train_fine_ce": means[0] if args.method != "aux_only" else None,
            "train_aux_ce": means[1] if args.method != "fine_only" else None,
            "val_fine_ce": val["fine_ce"] if args.method != "aux_only" else None,
            "val_aux_ce": val["aux_ce"] if args.method != "fine_only" else None,
            "val_fine_acc": val["fine_acc"] if args.method != "aux_only" else None,
            "val_aux_acc": val["aux_acc"] if args.method != "fine_only" else None,
            "val_mean_acc": ((val["fine_acc"] + val["aux_acc"]) / 2
                             if all(active_tasks(args.method)) else None),
            "val_checkpoint_score": score, "weight_fine": wmeans[0], "weight_aux": wmeans[1],
            "raw_grad_norm_fine": average_diag("norm_fine"),
            "raw_grad_norm_aux": average_diag("norm_aux"),
            "grad_cosine": average_diag("cos"),
            "grad_conflict_rate": average_diag("negative"),
            "grad_diagnostic_batches": diag_num,
            "gradnorm_G_fine": gradnorm_avg("weighted_norms", 0),
            "gradnorm_G_aux": gradnorm_avg("weighted_norms", 1),
            "gradnorm_target_fine": gradnorm_avg("targets", 0),
            "gradnorm_target_aux": gradnorm_avg("targets", 1),
            "gradnorm_r_fine": gradnorm_avg("inverse_rates", 0),
            "gradnorm_r_aux": gradnorm_avg("inverse_rates", 1),
            "elapsed_seconds": round(time.perf_counter() - started, 2),
        }
        with metrics_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)
        if score > best_score:
            best_score, best_epoch = score, epoch
            torch.save({
                "model": model.state_dict(),
                "balancer": balancer.state_dict() if balancer is not None else None,
                "method": args.method, "scenario": args.scenario,
                "initial_losses": initial_losses.detach().cpu() if initial_losses is not None else None,
                "epoch": epoch, "val_score": score,
            }, checkpoint)
        angle_text = (f"cos={average_diag('cos'):+.3f} "
                      f"conflict={average_diag('negative'):.1%}" if diag_num else "single-task")
        print(f"  ep={epoch:02d}/{args.epochs} val fine={val['fine_acc']:.2f} "
              f"aux={val['aux_acc']:.2f} weight={wmeans[0]:.3f}/{wmeans[1]:.3f} "
              f"{angle_text}", flush=True)

    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    if balancer is not None:
        balancer.load_state_dict(state["balancer"])
    test_metrics = evaluate(model, loader_test, args.scenario, device)
    if balancer is not None:
        final_weights = balancer.weights().cpu().tolist()
    elif args.method == "fine_only":
        final_weights = [1.0, 0.0]
    elif args.method == "aux_only":
        final_weights = [0.0, 1.0]
    elif args.method == "fixed":
        final_weights = [args.fixed_fine, args.fixed_aux]
    else:
        final_weights = [1.0, 1.0]
    active = active_tasks(args.method)
    result = {
        "scenario": args.scenario, "method": args.method, "seed": args.seed,
        "aux_task": "four-way CCW orientation prediction" if args.scenario == "rotation"
                    else "original-image R-B color quartile classification",
        "backbone": train["meta"].get("backbone"),
        "checkpoint_sha256": train["meta"].get("checkpoint_sha256"),
        "best_epoch": best_epoch, "best_validation_score_pct": best_score,
        "test_fine_acc_pct": test_metrics["fine_acc"] if active[0] else None,
        "test_aux_acc_pct": test_metrics["aux_acc"] if active[1] else None,
        "test_mean_acc_pct": ((test_metrics["fine_acc"] + test_metrics["aux_acc"]) / 2
                              if all(active) else None),
        "test_fine_ce": test_metrics["fine_ce"] if active[0] else None,
        "test_aux_ce": test_metrics["aux_ce"] if active[1] else None,
        "final_weight_fine": final_weights[0], "final_weight_aux": final_weights[1],
        "grad_cosine_all_epochs_mean": cosine_sum_total / diagnostics_total if diagnostics_total else None,
        "grad_conflict_rate_all_epochs": diagnostics_conflict / diagnostics_total if diagnostics_total else None,
        "n_gradient_probes": diagnostics_total,
        "n_train_images": len(dataset_train), "n_val_images": len(dataset_val),
        "n_test_images": len(dataset_test),
        "test_aux_count": len(dataset_test) * (4 if args.scenario == "rotation" else 1),
        "train_rotation_sampling": "one angle per image per epoch, four-angle cycle"
                                   if args.scenario == "rotation" else "not applicable",
        "val_rotation_sampling": "all four angles" if args.scenario == "rotation" else "not applicable",
        "val_fine_view": "0-degree only" if args.scenario == "rotation" else "original",
        "test_fine_view": "0-degree only" if args.scenario == "rotation" else "original",
        "weights_source_unverified": True,
        "seed_matched_rotation_assignments": True,
        "config": vars(args), "device": str(device),
        "pytorch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "warning": "Task gradient conflict not guaranteed. Inspect measured cosine/conflict rate."
    }
    write_json(run_dir / "results.json", result)
    print(f"[TEST] {args.scenario} {args.method} seed={args.seed} "
          f"fine={result['test_fine_acc_pct']} aux={result['test_aux_acc_pct']} "
          f"best_epoch={best_epoch} -> {run_dir}", flush=True)
    return result


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
