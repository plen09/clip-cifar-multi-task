#!/usr/bin/env python3
"""Build a no-reextraction auxiliary task: CIFAR-100 semantics vs color balance.

Auxiliary class is a train-quantile bin of original-image average (R - B).
All quantile thresholds are computed from the *45k training split only*.
Uses previously cached original clean CLIP features as-is.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

from src.data import read_feature_file, stratified_split
from src.offline_assets import ensure_offline_cifar100
from src.utils import write_json


def raw_color_scores(cifar_dir, split, expected_fine):
    source = Path(cifar_dir) / "cifar-100-python" / split
    with source.open("rb") as f:
        payload = pickle.load(f, encoding="bytes")
    array = np.asarray(payload[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
    if not np.array_equal(payload[b"fine_labels"], expected_fine.numpy()):
        raise ValueError("CIFAR raw pixels and CLIP feature labels have different sample order")
    color_diff = array[:, 0].mean(axis=(1, 2)) - array[:, 2].mean(axis=(1, 2))
    return torch.tensor(color_diff, dtype=torch.float32)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--cifar-root", type=Path, default=Path("data"))
    parser.add_argument("--cifar-archive", type=Path, default=Path("data/cifar-100-python.tar.gz"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/hetero_color"))
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    trainfile, testfile = args.out_dir / "train.pt", args.out_dir / "test.pt"
    if (trainfile.exists() or testfile.exists()) and not args.force:
        parser.error(f"Cache already exists in {args.out_dir}; use --force to overwrite intentionally")
    ensure_offline_cifar100(args.cifar_root, args.cifar_archive)
    original = [read_feature_file(args.features_dir / f"{s}.pt") for s in ("train", "test")]
    if original[0]["meta"].get("checkpoint_sha256") != original[1]["meta"].get("checkpoint_sha256"):
        raise ValueError("Train/test original CLIP checkpoints do not match")
    tr_idx, _ = stratified_split(original[0]["fine"], seed=args.split_seed)
    scores = [raw_color_scores(args.cifar_root, name, d["fine"])
              for name, d in zip(("train", "test"), original)]
    thresholds = torch.quantile(scores[0][tr_idx], torch.tensor([0.25, 0.5, 0.75]))
    if not (thresholds[0] < thresholds[1] < thresholds[2]):
        raise ValueError("Color thresholds are degenerate")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, source, score, destination in zip(("train", "test"), original, scores, (trainfile, testfile)):
        labels = torch.bucketize(score.contiguous(), thresholds.contiguous()).long()
        payload = {
            "features": source["features"], "fine": source["fine"], "aux": labels,
            "meta": {**source["meta"], "scenario": "color", "aux_classes": 4,
                     "aux_target": "quartiles of original-image mean(R)-mean(B)",
                     "quantile_train_split_seed": args.split_seed,
                     "color_thresholds": thresholds.tolist(), "split": split},
        }
        tmp = destination.with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.replace(destination)
        print(f"Saved {destination}: {tuple(payload['features'].shape)}, "
              f"color bins={torch.bincount(labels, minlength=4).tolist()}")
    write_json(args.out_dir / "cache_info.json", {
        "scenario": "color", "thresholds": thresholds.tolist(),
        "quantiles_fitted_on": "training split only", "split_seed": args.split_seed,
        "source_features_dir": str(args.features_dir),
        "warning": "Fast color proxy, not evidence of actual gradient conflict until diagnosed",
    })


if __name__ == "__main__":
    main()
