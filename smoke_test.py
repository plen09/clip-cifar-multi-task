#!/usr/bin/env python3
"""End-to-end test with synthetic 512-d embeddings; no network/CLIP/CIFAR needed.

This DOES NOT report benchmark accuracy. It only checks training, GradNorm,
checkpoint loading with weights_only=True, CSV/JSON output and plotting.
"""
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from plot_results import make_plots
from src.utils import write_json
from summarize import summarize_runs
from train import main as train_main


def fake_split(count, prototypes, seed, split):
    gen = torch.Generator().manual_seed(seed)
    fine = torch.arange(count) % 100
    coarse = fine % 20  # Synthetic labels, NOT the real CIFAR-100 hierarchy.
    features = F.normalize(
        prototypes[fine] + 0.4 * torch.randn(count, 512, generator=gen), dim=1
    ).half()
    return {
        "features": features,
        "fine": fine.long(),
        "coarse": coarse.long(),
        "meta": {"normalized": True, "backbone": "OpenAI CLIP ViT-B/32",
                 "split": split, "feature_dim": 512, "synthetic": True},
    }


def main():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(min(old_threads, 4))
    try:
        torch.manual_seed(4)
        prototypes = F.normalize(torch.randn(100, 512), dim=-1)
        with tempfile.TemporaryDirectory(prefix="clip_mtl_smoke_") as tmp:
            tmp = Path(tmp)
            features = tmp / "features"
            runs = tmp / "runs"
            results = tmp / "results"
            features.mkdir()
            torch.save(fake_split(1000, prototypes, 14, "train"), features / "train.pt")
            torch.save(fake_split(200, prototypes, 15, "test"), features / "test.pt")

            for method in ["fine_only", "coarse_only", "equal", "uncertainty", "gradnorm"]:
                train_main([
                    "--method", method, "--features-dir", str(features),
                    "--runs-dir", str(runs), "--seed", "0", "--epochs", "1",
                    "--batch-size", "128", "--device", "cpu",
                ])
                assert (runs / method / "seed_0" / "best.pt").exists()
                assert (runs / method / "seed_0" / "results.json").exists()
            rows = summarize_runs(runs, results)
            assert len(rows) == 5
            assert (results / "summary.csv").exists()
            count = make_plots(runs, results / "figures")
            assert count >= 4, "Plotting unexpectedly produced too few figures."
            print("\nSMOKE TEST PASS: five methods, checkpoints, metrics, summary, figures.")
            print("Synthetic data was temporary; no results are presented as real accuracy.")
    finally:
        torch.set_num_threads(old_threads)


if __name__ == "__main__":
    main()
