"""Small, fully offline checks for the new stress-test extension."""
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset

from compare_perturbations import aggregate
from smoke_test import fake_split
from src.data import read_feature_file, stratified_split, tensor_dataset
from src.perturb import (
    corrupt_training_labels, perturb_embeddings, subsample_train_ids,
)
from train import main as train_main


class PerturbationTests(unittest.TestCase):
    def test_train_fraction_stratified_and_deterministic(self):
        labels = torch.arange(100).repeat_interleave(20)
        tr, val = stratified_split(labels, val_fraction=0.1, seed=2026)
        a = subsample_train_ids(tr, labels, 0.1, seed=999)
        b = subsample_train_ids(tr, labels, 0.1, seed=999)
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(len(a), 200)  # 2 per fine category
        self.assertTrue(torch.equal(torch.bincount(labels[a], minlength=100),
                                    torch.full((100,), 2, dtype=torch.long)))
        self.assertEqual(len(set(a.tolist()) & set(val.tolist())), 0)

    def test_label_noise_changes_only_requested_task_and_keeps_original(self):
        n = 1500
        x = torch.randn(n, 512)
        f = torch.arange(n) % 100
        c = torch.arange(n) % 20
        ds = TensorDataset(x, f, c)
        changed, n_changed = corrupt_training_labels(ds, "coarse", 1.0, seed=77)
        self.assertEqual(n_changed, n)
        self.assertTrue(torch.equal(changed.tensors[1], ds.tensors[1]))
        self.assertTrue(torch.all(changed.tensors[2] != ds.tensors[2]))
        self.assertTrue(torch.equal(ds.tensors[2], c))
        changed2, _ = corrupt_training_labels(ds, "coarse", 1.0, seed=77)
        self.assertTrue(torch.equal(changed.tensors[2], changed2.tensors[2]))
        fine_ds, _ = corrupt_training_labels(ds, "fine", 0.3, seed=77)
        self.assertTrue(torch.equal(fine_ds.tensors[2], c))
        self.assertTrue(torch.all((fine_ds.tensors[1] >= 0) & (fine_ds.tensors[1] < 100)))

    def test_embedding_noise_is_reproducible_and_preserves_norm(self):
        x = F.normalize(torch.randn(32, 512), dim=-1)
        for mode, level in [("gaussian", 0.7), ("dropout", 0.3)]:
            g1 = torch.Generator().manual_seed(43)
            g2 = torch.Generator().manual_seed(43)
            y1 = perturb_embeddings(x, mode, level, g1)
            y2 = perturb_embeddings(x, mode, level, g2)
            self.assertTrue(torch.equal(y1, y2))
            self.assertTrue(torch.allclose(y1.norm(dim=-1), x.norm(dim=-1), atol=1e-5))
            self.assertFalse(torch.equal(x, y1))
        self.assertIs(perturb_embeddings(x), x)

    def test_train_smoke_with_corrupted_coarse_and_clean_val(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(min(old_threads, 2))
        try:
            with tempfile.TemporaryDirectory(prefix="clip_mtl_perturb_") as root:
                root = Path(root)
                feat = root / "features"
                feat.mkdir()
                torch.manual_seed(4)
                prototypes = F.normalize(torch.randn(100, 512), dim=-1)
                torch.save(fake_split(1200, prototypes, 12, "train"), feat / "train.pt")
                torch.save(fake_split(300, prototypes, 13, "test"), feat / "test.pt")
                # Keep the saved cache immutable and verify it wasn't edited by training.
                orig = read_feature_file(feat / "train.pt")["coarse"].clone()
                for method in ["equal", "uncertainty", "gradnorm"]:
                    result = train_main([
                        "--method", method, "--features-dir", str(feat),
                        "--runs-dir", str(root / "runs"), "--seed", "0", "--epochs", "1",
                        "--batch-size", "200", "--device", "cpu",
                        "--label-noise", "coarse", "--label-noise-rate", "0.2",
                        "--train-fraction", "0.5",
                    ])
                    self.assertTrue(result["perturbation"]["clean_validation"])
                    self.assertTrue(result["perturbation"]["clean_test"])
                    self.assertGreater(result["perturbation"]["num_changed_train_labels"], 0)
                self.assertTrue(torch.equal(read_feature_file(feat / "train.pt")["coarse"], orig))
        finally:
            torch.set_num_threads(old_threads)


if __name__ == "__main__":
    unittest.main()
