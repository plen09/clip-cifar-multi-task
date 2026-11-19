"""Reproducible training-only perturbations for cached CLIP embeddings.

This module never edits input train.pt/test.pt and never changes validation/test data.
The embedding perturbations are *not* pixel-space image corruptions.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset


def subsample_train_ids(ids, fine_labels, fraction=1.0, seed=2026):
    """Select the same fine-stratified subset for every method and model seed."""
    if not 0 < fraction <= 1:
        raise ValueError("train-fraction must be in (0, 1].")
    if fraction == 1.0:
        return ids
    labels = torch.as_tensor(fine_labels).cpu().numpy()
    train_ids = np.asarray(ids.cpu(), dtype=np.int64)
    rng = np.random.default_rng(seed)
    chosen = []
    for label in np.unique(labels[train_ids]):
        matches = train_ids[labels[train_ids] == label]
        count = max(1, int(round(len(matches) * fraction)))
        chosen.extend(rng.choice(matches, size=count, replace=False).tolist())
    return torch.tensor(sorted(chosen), dtype=torch.long)


def corrupt_training_labels(dataset, task="none", rate=0.0, seed=0):
    """Wrong labels are sampled uniformly from the *other* classes.

    Fine and coarse labels are changed independently on purpose. This creates
    *training-only* supervision inconsistency and tests task imbalance.
    Returns (new TensorDataset, number_modified), without mutating original.
    """
    if task not in ("none", "fine", "coarse"):
        raise ValueError(f"Unknown label noise task: {task}")
    if not 0 <= rate <= 1:
        raise ValueError("label-noise-rate must be in [0, 1].")
    if task == "none" or rate == 0:
        return dataset, 0
    x, fine, coarse = dataset.tensors
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    source = fine if task == "fine" else coarse
    num_classes = 100 if task == "fine" else 20
    mask = torch.rand(source.numel(), generator=gen) < rate
    count = int(mask.sum())
    altered = source.clone()
    if count:
        # 1..C-1 guarantees a different class (no false 'flips').
        offset = torch.randint(1, num_classes, (count,), generator=gen)
        altered[mask] = (altered[mask] + offset) % num_classes
    if task == "fine":
        return TensorDataset(x, altered, coarse), count
    return TensorDataset(x, fine, altered), count


def perturb_embeddings(x, mode="none", level=0.0, generator=None):
    """Apply fresh stochastic train-time feature noise, keeping per-row L2 norms.

    Gaussian sigma is a *relative whole-vector* scale: the expected norm of the
    added noise is about sigma times the input vector's norm. We scale each of
    the 512 coordinates by sigma/sqrt(d), then renormalize the output.
    Dropout masks a fraction of dimensions, then restores the original L2 norm.
    If input is unit-normalized, output stays unit-normalized.
    """
    if mode not in ("none", "gaussian", "dropout"):
        raise ValueError(f"Unknown feature noise mode: {mode}")
    if mode == "none" or level == 0:
        return x
    if not torch.is_floating_point(x) or x.ndim != 2:
        raise ValueError("Expected a 2-D floating-point embedding tensor.")
    if mode == "gaussian" and level < 0:
        raise ValueError("gaussian level must be nonnegative.")
    if mode == "dropout" and not (0 <= level < 1):
        raise ValueError("dropout level must be in [0,1).")
    norms = x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    if mode == "gaussian":
        noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
        y = x + noise * (level / math.sqrt(x.shape[-1])) * norms
    else:
        keep = (torch.rand(x.shape, generator=generator, device=x.device) >= level)
        y = x * keep.to(dtype=x.dtype)
    # Preserve the vector's magnitude. Entirely zero dropout rows remain zero,
    # which can occur only with an extreme dropout level on a 512-D vector.
    return F.normalize(y, p=2, dim=-1) * norms
