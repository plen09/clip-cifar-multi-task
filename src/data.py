import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset


def official_coarse_labels(cifar_root, train, expected_fine):
    """Read the *official* CIFAR-100 coarse labels, never a manual mapping.

    torchvision.datasets.CIFAR100 supplies fine targets; its extracted original
    pickles supply both fine_labels and coarse_labels in the same sample order.
    """
    file = Path(cifar_root) / "cifar-100-python" / ("train" if train else "test")
    if not file.is_file():
        raise FileNotFoundError(
            f"Missing {file}. Download CIFAR100 with torchvision first."
        )
    with file.open("rb") as handle:
        data = pickle.load(handle, encoding="bytes")
    fine = np.asarray(data[b"fine_labels"], dtype=np.int64)
    coarse = np.asarray(data[b"coarse_labels"], dtype=np.int64)
    if not np.array_equal(fine, np.asarray(expected_fine, dtype=np.int64)):
        raise ValueError("Official CIFAR-100 pickle labels disagree with torchvision order.")
    if not (np.all((coarse >= 0) & (coarse < 20))):
        raise ValueError("Official coarse labels must be in 0..19.")
    return torch.as_tensor(coarse, dtype=torch.long)


def read_feature_file(path):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    required = {"features", "fine", "coarse", "meta"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError(f"Invalid feature file {path}: missing keys {required}.")
    x, fine, coarse = payload["features"], payload["fine"], payload["coarse"]
    if not (x.ndim == 2 and x.shape[1] == 512):
        raise ValueError(f"Expected CLIP ViT-B/32 512-d features, got {x.shape}.")
    if fine.shape != (len(x),) or coarse.shape != (len(x),):
        raise ValueError("Feature/label lengths disagree.")
    if x.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        raise ValueError(f"Unexpected feature dtype: {x.dtype}.")
    if torch.isnan(x.float()).any() or not torch.isfinite(x.float()).all():
        raise ValueError("Features contain NaN or infinity.")
    return payload


def stratified_split(fine_labels, val_fraction=0.10, seed=2026):
    """Stable, per-fine-class train/val split shared by all methods/seeds.

    On full CIFAR-100 train (500 images/class), yields 450 train and 50 val
    images/class: 45,000 training + 5,000 validation examples total.
    """
    fine = torch.as_tensor(fine_labels).cpu().numpy()
    rng = np.random.default_rng(seed)
    train_ids, val_ids = [], []
    for label in np.unique(fine):
        indices = np.where(fine == label)[0]
        if len(indices) < 2:
            raise ValueError(f"Class {label} has <2 samples: cannot split.")
        indices = rng.permutation(indices)
        n_val = min(len(indices) - 1, max(1, int(round(len(indices) * val_fraction))))
        val_ids.extend(indices[:n_val].tolist())
        train_ids.extend(indices[n_val:].tolist())
    return torch.tensor(sorted(train_ids), dtype=torch.long), torch.tensor(sorted(val_ids), dtype=torch.long)


def tensor_dataset(payload, indices=None):
    x = payload["features"].float()
    fine = payload["fine"].long()
    coarse = payload["coarse"].long()
    if indices is not None:
        x, fine, coarse = x[indices], fine[indices], coarse[indices]
    return TensorDataset(x, fine, coarse)
