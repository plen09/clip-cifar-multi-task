"""Two genuinely different supervised tasks on frozen ViT-B/32 embeddings.

Primary: fine-grained object semantics vs 4-way image rotation.
Fast control: fine-grained object semantics vs original-image R-B color-bin.
Gradient diagnostics and PCGrad operate on exactly the shared trunk parameters.
"""
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import TensorDataset


class HeteroMLP(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, n_fine=100, n_aux=4):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.fine_head = nn.Linear(hidden_dim, n_fine)
        self.aux_head = nn.Linear(hidden_dim, n_aux)

    def forward(self, x):
        shared = self.shared(x)
        return self.fine_head(shared), self.aux_head(shared)

    def shared_parameters(self):
        return tuple(self.shared.parameters())


def read_hetero_cache(path, scenario):
    data = torch.load(Path(path), weights_only=True, map_location="cpu")
    if not isinstance(data, dict) or not all(k in data for k in ("features", "fine", "aux", "meta")):
        raise ValueError(f"Not a hetero feature cache: {path}")
    x = data["features"]
    n = x.shape[0]
    expected = (n, 4, 512) if scenario == "rotation" else (n, 512)
    if tuple(x.shape) != expected:
        raise ValueError(f"Expected {expected} feature shape for {scenario}, got {tuple(x.shape)}")
    if data["fine"].shape != (n,) or data["aux"].shape != (n,):
        raise ValueError("Label length mismatch")
    if data["meta"].get("scenario") != scenario:
        raise ValueError(f"Cache scenario mismatch: {data['meta'].get('scenario')} != {scenario}")
    if not torch.isfinite(x.float()).all():
        raise ValueError("Nonfinite cached features")
    if not bool(torch.all((data["fine"] >= 0) & (data["fine"] < 100))):
        raise ValueError("Fine labels out of range")
    if scenario == "color" and not bool(torch.all((data["aux"] >= 0) & (data["aux"] < 4))):
        raise ValueError("Color labels out of range")
    return data


def subset_dataset(cache, ids=None):
    x = cache["features"].float()
    fine = cache["fine"].long()
    aux = cache["aux"].long()
    original_indices = torch.arange(len(fine), dtype=torch.long)
    if ids is not None:
        x, fine, aux, original_indices = x[ids], fine[ids], aux[ids], original_indices[ids]
    return TensorDataset(x, fine, aux, original_indices)


def rotation_indices(ids, epoch, seed):
    """Each image covers each of four rotations once per four consecutive epochs.

    No RNG consumption by model or method affects the image-angle assignment.
    Also deterministic across dataloader orders and batch sizes.
    """
    return ((ids.to(dtype=torch.int64) * 2654435761
             + int(seed) * 1013904223 + int(epoch) - 1) % 4).long()


def cosine_and_norms(g1, g2, eps=1e-12):
    """Return cosine and individual L2 norms without flattening huge parameter arrays."""
    if len(g1) != len(g2) or not g1:
        raise ValueError("Gradient lists must be nonempty and aligned")
    dot = torch.stack([(a.float() * b.float()).sum() for a, b in zip(g1, g2)]).sum()
    n1_sq = torch.stack([a.float().square().sum() for a in g1]).sum()
    n2_sq = torch.stack([b.float().square().sum() for b in g2]).sum()
    cos = dot / (n1_sq.sqrt() * n2_sq.sqrt()).clamp_min(eps)
    return cos, n1_sq.sqrt(), n2_sq.sqrt(), dot


def task_grads(losses, params):
    shared = tuple(params)
    return tuple(tuple(g.detach() for g in torch.autograd.grad(
        loss, shared, retain_graph=True, create_graph=False, allow_unused=False
    )) for loss in losses)


def pcgrad_combine(g1, g2):
    """Two-task PCGrad: project each task away from the other iff dot < 0.

    Original task gradients are used for both projections (two-task form).
    Caller replaces shared trunk .grad ONLY. Private head grads stay unchanged.
    """
    _, n1, n2, dot = cosine_and_norms(g1, g2)
    if float(dot.detach()) >= 0.0:
        return tuple(a+b for a,b in zip(g1,g2)), False
    # Each projected component is orthogonal to the other original gradient.
    coef1 = dot / n2.square().clamp_min(1e-18)
    coef2 = dot / n1.square().clamp_min(1e-18)
    return tuple((a - coef1 * b) + (b - coef2 * a)
                 for a, b in zip(g1, g2)), True
