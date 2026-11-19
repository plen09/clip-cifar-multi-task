import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from src.hetero import (HeteroMLP, cosine_and_norms, pcgrad_combine,
                        read_hetero_cache, rotation_indices, task_grads)
from train_hetero import evaluate, train_features
from torch.utils.data import DataLoader
from src.hetero import subset_dataset


def test_rotations_cycle_all_angles():
    ids = torch.arange(12)
    values = torch.stack([rotation_indices(ids, ep, 0) for ep in range(1, 5)])
    assert sorted(values[:, 3].tolist()) == [0, 1, 2, 3]
    assert all(sorted(values[:, j].tolist()) == [0, 1, 2, 3] for j in range(len(ids)))
    assert torch.equal(rotation_indices(ids, 2, 8), rotation_indices(ids, 2, 8))


def test_rotation_train_feature_lookup():
    x = torch.arange(3*4*512, dtype=torch.float32).reshape(3, 4, 512)
    ids = torch.tensor([0, 15, 20])
    selected, angle = train_features(x, ids, "rotation", 7, 2)
    for i in range(3):
        assert torch.equal(selected[i], x[i, angle[i]])
    assert angle.dtype == torch.long


def test_pcgrad_projection_negatively_conflicting_grads():
    g1 = [torch.tensor([1., 0.])]
    g2 = [torch.tensor([-1., 1.])]
    original_cos = cosine_and_norms(g1, g2)[0]
    updated, conflict = pcgrad_combine(g1, g2)
    assert float(original_cos) < 0
    assert conflict
    assert torch.isfinite(updated[0]).all()
    # Each individual PCGrad-adjusted gradient has nonnegative dot with the other original.
    _, _, _, dot = cosine_and_norms(g1, g2)
    norm1 = (g1[0]**2).sum()
    norm2 = (g2[0]**2).sum()
    u1 = g1[0] - dot/norm2*g2[0]
    u2 = g2[0] - dot/norm1*g1[0]
    assert abs(torch.dot(u1, g2[0]).item()) < 1e-6
    assert abs(torch.dot(u2, g1[0]).item()) < 1e-6
    assert torch.allclose(updated[0], u1 + u2)


def test_pcgrad_no_conflict_equal_sum():
    g1 = [torch.tensor([1., 2.])]
    g2 = [torch.tensor([2., 1.])]
    out, conflict = pcgrad_combine(g1, g2)
    assert not conflict
    assert torch.equal(out[0], g1[0] + g2[0])


def test_model_shared_gradient_paths():
    torch.manual_seed(7)
    model = HeteroMLP(hidden_dim=16)
    x = torch.randn(8, 512)
    fine, aux = model(x)
    losses = [F.cross_entropy(fine, torch.randint(0, 100, (8,))),
              F.cross_entropy(aux, torch.randint(0, 4, (8,)))]
    g1, g2 = task_grads(losses, model.shared_parameters())
    cos, n1, n2, dot = cosine_and_norms(g1, g2)
    assert torch.isfinite(cos)
    assert n1 > 0 and n2 > 0
    assert -1.000001 <= cos.item() <= 1.000001
    assert torch.isfinite(dot)


def test_cache_validation_and_rotation_evaluate():
    with tempfile.TemporaryDirectory() as root:
        path = Path(root) / "test.pt"
        n = 8
        x = torch.randn(n, 4, 512).half()
        payload = {"features": x, "fine": torch.arange(n),
                   "aux": torch.full((n,), -1, dtype=torch.long),
                   "meta": {"scenario": "rotation", "normalized": True}}
        torch.save(payload, path)
        r = read_hetero_cache(path, "rotation")
        model = HeteroMLP(hidden_dim=16)
        results = evaluate(model, DataLoader(subset_dataset(r), batch_size=3),
                           "rotation", torch.device("cpu"))
        assert 0 <= results["aux_acc"] <= 100
        assert 0 <= results["fine_acc"] <= 100


def test_color_cache_evaluate():
    with tempfile.TemporaryDirectory() as root:
        path = Path(root) / "test.pt"
        payload = {"features": torch.randn(8, 512).half(),
                   "fine": torch.arange(8), "aux": torch.arange(8) % 4,
                   "meta": {"scenario": "color", "normalized": True}}
        torch.save(payload, path)
        r = read_hetero_cache(path, "color")
        model = HeteroMLP(hidden_dim=16)
        results = evaluate(model, DataLoader(subset_dataset(r), batch_size=3),
                           "color", torch.device("cpu"))
        assert results["aux_ce"] > 0


def test_pil_rotation_applied_before_transform():
    import numpy as np
    from PIL import Image
    from prepare_rotation_features import RotatedDataset

    arr = np.arange(16, dtype=np.uint8).reshape(4, 4)
    base = [(Image.fromarray(arr), 2)]
    transform = lambda image: torch.tensor(np.asarray(image).copy())
    original, i0 = RotatedDataset(base, transform, 0)[0]
    rotated, i1 = RotatedDataset(base, transform, 1)[0]
    assert i0 == i1 == 0
    assert torch.equal(original, torch.tensor(arr))
    assert torch.equal(rotated, torch.tensor(np.rot90(arr).copy()))


def test_color_score_red_blue_ordering():
    import pickle
    import numpy as np
    from prepare_color_features import raw_color_scores
    with tempfile.TemporaryDirectory() as root:
        base = Path(root) / 'cifar-100-python'
        base.mkdir()
        x = np.zeros((2, 3, 32, 32), dtype=np.uint8)
        x[0, 0] = 255  # red
        x[1, 2] = 255  # blue
        with (base / 'train').open('wb') as f:
            pickle.dump({b'data': x.reshape(2, 3072), b'fine_labels': [0, 1]}, f)
        scores = raw_color_scores(root, 'train', torch.tensor([0,1]))
        assert scores.tolist() == [255.0, -255.0]


def test_zero_degree_checkpoint_hash_check():
    from prepare_rotation_features import check_clean_base
    old = {"features": torch.randn(2, 512), "fine": torch.tensor([0, 1]),
           "coarse": torch.tensor([1, 2]),
           "meta": {"checkpoint_sha256": "x", "model_name": "ViT-B-32", "normalized": True}}
    check_clean_base(old, "x", "ViT-B-32", "train", old["fine"], old["coarse"])
    import pytest
    with pytest.raises(ValueError, match="different checkpoint"):
        check_clean_base(old, "y", "ViT-B-32", "train", old["fine"], old["coarse"])
