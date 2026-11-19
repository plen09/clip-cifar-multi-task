#!/usr/bin/env python3
"""Offline four-orientation CLIP ViT-B/32 feature extraction for CIFAR-100.

Uses already-cached clean 0-degree features if model hash/preprocess match,
then extracts 90, 180 and 270-degree views from raw local CIFAR images.
All angles are counterclockwise PIL 90-degree rotations *before* CLIP transform.
"""
import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image

from src.data import official_coarse_labels, read_feature_file
from src.offline_assets import digest, ensure_offline_cifar100, load_offline_openclip
from src.utils import device_from_string, write_json


class RotatedDataset(Dataset):
    def __init__(self, base, transform, angle_index):
        self.base, self.transform, self.angle_index = base, transform, int(angle_index)
        self.rotations = (None, Image.Transpose.ROTATE_90,
                          Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        pil, _ = self.base[index]
        operation = self.rotations[self.angle_index]
        if operation is not None:
            pil = pil.transpose(operation)
        return self.transform(pil), index


def check_clean_base(payload, weights_sha, model_name, split, fine, coarse):
    meta = payload["meta"]
    if (meta.get("checkpoint_sha256") != weights_sha
            or meta.get("model_name") != model_name or not meta.get("normalized")):
        raise ValueError(f"Existing 0-degree {split} feature cache uses a different checkpoint, "
                         "model name or non-normalized features. Use --no-reuse-clean "
                         "to compute 0-degree embeddings with the current weights.")
    if not torch.equal(payload["fine"].long(), fine) or not torch.equal(payload["coarse"].long(), coarse):
        raise ValueError("Existing 0-degree feature cache has a different CIFAR sample order")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cifar-root", type=Path, default=Path("data"))
    parser.add_argument("--cifar-archive", type=Path, default=Path("data/cifar-100-python.tar.gz"))
    parser.add_argument("--weights", type=Path, default=Path("weights/open_clip_model.safetensors"))
    parser.add_argument("--model-name", choices=["ViT-B-32", "ViT-B-32-quickgelu"],
                        default="ViT-B-32")
    parser.add_argument("--clean-features-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/hetero_rotation"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--no-reuse-clean", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.num_workers < 0:
        parser.error("batch-size must be positive; num-workers nonnegative")
    out_train, out_test = args.out_dir / "train.pt", args.out_dir / "test.pt"
    if (out_train.exists() or out_test.exists()) and not args.force:
        parser.error("Rotation cache exists: use --force only if replacing it intentionally")
    ensure_offline_cifar100(args.cifar_root, args.cifar_archive)
    device = device_from_string(args.device)
    model, preprocess, _ = load_offline_openclip(args.weights, args.model_name, device)
    weights_sha = digest(args.weights)
    from torchvision.datasets import CIFAR100
    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta_common = {
        "scenario": "rotation", "feature_dim": 512, "num_rotations": 4,
        "angles_ccw_deg": [0, 90, 180, 270], "normalized": True,
        "backbone": f"OpenCLIP {args.model_name} (local checkpoint; training origin unverified)",
        "model_name": args.model_name, "checkpoint_sha256": weights_sha,
        "preprocess": "OpenCLIP built-in eval preprocess for model-name",
        "pretrained_origin": "unverified; inspect your downloaded model card",
        "float_dtype": "float16", "rotation_before_clip_preprocess": True,
    }
    for is_train, dest in ((True, out_train), (False, out_test)):
        split = "train" if is_train else "test"
        base = CIFAR100(root=str(args.cifar_root), train=is_train, transform=None, download=False)
        fine = torch.tensor(base.targets, dtype=torch.long)
        coarse = official_coarse_labels(args.cifar_root, is_train, base.targets)
        n = len(base)
        feats = torch.empty((n, 4, 512), dtype=torch.float16)
        reuse = (not args.no_reuse_clean)
        if reuse:
            path = args.clean_features_dir / f"{split}.pt"
            if path.is_file():
                old = read_feature_file(path)
                check_clean_base(old, weights_sha, args.model_name, split, fine, coarse)
                feats[:, 0] = old["features"].half()
                print(f"Reused {split} cached 0-degree embeddings from {path}", flush=True)
            else:
                print(f"No clean cache for {split}; computing all four rotations", flush=True)
                reuse = False
        for angle in range(1 if reuse else 0, 4):
            ds = RotatedDataset(base, preprocess, angle)
            dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
            with torch.inference_mode():
                for imgs, indices in tqdm(dl, desc=f"{split}: rotate {90*angle:3d}°", unit="batch"):
                    images = imgs.to(device, non_blocking=True)
                    context = (torch.autocast("cuda", dtype=torch.float16)
                               if device.type == "cuda" else nullcontext())
                    with context:
                        y = model.encode_image(images).float()
                    if y.shape != (len(images), 512):
                        raise ValueError(f"Unexpected CLIP output shape {tuple(y.shape)}")
                    y = F.normalize(y, dim=-1)
                    if not torch.isfinite(y).all():
                        raise FloatingPointError("Non-finite CLIP embedding encountered")
                    feats[indices.long(), angle, :] = y.cpu().half()
        payload = {"features": feats, "fine": fine,
                   "aux": torch.full((n,), -1, dtype=torch.long),
                   "meta": {**meta_common, "split": split, "count": n,
                            "zero_degree_reused": reuse}}
        tmp = dest.with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.replace(dest)
        print(f"Saved {dest}: shape={tuple(feats.shape)}, size={dest.stat().st_size/1e6:.1f} MB", flush=True)
    write_json(args.out_dir / "cache_info.json", {
        **meta_common, "clean_features_dir": str(args.clean_features_dir),
        "train_shape": [50000, 4, 512], "test_shape": [10000, 4, 512],
        "note": "Frozen backbone; rotation task is not guaranteed to generate negative gradients.",
    })
    print("Next: python run_hetero.py --scenario rotation --quick", flush=True)


if __name__ == "__main__":
    main()
