#!/usr/bin/env python3
"""Extract frozen OpenCLIP ViT-B/32 features from OFFLINE CIFAR-100 and weights."""
import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import official_coarse_labels
from src.offline_assets import digest, ensure_offline_cifar100, load_offline_openclip
from src.utils import device_from_string, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cifar-root", type=Path, default=Path("data"),
                        help="Parent directory for cifar-100-python/, default data/.")
    parser.add_argument("--cifar-archive", type=Path, default=Path("data/cifar-100-python.tar.gz"))
    parser.add_argument("--weights", type=Path, default=Path("weights/open_clip_model.safetensors"))
    parser.add_argument("--model-name", choices=["ViT-B-32", "ViT-B-32-quickgelu"],
                        default="ViT-B-32", help="Match model card; OpenAI original used QuickGELU.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--force", action="store_true", help="Replace all existing feature files.")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch-size must be positive; num-workers must be nonnegative.")

    out_train, out_test = args.out_dir / "train.pt", args.out_dir / "test.pt"
    if (out_train.exists() or out_test.exists()) and not args.force:
        if out_train.exists() and out_test.exists():
            print(f"Cached features already exist in {args.out_dir}. No changes made. "
                  "If you changed checkpoints, use --force and rerun experiments separately.")
            return
        raise FileExistsError("Only one feature cache exists. Inspect it; then pass --force.")

    ensure_offline_cifar100(args.cifar_root, args.cifar_archive)
    device = device_from_string(args.device)
    print(f"Loading local OpenCLIP {args.model_name} from {args.weights} on {device} ...")
    model, preprocess, check_info = load_offline_openclip(args.weights, args.model_name, device)
    print(f"Weight structure accepted: {check_info['visual_tensors']} visual tensors")
    if device.type != "cuda":
        print("WARNING: CLIP feature extraction on CPU is usually slow.")
    try:
        from torchvision.datasets import CIFAR100
    except ImportError as e:
        raise RuntimeError("Install torchvision compatible with your PyTorch build.") from e

    checkpoint_sha256 = digest(args.weights)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    backbone = f"OpenCLIP {args.model_name} (local checkpoint; training origin unverified)"
    common_meta = {
        "backbone": backbone,
        "model_name": args.model_name,
        "checkpoint_filename": args.weights.name,
        "checkpoint_sha256": checkpoint_sha256,
        "preprocess": "OpenCLIP default eval preprocess for chosen model name",
        "preprocess_config_source": "OpenCLIP built-in defaults, not independently verified against model card",
        "normalized": not args.no_normalize,
        "feature_dim": 512,
        "dtype": "float16",
        "torch_version": str(torch.__version__),
    }
    print("NOTE: The filename does not prove the weights are OpenAI official pretrained weights.")
    print("NOTE: For QuickGELU-pretrained weights use --model-name ViT-B-32-quickgelu.")
    for train, destination in ((True, out_train), (False, out_test)):
        split = "train" if train else "test"
        ds = CIFAR100(root=str(args.cifar_root), train=train,
                      transform=preprocess, download=False)
        coarse = official_coarse_labels(args.cifar_root, train, ds.targets)
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=(device.type == "cuda")
        )
        chunks = []
        print(f"Extracting {split}: {len(ds)} images ...")
        with torch.inference_mode():
            for images, _ in tqdm(loader, desc=f"OpenCLIP {split}", unit="batch"):
                images = images.to(device, non_blocking=True)
                context = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
                with context:
                    emb = model.encode_image(images)
                emb = emb.float()
                if emb.ndim != 2 or emb.shape[1] != 512:
                    raise RuntimeError(f"Expected 512-d ViT-B/32 embeddings, got shape {tuple(emb.shape)}")
                if not args.no_normalize:
                    emb = F.normalize(emb, p=2, dim=-1)
                if not torch.isfinite(emb).all():
                    raise FloatingPointError("Nonfinite embedding! Check checkpoint, preprocess, device")
                chunks.append(emb.to(device="cpu", dtype=torch.float16))
        features = torch.cat(chunks, dim=0)
        payload = {
            "features": features,
            "fine": torch.tensor(ds.targets, dtype=torch.long),
            "coarse": coarse,
            "meta": {**common_meta, "split": split, "count": len(ds)},
        }
        temp_path = destination.with_suffix(".pt.tmp")
        torch.save(payload, temp_path)
        temp_path.replace(destination)
        print(f"Saved {destination} | {tuple(features.shape)} | {destination.stat().st_size / 1e6:.1f} MB")

    write_json(args.out_dir / "cache_info.json", {
        **common_meta, "train_file": str(out_train), "test_file": str(out_test),
        "note": "Offline frozen features; checkpoint training provenance not independently verified."
    })
    print("Offline feature extraction complete. Next: python run_all.py --quick")


if __name__ == "__main__":
    main()
