#!/usr/bin/env python3
"""Check RTX 50-series CUDA compatibility; optionally load OFFLINE OpenCLIP."""
import argparse
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-clip", action="store_true", help="Load local checkpoint and run one dummy image.")
    parser.add_argument("--weights", type=Path, default=Path("weights/open_clip_model.safetensors"))
    parser.add_argument("--model-name", choices=["ViT-B-32", "ViT-B-32-quickgelu"], default="ViT-B-32")
    args = parser.parse_args()
    print("Python:", sys.version.split()[0])
    print("PyTorch:", torch.__version__)
    print("Built with CUDA:", torch.version.cuda)
    try:
        import torchvision
        print("torchvision:", torchvision.__version__)
    except Exception as ex:
        print("torchvision import failed:", repr(ex))
        raise SystemExit(1) from ex

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        capability = torch.cuda.get_device_capability(0)
        print("Compute capability:", ".".join(map(str, capability)))
        arches = torch.cuda.get_arch_list()
        print("PyTorch supported architectures:", ", ".join(arches))
        if capability >= (12, 0) and "sm_120" not in arches:
            print("WARNING: This PyTorch build may not support RTX 50-series sm_120.")
        x = torch.randn(128, 128, device="cuda")
        y = (x @ x).sum()
        torch.cuda.synchronize()
        print("CUDA matrix multiplication: PASS", float(y.item()))
    else:
        print("CUDA: unavailable. Cached feature training can run on CPU.")

    if args.test_clip:
        from src.offline_assets import load_offline_openclip
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, _ = load_offline_openclip(args.weights, args.model_name, device=device)
        with torch.inference_mode():
            output = model.encode_image(torch.randn(1, 3, 224, 224, device=device))
        print("Local OpenCLIP inference output:", tuple(output.shape))
        assert output.shape == (1, 512), "Unexpected embedding dimension"
        print("Local OpenCLIP test: PASS (pretraining provenance still unverified)")


if __name__ == "__main__":
    main()
