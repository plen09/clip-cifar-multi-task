"""Offline asset checks for locally downloaded CIFAR-100 and OpenCLIP checkpoints.

No downloads are performed. CIFAR-100's archive is extracted using regular files
only (no symlinks, hardlinks, or untrusted archive paths).
"""
import hashlib
import shutil
import tarfile
from pathlib import Path, PurePosixPath

CIFAR_MD5 = "eb9058c3a382ffc7106e4002c42a8d85"
CIFAR_FILES = ("train", "test", "meta")
VISUAL_TENSORS = {
    "visual.conv1.weight": [768, 3, 32, 32],
    "visual.positional_embedding": [50, 768],
    "visual.proj": [768, 512],
}


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_offline_cifar100(root, archive, verify_md5=True):
    """Extract the official archive only if root/cifar-100-python is absent."""
    root, archive = Path(root), Path(archive)
    base = root / "cifar-100-python"
    if all((base / file).is_file() for file in CIFAR_FILES):
        return base
    if not archive.is_file():
        raise FileNotFoundError(
            f"CIFAR-100 data missing. Expected extracted directory {base}, "
            f"or archive {archive}."
        )
    if verify_md5 and digest(archive, "md5") != CIFAR_MD5:
        raise ValueError(
            f"CIFAR-100 archive MD5 mismatch: {archive}. It may be incomplete "
            "or not the official cifar-100-python.tar.gz."
        )

    print(f"Extracting {archive} -> {root} (offline)...")
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            name = PurePosixPath(member.name)
            if (name.is_absolute() or '..' in name.parts or not name.parts
                or name.parts[0] != 'cifar-100-python'
                or not (member.isfile() or member.isdir())):
                raise ValueError(f"Unsafe/unexpected CIFAR archive member: {member.name!r}")
        # Avoid tar.extractall: manually copy only normal files and directories.
        for member in members:
            destination = root.joinpath(*PurePosixPath(member.name).parts)
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                destination.parent.mkdir(parents=True, exist_ok=True)
                stream = tar.extractfile(member)
                if stream is None:
                    raise ValueError(f"Cannot read {member.name} from archive")
                with stream, destination.open("wb") as output:
                    shutil.copyfileobj(stream, output)
    if not all((base / f).is_file() for f in CIFAR_FILES):
        raise ValueError(f"Missing official CIFAR-100 train/test/meta after extracting {archive}")
    return base


def inspect_vit_b32_safetensors(path):
    """Check the *visual* tensors are the expected ViT-B/32 512-d architecture.

    A matching tensor structure does NOT establish a specific pretraining origin,
    activation type (GELU vs QuickGELU), or preprocessing config.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Local weights not found: {path}")
    if path.suffix != ".safetensors":
        raise ValueError(f"Expected .safetensors checkpoint, got: {path}")
    try:
        from safetensors import safe_open
    except ImportError as e:
        raise RuntimeError("Install safetensors: python -m pip install safetensors") from e
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        keys = set(checkpoint.keys())
        missing = [key for key in VISUAL_TENSORS if key not in keys]
        if missing:
            sample = ', '.join(sorted(keys)[:7])
            raise ValueError(
                f"Checkpoint is not a directly loadable OpenCLIP ViT-B/32 state dict: "
                f"missing {missing}. Example keys: {sample}. "
                "It may be a Hugging Face Transformers checkpoint or another architecture."
            )
        for key, expected in VISUAL_TENSORS.items():
            actual = checkpoint.get_slice(key).get_shape()
            if list(actual) != expected:
                raise ValueError(
                    f"Wrong visual architecture: {key} shape={actual}, expected={expected} "
                    "for ViT-B/32 at 224px with 512-d output."
                )
        visual_count = sum(k.startswith('visual.') for k in keys)
        if visual_count < 100:
            raise ValueError(f"Too few visual tensors ({visual_count}); checkpoint seems incomplete.")
        return {"checkpoint_tensors": len(keys), "visual_tensors": visual_count,
                "visual_conv1_shape": VISUAL_TENSORS["visual.conv1.weight"]}


def load_offline_openclip(path, model_name="ViT-B-32", device="cpu"):
    """Load local OpenCLIP checkpoint without contacting Hugging Face or GitHub."""
    info = inspect_vit_b32_safetensors(path)
    if model_name not in ("ViT-B-32", "ViT-B-32-quickgelu"):
        raise ValueError("Model name must be ViT-B-32 or ViT-B-32-quickgelu")
    try:
        import open_clip
    except ImportError as e:
        raise RuntimeError("Install open_clip_torch: python -m pip install open_clip_torch") from e
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=str(Path(path).resolve()), device=device
    )
    # OpenCLIP's loading can be permissive (non-strict). Refuse incomplete
    # visual weights, because we only use the image tower for feature extraction.
    from safetensors import safe_open
    with safe_open(str(path), framework="pt", device="cpu") as ckpt:
        checkpoint_keys = set(ckpt.keys())
        visual_state = {name: tensor for name, tensor in model.state_dict().items()
                        if name.startswith("visual.")}
        missing = [name for name in visual_state if name not in checkpoint_keys]
        mismatched = [name for name, tensor in visual_state.items()
                      if name in checkpoint_keys and
                      list(tensor.shape) != list(ckpt.get_slice(name).get_shape())]
    if missing or mismatched:
        raise ValueError(
            f"Checkpoint does not fully match {model_name} visual tower. "
            f"Missing tensors: {missing[:8]}; shape mismatches: {mismatched[:8]}. "
            "Check the model card's architecture and use the corresponding --model-name."
        )
    model.eval().requires_grad_(False)
    return model, preprocess, info
