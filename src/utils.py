import json
import math
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from_string(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Run python check_env.py.")
    return torch.device(value)


def write_json(path, obj):
    def to_jsonable(x):
        if isinstance(x, dict):
            return {str(k): to_jsonable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [to_jsonable(v) for v in x]
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, (np.integer, np.floating)):
            return x.item()
        if isinstance(x, float) and not math.isfinite(x):
            return None
        return x

    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(
        json.dumps(to_jsonable(obj), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
