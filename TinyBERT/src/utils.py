import json
import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model) -> float:
    """Size of parameters + buffers in MB. Use the SAME method for every model."""
    size = sum(p.nelement() * p.element_size() for p in model.parameters())
    size += sum(b.nelement() * b.element_size() for b in model.buffers())
    return size / (1024 * 1024)


def save_json(obj: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
