from typing import Dict, Any
import yaml, random, os
import numpy as np
import torch

def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device(name: str):
    return torch.device(name if (name == "cpu" or torch.cuda.is_available()) else "cpu")
