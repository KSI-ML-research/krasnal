import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
