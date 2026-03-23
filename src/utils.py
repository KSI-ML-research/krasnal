import random
from pathlib import Path

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


def print_model_config(
    *,
    stage: str,
    params_m: float,
    dataset_size: int,
    dataset_label: str,
    config,
    vocab_size: int,
    device: str,
    dtype: str,
    compile_enabled: bool,
    artifact_dir: Path,
) -> None:
    """Print a compact model and runtime summary."""
    print(
        f"{'=' * 60}\n"
        f"  {stage}  |  {params_m:.2f}M params  |  {dataset_size:,} {dataset_label}\n"
        f"  layers={config.n_layer}, heads={config.n_head}, embd={config.n_embd}, "
        f"context={config.block_size}, vocab={vocab_size}\n"
        f"  Device: {device}  |  dtype: {dtype}  |  compile: {compile_enabled}\n"
        f"  Artifact dir: {artifact_dir.name}\n"
        f"{'=' * 60}"
    )
