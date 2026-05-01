import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

import wandb
from krasnal.config import GPTConfig


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


def init_wandb(
    *,
    project: str,
    config: dict,
    stage: str,
) -> tuple[str, str, str]:
    """Initialize wandb (tagged with stage) and return run URL components."""
    wandb.init(project=project, config=config, tags=[stage])
    run_id = wandb.run.id  # type: ignore[union-attr]
    entity = wandb.run.entity  # type: ignore[union-attr]
    proj = wandb.run.project  # type: ignore[union-attr]
    return run_id, entity, proj


def save_wandb_run(
    *,
    artifact_dir: Path,
    run_config: dict,
    wandb_run_url: str,
    artifact_name: str,
    artifact_type: str,
) -> None:
    """Save config, run URL, and log artifact to wandb."""
    with open(artifact_dir / "config.json", "w") as f:
        json.dump(run_config, f, indent=2)
    with open(artifact_dir / "wandb_run_link.txt", "w") as f:
        f.write(f"{wandb_run_url}\n")

    artifact = wandb.Artifact(artifact_name, type=artifact_type)
    artifact.add_dir(str(artifact_dir))
    wandb.log_artifact(artifact)


def format_eval_metric_key(key: str) -> str:
    if key == "val_loss":
        return "eval/val_loss"
    if key.startswith("qa/") or key.startswith("cot_"):
        return f"eval/{key}"
    return f"eval/game/{key}"


REQUIRED_CONFIG_KEYS = {"block_size", "n_layer", "n_head", "n_embd"}


def resolve_runtime_device() -> torch.device:
    """Pick the best available inference device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_artifact_config(config_path: Path) -> dict[str, Any]:
    with config_path.open() as f:
        config = json.load(f)
    missing = REQUIRED_CONFIG_KEYS - config.keys()
    if missing:
        missing_keys = ", ".join(sorted(missing))
        raise ValueError(f"{config_path} is missing required model keys: {missing_keys}")
    return config


def build_gpt_config_from_artifact(artifact_dir: Path, vocab_size: int | None = None) -> GPTConfig:
    """Build GPTConfig from an artifact directory."""
    config = _load_artifact_config(artifact_dir / "config.json")
    return GPTConfig(
        block_size=int(config["block_size"]),
        n_layer=int(config["n_layer"]),
        n_head=int(config["n_head"]),
        n_embd=int(config["n_embd"]),
        vocab_size=vocab_size if vocab_size is not None else config.get("vocab_size"),
        dropout=float(config.get("dropout", 0.0)),
    )
