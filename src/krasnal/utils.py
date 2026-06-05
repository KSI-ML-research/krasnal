import json
import os
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
        f"context={config.block_size}, vocab={vocab_size}, mlp={config.mlp_activation}\n"
        f"  Device: {device}  |  dtype: {dtype}  |  compile: {compile_enabled}\n"
        f"  Artifact dir: {artifact_dir.name}\n"
        f"{'=' * 60}"
    )


def ablation_metadata_from_env() -> dict[str, str]:
    run_name = os.environ.get("RUN_NAME", "")
    if not run_name:
        return {}

    metadata = {
        "ablation_name": run_name,
        "ablation_group": os.environ.get("RUN_GROUP", ""),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME", ""),
        "tokenized_dir": os.environ.get("KRASNAL_TOKENIZED_DIR", ""),
        "wandb_name": os.environ.get("WANDB_NAME", ""),
        "wandb_group": os.environ.get("WANDB_RUN_GROUP", ""),
    }
    if "-" in run_name:
        data_variant, train_variant = run_name.rsplit("-", 1)
        metadata["ablation_data_variant"] = data_variant
        metadata["ablation_train_variant"] = train_variant
    return {key: value for key, value in metadata.items() if value}


def ablation_tags(metadata: dict[str, str]) -> tuple[str, ...]:
    if not metadata:
        return ()
    tags = ["ablation"]
    for key in ("ablation_data_variant", "ablation_train_variant"):
        if value := metadata.get(key):
            tags.append(value)
    return tuple(tags)


def init_wandb(
    *,
    project: str,
    config: dict,
    stage: str,
    name: str | None = None,
    group: str | None = None,
    tags: tuple[str, ...] = (),
) -> tuple[str, str, str]:
    """Initialize wandb (tagged with stage) and return run URL components."""
    init_kwargs: dict[str, Any] = {
        "project": project,
        "config": config,
        "tags": list(dict.fromkeys((stage, *tags))),
    }
    if name:
        init_kwargs["name"] = name
    if group:
        init_kwargs["group"] = group
    wandb.init(**init_kwargs)
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
    write_artifact_config_json(artifact_dir, run_config)
    with open(artifact_dir / "wandb_run_link.txt", "w") as f:
        f.write(f"{wandb_run_url}\n")

    artifact = wandb.Artifact(artifact_name, type=artifact_type)
    artifact.add_dir(str(artifact_dir))
    wandb.log_artifact(artifact)


def format_eval_metric_key(key: str) -> str:
    if key == "val_loss":
        return "eval/val_loss"
    if key.startswith("qa/"):
        return f"eval/{key}"
    if key.startswith("acc_elo_"):
        return f"eval/game/elo/{key}"
    return f"eval/game/{key}"


def log_eval_metrics_to_wandb(metrics: dict[str, Any], *, step: int | None = None) -> None:
    payload = {format_eval_metric_key(k): v for k, v in metrics.items()}
    for qa_key in ("qa/what_is_on/acc_matrix", "qa/what_is_on/acc_matrix_baseline"):
        if qa_key in metrics:
            heatmap = metrics[qa_key]
            key = format_eval_metric_key(qa_key)
            payload[key] = heatmap
            wandb.run.summary[key] = heatmap  # type: ignore[index]
    if step is None:
        wandb.log(payload)
    else:
        wandb.log(payload, step=step)


REQUIRED_CONFIG_KEYS = {
    "block_size",
    "n_layer",
    "n_head",
    "n_embd",
    "vocab_size",
    "dropout",
    "use_clock_encodings",
    "clock_encoding_hidden",
    "mlp_activation",
}


def write_artifact_config_json(artifact_dir: Path, run_config: dict[str, Any]) -> None:
    """Write ``config.json`` at run start (architecture metadata before checkpoint save)."""
    missing = REQUIRED_CONFIG_KEYS - run_config.keys()
    if missing:
        raise ValueError(
            "run_config missing inference keys: " + ", ".join(sorted(missing)),
        )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with (artifact_dir / "config.json").open("w") as f:
        json.dump(run_config, f, indent=2)


def resolve_runtime_device() -> torch.device:
    """Pick the best available inference device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_model_config_json(config_path: Path) -> dict[str, Any]:
    """Load and validate ``config.json`` (training / inference manifest)."""
    with config_path.open() as f:
        config = json.load(f)
    missing = REQUIRED_CONFIG_KEYS - config.keys()
    if missing:
        missing_keys = ", ".join(sorted(missing))
        raise ValueError(f"{config_path} is missing required model keys: {missing_keys}")
    return config


def gpt_config_from_artifact_dict(config: dict[str, Any]) -> GPTConfig:
    """Build ``GPTConfig`` from a validated artifact config dict (single source for field names)."""
    return GPTConfig(
        block_size=int(config["block_size"]),
        n_layer=int(config["n_layer"]),
        n_head=int(config["n_head"]),
        n_embd=int(config["n_embd"]),
        vocab_size=int(config["vocab_size"]),
        dropout=float(config["dropout"]),
        use_clock_encodings=bool(config["use_clock_encodings"]),
        clock_encoding_hidden=int(config["clock_encoding_hidden"]),
        mlp_activation=str(config["mlp_activation"]),
    )
