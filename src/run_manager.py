import hashlib
import json
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch.nn as nn

from config import RUNS_DIR


def compute_run_hash(
    stage: str,
    model_config: Any,
    train_config: Any,
    seed: int,
    model_repr: str,
    dataset_mtime: int,
) -> str:
    """Generate a deterministic 8-character hash for a run configuration.

    Args:
        stage: Training stage identifier (e.g., "pretrain", "finetune").
        model_config: Model configuration object (dataclass or dict).
        train_config: Training configuration object (dataclass or dict).
        seed: Random seed used for the run.
        model_repr: String representation of the model architecture.
        dataset_mtime: Modification time (Unix timestamp) of the dataset file.

    Returns:
        An 8-character hex string representing the SHA256 hash of the configuration.
    """
    config_dict = {
        "stage": stage,
        "model": _dataclass_to_dict(model_config),
        "train": _dataclass_to_dict(train_config),
        "seed": seed,
        "model_repr": model_repr,
        "dataset_mtime": dataset_mtime,
    }
    payload = json.dumps(config_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def _dataclass_to_dict(obj: Any) -> dict[str, Any]:
    """Recursively convert dataclass objects and nested structures to dictionaries.

    Handles dataclasses, lists, tuples, and nested dicts. Primitive values are
    returned as-is.

    Args:
        obj: A dataclass, dict, list, tuple, or primitive value to convert.

    Returns:
        A dictionary representation of the input, or the input itself if primitive.
    """
    if is_dataclass(obj):
        return {k: _dataclass_to_dict(v) for k, v in asdict(obj).items()}
    elif isinstance(obj, (list, tuple)):
        return [_dataclass_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    return obj


def compute_params_M(model: nn.Module) -> int:
    """Calculate the model's parameter count in millions.

    Args:
        model: A PyTorch model with a get_num_params() method.

    Returns:
        The number of model parameters rounded to the nearest million.
    """
    return round(model.get_num_params() / 1_000_000)


def get_git_commit_hash() -> str:
    """Retrieve the current git commit hash.

    Returns:
        The short (7-character) git commit hash, or "nogit" if git is unavailable.
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError, FileNotFoundError:
        return "nogit"


def get_git_status() -> str:
    """Check if the working directory has uncommitted changes.

    Returns:
        "clean" if working directory is clean, "dirty" if there are uncommitted
        changes, or "unknown" if git is unavailable.
    """
    try:
        result = subprocess.check_output(
            ["git", "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return "dirty" if result else "clean"
    except subprocess.CalledProcessError, FileNotFoundError:
        return "unknown"


def create_run_folder(
    stage: str,
    params_M: int,
    model_config: Any,
    train_config: Any,
    seed: int,
    model_repr: str = "",
    dataset_mtime: int = 0,
) -> tuple[Path, str, str]:
    """Create a run directory for the given configuration.

    Creates a folder in RUNS_DIR with the name format:
    {stage}_{params_M}M_{run_hash}_{commit_hash}

    Args:
        stage: Training stage identifier.
        params_M: Model parameter count in millions.
        model_config: Model configuration object.
        train_config: Training configuration object.
        seed: Random seed for the run.
        model_repr: Optional model architecture representation.
        dataset_mtime: Modification time of the dataset file.

    Returns:
        A tuple of (folder_path, run_hash, commit_hash).
    """
    run_hash = compute_run_hash(
        stage, model_config, train_config, seed, model_repr=model_repr, dataset_mtime=dataset_mtime
    )
    commit_hash = get_git_commit_hash()

    folder_name = f"{stage}_{params_M}M_{run_hash}_{commit_hash}"
    folder_path = RUNS_DIR / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)

    return folder_path, run_hash, commit_hash


def find_runs_by_hash(run_hash: str, stage: str | None = None) -> list[Path]:
    """Find all runs matching a given configuration hash.

    Searches through all run folders in RUNS_DIR and returns those whose
    config.json contains the matching run_hash. Optionally filters by stage.

    Args:
        run_hash: The run hash to search for.
        stage: Optional stage identifier to further filter results.

    Returns:
        A sorted list of Path objects pointing to matching run folders.
        Returns an empty list if RUNS_DIR doesn't exist or no matches found.
    """
    if not RUNS_DIR.exists():
        return []
    matching = []
    for folder in RUNS_DIR.iterdir():
        if not folder.is_dir():
            continue
        config_file = folder / "config.json"
        if not config_file.exists():
            continue
        try:
            with config_file.open() as f:
                config = json.load(f)
            if config.get("run_hash") == run_hash:
                if stage is None or config.get("stage") == stage:
                    matching.append(folder)
        except OSError, json.JSONDecodeError:
            continue
    matching.sort(key=lambda p: p.name)
    return matching


def find_run_by_hash(run_hash: str, stage: str | None = None) -> Path | None:
    """Find a unique run matching a given configuration hash.

    Uses find_runs_by_hash and enforces uniqueness of hash-to-folder mapping.

    Args:
        run_hash: The run hash to search for.
        stage: Optional stage identifier to further filter results.

    Returns:
        The Path to the matching run, or None if no match is found.

    Raises:
        ValueError: If more than one run matches the hash.
    """
    runs = find_runs_by_hash(run_hash, stage=stage)
    if not runs:
        return None
    if len(runs) > 1:
        names = ", ".join(run.name for run in runs)
        raise ValueError(f"Multiple runs found for hash {run_hash}: {names}")
    return runs[0]


def find_latest_run(stage: str | None = None) -> Path | None:
    """Find the most recent run folder, optionally filtered by stage.

    Recency is determined lexicographically by folder name, which includes
    a sortable timestamp segment (`YYYYMMDD_HHMMSS`).

    Args:
        stage: Optional stage identifier to filter runs by `config.json`.

    Returns:
        Path to the most recent run folder or None when no run is found.
    """
    if not RUNS_DIR.exists():
        return None

    runs: list[Path] = []
    for folder in RUNS_DIR.iterdir():
        if not folder.is_dir():
            continue
        config_file = folder / "config.json"
        if not config_file.exists():
            continue
        if stage is None:
            runs.append(folder)
            continue
        try:
            with config_file.open() as f:
                config = json.load(f)
        except OSError, json.JSONDecodeError:
            continue
        if config.get("stage") == stage:
            runs.append(folder)

    if not runs:
        return None
    runs.sort(key=lambda p: p.name)
    return runs[-1]


def save_run_config(
    folder: Path,
    stage: str,
    run_hash: str,
    params_M: int,
    model_config: Any,
    train_config: Any,
    seed: int,
    commit_hash: str,
    parent: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save run configuration and metadata to a config.json file.

    Creates a config.json file in the specified folder containing all run
    configuration parameters, timestamps, git information, and optional
    custom fields.

    Args:
        folder: Directory where config.json will be saved.
        stage: Training stage identifier.
        run_hash: The run's configuration hash.
        params_M: Model parameter count in millions.
        model_config: Model configuration object.
        train_config: Training configuration object.
        seed: Random seed used in the run.
        commit_hash: Git commit hash.
        parent: Optional hash of a parent run (for multi-stage training).
        extra: Optional dict of additional fields to include in the config.

    Returns:
        The Path to the created config.json file.
    """
    config = {
        "stage": stage,
        "run_hash": run_hash,
        "params_M": params_M,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "commit_hash": commit_hash,
        "git_status": get_git_status(),
        "parent": parent,
        "model_config": _dataclass_to_dict(model_config),
        "train_config": _dataclass_to_dict(train_config),
        "seed": seed,
    }
    if extra:
        config.update(extra)

    config_path = folder / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
    return config_path


def update_run_config(folder: Path, updates: dict[str, Any]) -> None:
    """Update run configuration by merging new values into the config.json file.

    Loads existing config.json (or creates an empty config if it doesn't exist),
    merges in the provided updates, and writes back to disk.

    Args:
        folder: Directory containing or where config.json will be created.
        updates: Dictionary of key-value pairs to merge into the config.
    """
    config_path = folder / "config.json"
    if config_path.exists():
        with config_path.open() as f:
            config = json.load(f)
    else:
        config = {}
    config.update(updates)
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
