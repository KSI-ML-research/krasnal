import hashlib
import json
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path

import torch.nn as nn

from .config import RUNS_DIR


def compute_run_hash(model_config, train_config, seed: int) -> str:
    config_dict = {
        "model": _dataclass_to_dict(model_config),
        "train": _dataclass_to_dict(train_config),
        "seed": seed,
    }
    payload = json.dumps(config_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def _dataclass_to_dict(obj) -> dict:
    if is_dataclass(obj):
        return {k: _dataclass_to_dict(v) for k, v in asdict(obj).items()}
    elif isinstance(obj, (list, tuple)):
        return [_dataclass_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    return obj


def compute_params_M(model: nn.Module) -> int:
    return round(model.get_num_params() / 1_000_000)


def get_git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def get_git_status() -> str:
    try:
        result = subprocess.check_output(
            ["git", "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return "dirty" if result else "clean"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def create_run_folder(
    stage: str,
    params_M: int,
    model_config,
    train_config,
    seed: int,
    _parent: str | None = None,
) -> tuple[Path, str, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_hash = compute_run_hash(model_config, train_config, seed)
    commit_hash = get_git_commit_hash()

    folder_name = f"{stage}_{params_M}M_{timestamp}_{run_hash}_{commit_hash}"
    folder_path = RUNS_DIR / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)

    return folder_path, run_hash, commit_hash


def find_runs_by_hash(run_hash: str, stage: str | None = None) -> list[Path]:
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
        except (OSError, json.JSONDecodeError):
            continue
    matching.sort(key=lambda p: p.name)
    return matching


def find_run_by_hash(run_hash: str, stage: str | None = None) -> Path | None:
    runs = find_runs_by_hash(run_hash, stage=stage)
    if not runs:
        return None
    return runs[-1]


def find_latest_run(stage: str) -> Path | None:
    if not RUNS_DIR.exists():
        return None
    matching = []
    for folder in RUNS_DIR.iterdir():
        if not folder.is_dir():
            continue
        if not folder.name.startswith(stage + "_"):
            continue
        config_file = folder / "config.json"
        if not config_file.exists():
            continue
        try:
            with config_file.open() as f:
                config = json.load(f)
            if config.get("stage") == stage:
                matching.append(folder)
        except (OSError, json.JSONDecodeError):
            continue
    if not matching:
        return None
    matching.sort(key=lambda p: p.name)
    return matching[-1]


def save_run_config(
    folder: Path,
    stage: str,
    run_hash: str,
    params_M: int,
    model_config,
    train_config,
    seed: int,
    commit_hash: str,
    parent: str | None = None,
    parent_hash: str | None = None,
    extra: dict | None = None,
):
    config = {
        "stage": stage,
        "run_hash": run_hash,
        "params_M": params_M,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "commit_hash": commit_hash,
        "git_status": get_git_status(),
        "parent": parent,
        "parent_hash": parent_hash,
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


def update_run_config(folder: Path, updates: dict):
    config_path = folder / "config.json"
    if config_path.exists():
        with config_path.open() as f:
            config = json.load(f)
    else:
        config = {}
    config.update(updates)
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
