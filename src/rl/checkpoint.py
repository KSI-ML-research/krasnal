from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from config import ARTIFACTS_DIR
from trainer import save_model_state


def resolve_latest_pretrain_path() -> Path:
    pretrain_dir = ARTIFACTS_DIR / "pretrain"
    if not pretrain_dir.exists():
        raise FileNotFoundError(f"No artifacts found in {pretrain_dir}")

    subdirs = sorted(
        (path for path in pretrain_dir.iterdir() if path.is_dir()),
        key=lambda p: p.stat().st_mtime,
    )
    if not subdirs:
        raise FileNotFoundError(f"No pretrain runs found in {pretrain_dir}")

    model_path = subdirs[-1] / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Latest pretrain checkpoint not found at {model_path}")
    return model_path


def resolve_pretrained_checkpoint(model_path: str | None, latest_pretrain: bool) -> Path:
    if bool(model_path) == bool(latest_pretrain):
        raise ValueError("Use exactly one of --model or --latest-pretrain")
    if latest_pretrain:
        return resolve_latest_pretrain_path()

    resolved = Path(model_path).expanduser()  # type: ignore[arg-type]
    if resolved.is_dir():
        resolved = resolved / "model.pt"
    if not resolved.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {resolved}")
    return resolved


@dataclass
class CheckpointTimer:
    interval_seconds: float
    time_fn: Any = time.monotonic

    def __post_init__(self) -> None:
        if self.interval_seconds <= 0:
            raise ValueError("interval_seconds must be > 0")
        if self.time_fn is None:
            self.time_fn = time.monotonic
        self.last_save_time = float(self.time_fn())

    def should_save(self) -> bool:
        return float(self.time_fn()) - self.last_save_time >= self.interval_seconds

    def mark_saved(self) -> None:
        self.last_save_time = float(self.time_fn())


def save_checkpoint(
    model: torch.nn.Module,
    *,
    tokenizer,
    checkpoint_root: Path,
    iter_num: int,
    kind: str,
    metadata: dict[str, Any],
) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    checkpoint_dir = checkpoint_root / f"{kind}_iter_{iter_num:06d}_{timestamp}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    save_model_state(model, checkpoint_dir / "model.pt", tokenizer=tokenizer)
    payload = {"iter_num": iter_num, "kind": kind, **metadata}
    with open(checkpoint_dir / "metadata.json", "w") as f:
        json.dump(payload, f, indent=2)
    return checkpoint_dir
