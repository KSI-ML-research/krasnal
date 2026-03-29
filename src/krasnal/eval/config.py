from dataclasses import dataclass, field
from pathlib import Path

from .metrics import DEFAULT_METRICS


@dataclass
class EvalConfig:
    eval_dataset_path: Path
    num_games: int = 100
    metrics: list[str] = field(default_factory=lambda: DEFAULT_METRICS)
    eval_interval: int = 5000
    seed: int | None = None
    stockfish_depth: int = 15
    acpl_sample_size: int = 10
