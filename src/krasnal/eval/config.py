from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvalConfig:
    eval_dataset_path: Path
    num_games: int = 100
    metrics: list[str] | None = None
    eval_interval: int = 5000
    seed: int | None = None
