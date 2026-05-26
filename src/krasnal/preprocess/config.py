"""Configuration dataclass for preprocessing"""

from __future__ import annotations

import dataclasses
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class PreprocessConfig:
    """Immutable config bundle passed through tokenization and shard workers."""

    seed: int
    block_size: int
    piece_aware_moves: bool = False
    side_prefixed_moves: bool = True
    include_check_qa: bool = True
    check_qa_prob: float = 0.5
    include_what_is_on_qa: bool = False
    what_is_on_prob: float = 0.0
    time_control_enabled: bool = True
    move_vocab_path: Path | None = None
