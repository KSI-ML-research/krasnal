from collections.abc import Callable
from typing import Any, ClassVar

from .base import Metric
from .context import EvalContext
from .core import CoreMetric


class FilteredMetric(Metric):
    """Wraps a core metric with a filter function."""

    def __init__(
        self,
        core: CoreMetric,
        filter_fn: Callable[[EvalContext], bool],
        result_key: str | None = None,
    ):
        self.core = core
        self.filter_fn = filter_fn
        self.result_key = result_key or f"{core.name}_filtered"
        self.buffer: list[float] = []

    def compute(self, ctx: EvalContext) -> dict[str, Any]:
        if not self.filter_fn(ctx):
            return {}
        value = self.core.compute_value(ctx)
        if value is not None:
            self.buffer.append(value)
        return {}

    def finalize(self) -> dict[str, Any]:
        if not self.buffer:
            return {self.result_key: 0.0}
        return {self.result_key: sum(self.buffer) / len(self.buffer)}


class WhenInCheckMetric(FilteredMetric):
    """Filter positions where king is in check."""

    def __init__(self, core: CoreMetric):
        super().__init__(
            core=core,
            filter_fn=lambda ctx: ctx.in_check is True,
            result_key=f"{core.name}_when_in_check",
        )


class WhenGivesCheckMetric(FilteredMetric):
    """Filter positions where move gives check."""

    def __init__(self, core: CoreMetric):
        super().__init__(
            core=core,
            filter_fn=lambda ctx: ctx.gives_check is True,
            result_key=f"{core.name}_when_gives_check",
        )


PHASES = ["opening", "middlegame", "endgame"]


class ByPhaseMetric(FilteredMetric):
    """Filter by game phase (opening/middlegame/endgame)."""

    def __init__(self, core: CoreMetric, phase: str):
        if phase not in PHASES:
            raise ValueError(f"Invalid phase: {phase}. Must be one of {PHASES}")
        super().__init__(
            core=core,
            filter_fn=lambda ctx: ctx.phase == phase,
            result_key=f"{core.name}_{phase}",
        )


class PerPieceMetric(Metric):
    """Group by piece type and aggregate."""

    PIECE_NAMES: ClassVar[dict[int, str]] = {
        1: "pawn",
        2: "knight",
        3: "bishop",
        4: "rook",
        5: "queen",
        6: "king",
    }

    def __init__(self, core: CoreMetric):
        self.core = core
        self.buffers: dict[int, list[float]] = {p: [] for p in self.PIECE_NAMES}

    @property
    def name(self) -> str:
        return f"target_piece_{self.core.name}"

    def compute(self, ctx: EvalContext) -> dict[str, Any]:
        if ctx.piece_type not in self.buffers:
            return {}
        value = self.core.compute_value(ctx)
        if value is not None:
            self.buffers[ctx.piece_type].append(value)
        return {}

    def finalize(self) -> dict[str, Any]:
        result = {}
        for ptype, values in self.buffers.items():
            piece_name = self.PIECE_NAMES.get(ptype, f"piece_{ptype}")
            key = f"target_{piece_name}_{self.core.name}"
            result[key] = sum(values) / len(values) if values else 0.0
        return result
