from abc import ABC, abstractmethod
from typing import Any

from .context import EvalContext


class Metric(ABC):
    """Base class for all evaluation metrics."""

    @abstractmethod
    def compute(self, ctx: EvalContext) -> dict[str, Any]:
        """Compute metric for a single position."""
        pass
