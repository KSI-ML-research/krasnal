from __future__ import annotations

from .acpl import ACPLMetric
from .cot_acpl import CoTACPLMetric
from .illegal_mass import IllegalMassMetric
from .top1_legal import Top1LegalMetric

METRIC_REGISTRY: dict[str, type] = {
    "top1_legal": Top1LegalMetric,
    "illegal_mass": IllegalMassMetric,
    "acpl": ACPLMetric,
    "cot_acpl": CoTACPLMetric,
}

__all__ = [
    "Top1LegalMetric",
    "IllegalMassMetric",
    "ACPLMetric",
    "CoTACPLMetric",
    "METRIC_REGISTRY",
]
