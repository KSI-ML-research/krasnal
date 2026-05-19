from typing import Any

from krasnal.tokens import ELO_BUCKETS

from .context import EvalContext
from .core import AccuracyCore, IllegalMassCore, MRRCore, Top1LegalCore
from .filtered import (
    ByEloMetric,
    ByPhaseMetric,
    PerPieceMetric,
    WhenGivesCheckMetric,
    WhenInCheckMetric,
)

METRIC_REGISTRY: dict[str, Any] = {
    "top1_legal": lambda **_: Top1LegalCore(),
    "acc": lambda **_: AccuracyCore(),
    "illegal_mass": lambda **_: IllegalMassCore(),
    "mrr": lambda **_: MRRCore(),
    "top1_legal_when_in_check": lambda **_: WhenInCheckMetric(Top1LegalCore()),
    "acc_when_in_check": lambda **_: WhenInCheckMetric(AccuracyCore()),
    "illegal_mass_when_in_check": lambda **_: WhenInCheckMetric(IllegalMassCore()),
    "mrr_when_in_check": lambda **_: WhenInCheckMetric(MRRCore()),
    "top1_legal_when_gives_check": lambda **_: WhenGivesCheckMetric(Top1LegalCore()),
    "acc_when_gives_check": lambda **_: WhenGivesCheckMetric(AccuracyCore()),
    "illegal_mass_when_gives_check": lambda **_: WhenGivesCheckMetric(IllegalMassCore()),
    "mrr_when_gives_check": lambda **_: WhenGivesCheckMetric(MRRCore()),
    "top1_legal_opening": lambda **_: ByPhaseMetric(Top1LegalCore(), "opening"),
    "acc_opening": lambda **_: ByPhaseMetric(AccuracyCore(), "opening"),
    "illegal_mass_opening": lambda **_: ByPhaseMetric(IllegalMassCore(), "opening"),
    "mrr_opening": lambda **_: ByPhaseMetric(MRRCore(), "opening"),
    "top1_legal_middlegame": lambda **_: ByPhaseMetric(Top1LegalCore(), "middlegame"),
    "acc_middlegame": lambda **_: ByPhaseMetric(AccuracyCore(), "middlegame"),
    "illegal_mass_middlegame": lambda **_: ByPhaseMetric(IllegalMassCore(), "middlegame"),
    "mrr_middlegame": lambda **_: ByPhaseMetric(MRRCore(), "middlegame"),
    "top1_legal_endgame": lambda **_: ByPhaseMetric(Top1LegalCore(), "endgame"),
    "acc_endgame": lambda **_: ByPhaseMetric(AccuracyCore(), "endgame"),
    "illegal_mass_endgame": lambda **_: ByPhaseMetric(IllegalMassCore(), "endgame"),
    "mrr_endgame": lambda **_: ByPhaseMetric(MRRCore(), "endgame"),
    **{
        f"acc_elo_{bucket_name}": lambda elo_token=elo_token, **_: ByEloMetric(
            AccuracyCore(), elo_token
        )
        for elo_token, bucket_name in ELO_BUCKETS.items()
    },
    "target_piece_top1_legal": lambda **_: PerPieceMetric(Top1LegalCore()),
    "target_piece_acc": lambda **_: PerPieceMetric(AccuracyCore()),
}

__all__ = [
    "METRIC_REGISTRY",
    "EvalContext",
]
