from .acpl import ACPLMetric
from .context import EvalContext
from .gives_check import (
    AccWhenGivesCheckMetric,
    IllegalMassWhenGivesCheckMetric,
    Top1LegalWhenGivesCheckMetric,
)
from .illegal_mass import IllegalMassMetric
from .in_check import AccWhenInCheckMetric, IllegalMassWhenInCheckMetric, Top1LegalWhenInCheckMetric
from .mrr import MRRMetric
from .per_piece import PerPieceAccuracyMetric, PerPieceLegalMetric
from .phases import AccPhaseMetric, IllegalMassPhaseMetric, Top1LegalPhaseMetric
from .top1_legal import Top1LegalMetric

METRIC_REGISTRY = {
    "top1_legal": Top1LegalMetric,
    "mrr": MRRMetric,
    "acpl": ACPLMetric,
    "illegal_mass": IllegalMassMetric,
    "target_piece_legal": PerPieceLegalMetric,
    "target_piece_acc": PerPieceAccuracyMetric,
    "top1_legal_when_in_check": Top1LegalWhenInCheckMetric,
    "acc_when_in_check": AccWhenInCheckMetric,
    "illegal_mass_when_in_check": IllegalMassWhenInCheckMetric,
    "top1_legal_when_gives_check": Top1LegalWhenGivesCheckMetric,
    "acc_when_gives_check": AccWhenGivesCheckMetric,
    "illegal_mass_when_gives_check": IllegalMassWhenGivesCheckMetric,
    "top1_legal_opening": Top1LegalPhaseMetric,
    "top1_legal_middlegame": Top1LegalPhaseMetric,
    "top1_legal_endgame": Top1LegalPhaseMetric,
    "acc_opening": AccPhaseMetric,
    "acc_middlegame": AccPhaseMetric,
    "acc_endgame": AccPhaseMetric,
    "illegal_mass_opening": IllegalMassPhaseMetric,
    "illegal_mass_middlegame": IllegalMassPhaseMetric,
    "illegal_mass_endgame": IllegalMassPhaseMetric,
}

DEFAULT_METRICS = list(METRIC_REGISTRY.keys())

__all__ = [
    "EvalContext",
    "Top1LegalMetric",
    "MRRMetric",
    "ACPLMetric",
    "IllegalMassMetric",
    "PerPieceLegalMetric",
    "PerPieceAccuracyMetric",
    "Top1LegalWhenInCheckMetric",
    "AccWhenInCheckMetric",
    "IllegalMassWhenInCheckMetric",
    "Top1LegalWhenGivesCheckMetric",
    "AccWhenGivesCheckMetric",
    "IllegalMassWhenGivesCheckMetric",
    "Top1LegalPhaseMetric",
    "AccPhaseMetric",
    "IllegalMassPhaseMetric",
    "METRIC_REGISTRY",
    "DEFAULT_METRICS",
]
