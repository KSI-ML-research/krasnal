from typing import Any

from .acpl import ACPLMetric
from .base import Metric
from .blunder_rate import BlunderRateMetric
from .context import EvalContext
from .core import AccuracyCore, IllegalMassCore, MRRCore, Top1LegalCore
from .cot import (
    CotFormatValidMetric,
    CotPostThinkMRRMetric,
    CotPostThinkTop1LegalMetric,
    CotPostThinkTop1Metric,
    CotThinkTokenRecallMetric,
)
from .filtered import (
    ByPhaseMetric,
    PerPieceMetric,
    WhenGivesCheckMetric,
    WhenInCheckMetric,
)
from .stockfish_top1 import StockfishTop1AgreementMetric


def _create_top1_legal(**_: Any) -> Metric:
    return Top1LegalCore()


def _create_accuracy(**_: Any) -> Metric:
    return AccuracyCore()


def _create_illegal_mass(**_: Any) -> Metric:
    return IllegalMassCore()


def _create_mrr(**_: Any) -> Metric:
    return MRRCore()


def _create_top1_legal_when_in_check(**_: Any) -> Metric:
    return WhenInCheckMetric(Top1LegalCore())


def _create_accuracy_when_in_check(**_: Any) -> Metric:
    return WhenInCheckMetric(AccuracyCore())


def _create_illegal_mass_when_in_check(**_: Any) -> Metric:
    return WhenInCheckMetric(IllegalMassCore())


def _create_top1_legal_when_gives_check(**_: Any) -> Metric:
    return WhenGivesCheckMetric(Top1LegalCore())


def _create_accuracy_when_gives_check(**_: Any) -> Metric:
    return WhenGivesCheckMetric(AccuracyCore())


def _create_illegal_mass_when_gives_check(**_: Any) -> Metric:
    return WhenGivesCheckMetric(IllegalMassCore())


def _create_top1_legal_opening(**_: Any) -> Metric:
    return ByPhaseMetric(Top1LegalCore(), "opening")


def _create_accuracy_opening(**_: Any) -> Metric:
    return ByPhaseMetric(AccuracyCore(), "opening")


def _create_illegal_mass_opening(**_: Any) -> Metric:
    return ByPhaseMetric(IllegalMassCore(), "opening")


def _create_top1_legal_middlegame(**_: Any) -> Metric:
    return ByPhaseMetric(Top1LegalCore(), "middlegame")


def _create_accuracy_middlegame(**_: Any) -> Metric:
    return ByPhaseMetric(AccuracyCore(), "middlegame")


def _create_illegal_mass_middlegame(**_: Any) -> Metric:
    return ByPhaseMetric(IllegalMassCore(), "middlegame")


def _create_top1_legal_endgame(**_: Any) -> Metric:
    return ByPhaseMetric(Top1LegalCore(), "endgame")


def _create_accuracy_endgame(**_: Any) -> Metric:
    return ByPhaseMetric(AccuracyCore(), "endgame")


def _create_illegal_mass_endgame(**_: Any) -> Metric:
    return ByPhaseMetric(IllegalMassCore(), "endgame")


def _create_target_piece_top1_legal(**_: Any) -> Metric:
    return PerPieceMetric(Top1LegalCore())


def _create_target_piece_accuracy(**_: Any) -> Metric:
    return PerPieceMetric(AccuracyCore())


COT_METRICS = [
    "cot_format_valid",
    "cot_post_think_top1",
    "cot_post_think_mrr",
    "cot_post_think_top1_legal",
    "cot_think_token_recall",
]

EXCLUDED_FROM_DEFAULT = [*COT_METRICS, "target_piece_illegal_mass"]

METRIC_REGISTRY: dict[str, Any] = {
    # Core metrics
    "top1_legal": _create_top1_legal,
    "acc": _create_accuracy,
    "illegal_mass": _create_illegal_mass,
    "mrr": _create_mrr,
    # Condition variants
    "top1_legal_when_in_check": _create_top1_legal_when_in_check,
    "acc_when_in_check": _create_accuracy_when_in_check,
    "illegal_mass_when_in_check": _create_illegal_mass_when_in_check,
    "top1_legal_when_gives_check": _create_top1_legal_when_gives_check,
    "acc_when_gives_check": _create_accuracy_when_gives_check,
    "illegal_mass_when_gives_check": _create_illegal_mass_when_gives_check,
    # Phase variants
    "top1_legal_opening": _create_top1_legal_opening,
    "acc_opening": _create_accuracy_opening,
    "illegal_mass_opening": _create_illegal_mass_opening,
    "top1_legal_middlegame": _create_top1_legal_middlegame,
    "acc_middlegame": _create_accuracy_middlegame,
    "illegal_mass_middlegame": _create_illegal_mass_middlegame,
    "top1_legal_endgame": _create_top1_legal_endgame,
    "acc_endgame": _create_accuracy_endgame,
    "illegal_mass_endgame": _create_illegal_mass_endgame,
    # Per-piece variants
    "target_piece_top1_legal": _create_target_piece_top1_legal,
    "target_piece_acc": _create_target_piece_accuracy,
    # Stockfish-based metrics (keep as-is - need special args)
    "acpl": ACPLMetric,
    "blunder_rate": BlunderRateMetric,
    "stockfish_top1": StockfishTop1AgreementMetric,
    # CoT metrics
    "cot_format_valid": CotFormatValidMetric,
    "cot_post_think_top1": CotPostThinkTop1Metric,
    "cot_post_think_mrr": CotPostThinkMRRMetric,
    "cot_post_think_top1_legal": CotPostThinkTop1LegalMetric,
    "cot_think_token_recall": CotThinkTokenRecallMetric,
}

DEFAULT_METRICS = [k for k in METRIC_REGISTRY if k not in EXCLUDED_FROM_DEFAULT]

__all__ = [
    "COT_METRICS",
    "DEFAULT_METRICS",
    "METRIC_REGISTRY",
    "EvalContext",
]
