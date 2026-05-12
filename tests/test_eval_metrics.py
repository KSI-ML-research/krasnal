import pytest
import torch

from krasnal.eval.metrics import EvalContext
from krasnal.eval.metrics.core import (
    AccuracyCore,
    IllegalMassCore,
    MRRCore,
    Top1LegalCore,
)
from krasnal.eval.metrics.filtered import (
    ByEloMetric,
    ByPhaseMetric,
    PerPieceMetric,
    WhenGivesCheckMetric,
    WhenInCheckMetric,
)
from krasnal.tokens import ELO_1500_1599_ID, ELO_2000_2099_ID


def test_top1_legal_returns_1_when_top_is_legal():
    ctx = EvalContext(
        probs=torch.tensor([0.1, 0.8, 0.1]),
        legal_ids=[1, 2],
    )
    result = Top1LegalCore().compute(ctx)
    assert result["top1_legal"] == 1.0


def test_top1_legal_returns_0_when_top_is_illegal():
    probs = torch.tensor([0.05, 0.05, 0.9])
    ctx = EvalContext(probs=probs, legal_ids=[0, 1])
    result = Top1LegalCore().compute(ctx)
    assert result["top1_legal"] == 0.0


def test_top1_legal_returns_empty_when_probs_missing():
    ctx = EvalContext(probs=None, legal_ids=[0, 1])
    result = Top1LegalCore().compute(ctx)
    assert result == {}


def test_accuracy_returns_1_when_top_matches_actual():
    ctx = EvalContext(
        probs=torch.tensor([0.1, 0.8, 0.1]),
        actual_token=1,
    )
    result = AccuracyCore().compute(ctx)
    assert result["acc"] == 1.0


def test_accuracy_returns_0_when_top_doesnt_match_actual():
    ctx = EvalContext(
        probs=torch.tensor([0.8, 0.1, 0.1]),
        actual_token=2,
    )
    result = AccuracyCore().compute(ctx)
    assert result["acc"] == 0.0


def test_accuracy_returns_empty_when_actual_missing():
    ctx = EvalContext(probs=torch.zeros(10), actual_token=None)
    result = AccuracyCore().compute(ctx)
    assert result == {}


def test_illegal_mass_sums_probability_on_illegal_tokens():
    probs = torch.tensor([0.1, 0.2, 0.3, 0.4])
    ctx = EvalContext(probs=probs, legal_ids=[0, 2])
    result = IllegalMassCore().compute(ctx)
    assert result["illegal_mass"] == pytest.approx(0.6)


def test_illegal_mass_returns_empty_when_missing():
    ctx = EvalContext(probs=None, legal_ids=[0, 1])
    result = IllegalMassCore().compute(ctx)
    assert result == {}


def test_mrr_returns_1_when_actual_is_top():
    probs = torch.zeros(5)
    probs[2] = 1.0
    ctx = EvalContext(probs=probs, actual_token=2)
    result = MRRCore().compute(ctx)
    assert result["mrr"] == 1.0


def test_mrr_returns_fraction_for_lower_ranks():
    probs = torch.tensor([0.4, 0.3, 0.2, 0.1])
    ctx = EvalContext(probs=probs, actual_token=3)
    result = MRRCore().compute(ctx)
    assert result["mrr"] == 1.0 / 4


def test_mrr_returns_empty_when_actual_missing():
    ctx = EvalContext(probs=torch.zeros(10), actual_token=None)
    result = MRRCore().compute(ctx)
    assert result == {}


def test_when_in_check_buffers_only_check_positions():
    metric = WhenInCheckMetric(Top1LegalCore())
    metric.compute(EvalContext(probs=torch.zeros(5), in_check=True, legal_ids=[0]))
    metric.compute(EvalContext(probs=torch.zeros(5), in_check=False, legal_ids=[0]))
    metric.compute(EvalContext(probs=torch.zeros(5), in_check=None, legal_ids=[0]))
    assert len(metric.buffer) == 1


def test_when_in_check_finalizes_average():
    metric = WhenInCheckMetric(Top1LegalCore())
    metric.compute(EvalContext(probs=torch.tensor([0.9, 0.1]), in_check=True, legal_ids=[0]))
    metric.compute(EvalContext(probs=torch.tensor([0.9, 0.1]), in_check=True, legal_ids=[1]))
    result = metric.finalize()
    assert result["top1_legal_when_in_check"] == 0.5


def test_when_gives_check_buffers_only_check_giving_positions():
    metric = WhenGivesCheckMetric(Top1LegalCore())
    metric.compute(EvalContext(probs=torch.zeros(5), gives_check=True, legal_ids=[0]))
    metric.compute(EvalContext(probs=torch.zeros(5), gives_check=False, legal_ids=[0]))
    assert len(metric.buffer) == 1


def test_by_phase_buffers_only_matching_phase():
    metric = ByPhaseMetric(Top1LegalCore(), "opening")
    metric.compute(EvalContext(phase="opening", probs=torch.zeros(5), legal_ids=[0]))
    metric.compute(EvalContext(phase="middlegame", probs=torch.zeros(5), legal_ids=[0]))
    assert len(metric.buffer) == 1


def test_by_phase_finalizes_by_phase():
    metric = ByPhaseMetric(AccuracyCore(), "opening")
    metric.compute(EvalContext(phase="opening", probs=torch.tensor([0.9, 0.1]), actual_token=0))
    metric.compute(EvalContext(phase="opening", probs=torch.tensor([0.9, 0.1]), actual_token=1))
    result = metric.finalize()
    assert result["acc_opening"] == 0.5


def test_by_phase_ignores_non_matching_phases():
    metric = ByPhaseMetric(Top1LegalCore(), "middlegame")
    metric.compute(EvalContext(phase="opening", probs=torch.zeros(5), legal_ids=[0]))
    metric.compute(EvalContext(phase="endgame", probs=torch.zeros(5), legal_ids=[0]))
    assert len(metric.buffer) == 0


def test_by_elo_finalizes_by_player_elo_bucket():
    metric = ByEloMetric(AccuracyCore(), ELO_1500_1599_ID)
    metric.compute(
        EvalContext(
            player_elo_token=ELO_1500_1599_ID,
            probs=torch.tensor([0.9, 0.1]),
            actual_token=0,
        )
    )
    metric.compute(
        EvalContext(
            player_elo_token=ELO_1500_1599_ID,
            probs=torch.tensor([0.9, 0.1]),
            actual_token=1,
        )
    )
    metric.compute(
        EvalContext(
            player_elo_token=ELO_2000_2099_ID,
            probs=torch.tensor([0.9, 0.1]),
            actual_token=0,
        )
    )

    result = metric.finalize()

    assert result["acc/acc_elo_1500_1599"] == 0.5


def test_per_piece_buffers_by_piece_type():
    metric = PerPieceMetric(Top1LegalCore())
    metric.compute(EvalContext(piece_type=1, probs=torch.zeros(5), legal_ids=[0]))
    metric.compute(EvalContext(piece_type=2, probs=torch.zeros(5), legal_ids=[0]))
    metric.compute(EvalContext(piece_type=1, probs=torch.zeros(5), legal_ids=[0]))
    assert len(metric.buffers[1]) == 2
    assert len(metric.buffers[2]) == 1


def test_per_piece_finalizes_by_piece():
    metric = PerPieceMetric(AccuracyCore())
    metric.compute(EvalContext(piece_type=1, probs=torch.tensor([0.9, 0.1]), actual_token=0))
    metric.compute(EvalContext(piece_type=1, probs=torch.tensor([0.9, 0.1]), actual_token=1))
    result = metric.finalize()
    assert result["target_pawn_acc"] == 0.5  # 1/2 correct (first matches, second doesn't)


def test_per_piece_skips_unknown_piece_types():
    metric = PerPieceMetric(Top1LegalCore())
    metric.compute(EvalContext(piece_type=99, probs=torch.zeros(5), legal_ids=[0]))
    assert all(len(buf) == 0 for buf in metric.buffers.values())
