from pathlib import Path

import pytest
import torch

from src.rl.reward import score_phase1_rollouts
from src.tokenizer import SOS_ID, THINK_END_ID, THINK_START_ID, Tokenizer


@pytest.fixture
def tokenizer():
    return Tokenizer(Path("data/all_uci_moves.txt"))


def test_scores_fully_legal_thinking_and_played_move(tokenizer):
    row = torch.tensor(
        [
            SOS_ID,
            THINK_START_ID,
            tokenizer.move_to_id["e2e4"],
            tokenizer.move_to_id["e7e5"],
            THINK_END_ID,
            tokenizer.move_to_id["g1f3"],
        ]
    ).unsqueeze(0)

    rewards, metrics = score_phase1_rollouts(
        row,
        torch.tensor([1]),
        torch.tensor([2]),
        tokenizer,
    )

    assert rewards.tolist() == pytest.approx([1.5])
    assert metrics["legal_thinking_ratio"] == pytest.approx(1.0)
    assert metrics["played_move_legal_rate"] == pytest.approx(1.0)


def test_stops_thinking_score_on_first_illegal_move(tokenizer):
    row = torch.tensor(
        [
            SOS_ID,
            THINK_START_ID,
            tokenizer.move_to_id["e2e4"],
            tokenizer.move_to_id["d2d4"],
            THINK_END_ID,
            tokenizer.move_to_id["e2e4"],
        ]
    ).unsqueeze(0)

    rewards, metrics = score_phase1_rollouts(
        row,
        torch.tensor([1]),
        torch.tensor([2]),
        tokenizer,
    )

    assert rewards.tolist() == pytest.approx([1.0])
    assert metrics["legal_thinking_ratio"] == pytest.approx(0.5)
    assert metrics["played_move_legal_rate"] == pytest.approx(1.0)


def test_played_move_is_checked_on_original_board(tokenizer):
    row = torch.tensor(
        [
            SOS_ID,
            THINK_START_ID,
            tokenizer.move_to_id["e2e4"],
            tokenizer.move_to_id["e7e5"],
            THINK_END_ID,
            tokenizer.move_to_id["e7e5"],
        ]
    ).unsqueeze(0)

    rewards, metrics = score_phase1_rollouts(
        row,
        torch.tensor([1]),
        torch.tensor([2]),
        tokenizer,
    )

    assert rewards.tolist() == pytest.approx([1.0])
    assert metrics["played_move_legal_rate"] == pytest.approx(0.0)
