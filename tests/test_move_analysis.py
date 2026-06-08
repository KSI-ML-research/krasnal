import pytest
import torch

from krasnal.inference.move_analysis import (
    analyze_move,
    delay_to_seconds,
    move_entropy,
    ply_scaling,
)


def test_compute_move_entropy_on_uniform_distribution():
    legal_probs = torch.tensor([0.25, 0.25, 0.25, 0.25])
    entropy = move_entropy(legal_probs)
    assert entropy == pytest.approx(1.386294, rel=1e-5)


def test_compute_move_entropy_on_peaked_distribution():
    legal_probs = torch.tensor([0.0, 0.0, 1.0, 0.0])
    entropy = move_entropy(legal_probs)
    assert entropy == pytest.approx(0.0, abs=1e-6)


def test_compute_move_entropy_on_mixed_distribution():
    legal_probs = torch.tensor([0.5, 0.25, 0.25, 0.0])
    entropy = move_entropy(legal_probs)
    assert entropy > 0.0
    assert entropy < 1.386294


def test_delay_to_seconds_starts_at_base_delay():
    assert delay_to_seconds(0.0) == pytest.approx(0.5)


def test_delay_to_seconds_grows_with_delay():
    low = delay_to_seconds(0.5)
    mid = delay_to_seconds(1.5)
    high = delay_to_seconds(3.0)

    assert low < mid < high
    assert high <= 20.0


def test_delay_to_seconds_rejects_invalid_arguments():
    with pytest.raises(ValueError):
        delay_to_seconds(1.0, base_delay=-0.1)
    with pytest.raises(ValueError):
        delay_to_seconds(1.0, base_delay=2.0, max_delay=1.0)
    with pytest.raises(ValueError):
        delay_to_seconds(1.0, scale_factor=0.0)


def test_analyze_move_returns_entropy_and_delay():
    probs = torch.tensor([0.2, 0.3, 0.5])
    ply = 5
    result = analyze_move(probs, ply)

    assert result.move_dist_entropy == pytest.approx(move_entropy(probs))
    expected_seconds = delay_to_seconds(result.move_dist_entropy * ply_scaling(ply))
    assert result.delay_seconds == pytest.approx(expected_seconds)
