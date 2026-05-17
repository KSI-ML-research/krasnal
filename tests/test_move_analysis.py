import pytest
import torch

from krasnal.inference.move_analysis import analyze_move, compute_move_entropy, entropy_to_delay


def test_compute_move_entropy_on_uniform_distribution():
    legal_probs = torch.tensor([0.25, 0.25, 0.25, 0.25])
    entropy = compute_move_entropy(legal_probs)
    assert entropy == pytest.approx(1.386294, rel=1e-5)


def test_compute_move_entropy_on_peaked_distribution():
    legal_probs = torch.tensor([0.0, 0.0, 1.0, 0.0])
    entropy = compute_move_entropy(legal_probs)
    assert entropy == pytest.approx(0.0, abs=1e-6)


def test_compute_move_entropy_on_mixed_distribution():
    legal_probs = torch.tensor([0.5, 0.25, 0.25, 0.0])
    entropy = compute_move_entropy(legal_probs)
    assert entropy > 0.0
    assert entropy < 1.386294


def test_entropy_to_delay_starts_at_base_delay():
    assert entropy_to_delay(0.0) == pytest.approx(0.5)


def test_entropy_to_delay_grows_with_entropy():
    low = entropy_to_delay(0.5)
    mid = entropy_to_delay(1.5)
    high = entropy_to_delay(3.0)

    assert low < mid < high
    assert high <= 20.0


def test_entropy_to_delay_rejects_invalid_arguments():
    with pytest.raises(ValueError):
        entropy_to_delay(1.0, base_delay=-0.1)
    with pytest.raises(ValueError):
        entropy_to_delay(1.0, base_delay=2.0, max_delay=1.0)
    with pytest.raises(ValueError):
        entropy_to_delay(1.0, scale_factor=0.0)


def test_analyze_move_returns_entropy_and_delay():
    probs = torch.tensor([0.2, 0.3, 0.5])
    result = analyze_move(probs)

    assert result.entropy == pytest.approx(compute_move_entropy(probs))
    assert result.delay_seconds == pytest.approx(entropy_to_delay(result.entropy))