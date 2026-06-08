"""Quick calibration script for 'analyze_move'.

Generates several example legal-prob distributions and evaluates
'analyze_move' across a range of 'ply' values to show the
heuristic baseline behavior.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import torch

from krasnal.inference.move_analysis import analyze_move


def make_uniform(n: int) -> torch.Tensor:
    p = torch.ones(n, dtype=torch.float32)
    return p / p.sum()


def make_peaked(n: int, peak_idx: int = 0, peak: float = 0.9) -> torch.Tensor:
    rest = (1.0 - peak) / (n - 1)
    p = torch.full((n,), rest, dtype=torch.float32)
    p[peak_idx] = peak
    return p


def make_two_peak(n: int, a: float = 0.5, b: float = 0.3) -> torch.Tensor:
    rest = max(0.0, 1.0 - a - b) / (n - 2)
    p = torch.full((n,), rest, dtype=torch.float32)
    p[0] = a
    p[1] = b
    return p


def make_random_temp(n: int, temp: float = 1.0, seed: int | None = None) -> torch.Tensor:
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)
    logits = torch.randn(n, generator=rng)
    logits = logits / float(max(1e-6, temp))
    p = torch.softmax(logits, dim=-1)
    return p


def eval_scenarios(n: int, plies: List[int]) -> None:
    scenarios = [
        ("uniform", make_uniform(n)),
        ("peaked", make_peaked(n, peak_idx=0, peak=0.9)),
        ("two_peak", make_two_peak(n, a=0.5, b=0.3)),
        ("rand_cold", make_random_temp(n, temp=0.2, seed=0)),
        ("rand_hot", make_random_temp(n, temp=2.0, seed=1)),
    ]

    print(f"{'scenario':12} {'ply':>4} {'entropy':>10} {'ply_factor':>12} {'delay':>10} {'delay_s':>10}")
    print("-" * 64)
    rows = []
    for name, probs in scenarios:
        for ply in plies:
            res = analyze_move(probs, ply)
            print(f"{name:12} {ply:4d} {res.move_dist_entropy:10.4f} {res.ply_factor:12.4f} {res.delay:10.4f} {res.delay_seconds:10.4f}")
            rows.append((name, ply, res.move_dist_entropy, res.ply_factor, res.delay, res.delay_seconds))

    by_scenario = defaultdict(list)
    for name, ply, entropy, ply_factor, delay_value, delay_seconds in rows:
        by_scenario[name].append((ply, entropy, ply_factor, delay_value, delay_seconds))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    for name, points in by_scenario.items():
        points.sort(key=lambda item: item[0])
        plies_sorted = [item[0] for item in points]
        entropy = [item[1] for item in points]
        delay_seconds = [item[4] for item in points]
        ax.plot(entropy, plies_sorted, delay_seconds, marker="o", label=name)
        ax.scatter(entropy, plies_sorted, delay_seconds, s=30)

    ax.set_title("entropy vs ply vs delay_seconds")
    ax.set_xlabel("entropy")
    ax.set_ylabel("ply")
    ax.set_zlabel("delay_seconds")
    ax.legend()

    fig.tight_layout()

    output_dir = Path("artifacts") / "move_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "calibration_3d.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.show()


if __name__ == "__main__":
    # small vocab size (legal moves count)
    vocab_n = 16
    plies = [0, 5, 10, 20, 35, 40, 60]
    eval_scenarios(vocab_n, plies)
