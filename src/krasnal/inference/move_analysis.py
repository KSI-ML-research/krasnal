from dataclasses import dataclass
import math

import torch
import torch.distributions


DEFAULT_BASE_DELAY = 0.5
DEFAULT_MAX_DELAY = 20.0
DEFAULT_SCALE_FACTOR = 2.0


@dataclass(frozen=True, slots=True)
class MoveAnalysisResult:
    move_dist_entropy: float    # entropy on move distribution
    ply_factor: float   # ply with a function applied  
    delay: float # delay based on previous factors        
    delay_seconds: float    # delay converted to seconds


def move_entropy (legal_probs: torch.Tensor) -> float:
    return float(torch.distributions.Categorical(legal_probs).entropy().item())

def ply_scaling(ply: int) -> float:
    if ply < 0:
        ply = 0
    coef = -0.0008
    c = 1.5
    return max(0.7, (coef * (float(ply) - 35) ** 2 + c))

def delay(
    move_dist_entropy: float,
    ply_factor: float,
) -> float:
    return move_dist_entropy * ply_factor

def delay_to_seconds(
    delay: float,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    scale_factor: float = DEFAULT_SCALE_FACTOR,
) -> float:
    if base_delay < 0:
        raise ValueError("base_delay must be non-negative")
    if max_delay < base_delay:
        raise ValueError("max_delay must be greater than or equal to base_delay")
    if scale_factor <= 0:
        raise ValueError("scale_factor must be positive")

    delay = base_delay + (max_delay - base_delay) * (1.0 - math.exp(-delay / scale_factor)) #applying function to convert delay to seconds
    return min(delay, max_delay)


def analyze_move(
    legal_probs: torch.Tensor,
    ply: int,
) -> MoveAnalysisResult:
    move_dist_entropy = move_entropy(legal_probs)
    ply_factor = ply_scaling(ply)
    d = delay(move_dist_entropy, ply_factor)
    d_seconds = delay_to_seconds(d)
    return MoveAnalysisResult(move_dist_entropy, ply_factor, d, d_seconds)