from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import bulletchess
import torch

from krasnal.inference import Game, InferenceSession
from krasnal.tokens import BLACK_WON_ID, WHITE_WON_ID, legal_token_ids, to_uci


@dataclass(frozen=True)
class PuzzleBucket:
    name: str
    min_rating: int
    max_rating_exclusive: int | None = None

    def contains(self, rating: int) -> bool:
        if rating < self.min_rating:
            return False
        if self.max_rating_exclusive is None:
            return True
        return rating < self.max_rating_exclusive


DEFAULT_PUZZLE_BUCKETS: tuple[PuzzleBucket, ...] = (
    PuzzleBucket(name="1000_1200", min_rating=1000, max_rating_exclusive=1200),
    PuzzleBucket(name="1200_1400", min_rating=1200, max_rating_exclusive=1400),
    PuzzleBucket(name="1400_1600", min_rating=1400, max_rating_exclusive=1600),
    PuzzleBucket(name="1600_1800", min_rating=1600, max_rating_exclusive=1800),
    PuzzleBucket(name="1800_plus", min_rating=1800, max_rating_exclusive=None),
)


@dataclass
class _BucketStats:
    total: int = 0
    evaluated: int = 0
    skipped: int = 0
    exact_matches: int = 0
    predicted_legal: int = 0
    solution_legal: int = 0
    solved_with_rating: list[tuple[int, int]] = field(default_factory=list)

    def record_skip(self) -> None:
        self.total += 1
        self.skipped += 1

    def record_eval(
        self,
        *,
        exact_match: bool,
        predicted_is_legal: bool,
        solution_is_legal: bool,
        rating: int | None,
    ) -> None:
        self.total += 1
        self.evaluated += 1
        self.exact_matches += int(exact_match)
        self.predicted_legal += int(predicted_is_legal)
        self.solution_legal += int(solution_is_legal)
        if rating is not None:
            self.solved_with_rating.append((rating, int(exact_match)))

    def to_metrics(self, prefix: str) -> dict[str, float | int]:
        exact_match = self.exact_matches / self.evaluated if self.evaluated else 0.0
        predicted_legal = self.predicted_legal / self.evaluated if self.evaluated else 0.0
        solution_legal = self.solution_legal / self.evaluated if self.evaluated else 0.0
        pseudo_elo = estimate_pseudo_elo(self.solved_with_rating)
        return {
            f"{prefix}/total": self.total,
            f"{prefix}/evaluated": self.evaluated,
            f"{prefix}/skipped": self.skipped,
            f"{prefix}/exact_match": exact_match,
            f"{prefix}/predicted_legal": predicted_legal,
            f"{prefix}/solution_legal": solution_legal,
            f"{prefix}/pseudo_elo": pseudo_elo,
        }


def load_puzzles_jsonl(puzzle_path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    puzzles: list[dict[str, Any]] = []
    with puzzle_path.open() as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            puzzles.append(json.loads(line))
    if not puzzles:
        raise ValueError(f"No puzzles found in {puzzle_path}")
    return puzzles


def sample_puzzles(
    puzzles: list[dict[str, Any]],
    sample_size: int | None,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    if sample_size is None or sample_size <= 0 or sample_size >= len(puzzles):
        return puzzles
    rng = random.Random(seed)
    return rng.sample(puzzles, sample_size)


def estimate_pseudo_elo(
    solved_outcomes: list[tuple[int, int]],
    *,
    min_rating: float = 200.0,
    max_rating: float = 3200.0,
    iters: int = 48,
) -> float:
    if not solved_outcomes:
        return 0.0

    solved_total = sum(solved for _, solved in solved_outcomes)
    if solved_total <= 0:
        return min_rating
    if solved_total >= len(solved_outcomes):
        return max_rating

    lo = min_rating
    hi = max_rating
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        expected = 0.0
        for puzzle_rating, _ in solved_outcomes:
            expected += _solve_probability(mid, float(puzzle_rating))
        if expected > solved_total:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


def evaluate_model_on_puzzles(
    *,
    model: torch.nn.Module,
    device: torch.device,
    puzzles: list[dict[str, Any]],
    buckets: tuple[PuzzleBucket, ...] = DEFAULT_PUZZLE_BUCKETS,
) -> dict[str, float | int]:
    bucket_stats = {bucket.name: _BucketStats() for bucket in buckets}
    overall_stats = _BucketStats()

    for puzzle in puzzles:
        fen = puzzle.get("fen")
        solution = puzzle.get("solution")
        rating = _parse_rating(puzzle.get("rating"))
        bucket_name = _bucket_name_for_rating(rating, buckets)
        bucket = bucket_stats.get(bucket_name) if bucket_name else None

        if not fen or not solution:
            overall_stats.record_skip()
            if bucket is not None:
                bucket.record_skip()
            continue

        try:
            board = bulletchess.Board.from_fen(fen)
        except Exception:
            overall_stats.record_skip()
            if bucket is not None:
                bucket.record_skip()
            continue

        legal_moves = {move.uci() for move in board.legal_moves()}
        if not legal_moves:
            overall_stats.record_skip()
            if bucket is not None:
                bucket.record_skip()
            continue

        predicted_move = _predict_top1_move(model=model, device=device, board=board, fen=fen)
        if predicted_move is None:
            overall_stats.record_skip()
            if bucket is not None:
                bucket.record_skip()
            continue

        exact_match = predicted_move == solution
        predicted_is_legal = predicted_move in legal_moves
        solution_is_legal = solution in legal_moves

        overall_stats.record_eval(
            exact_match=exact_match,
            predicted_is_legal=predicted_is_legal,
            solution_is_legal=solution_is_legal,
            rating=rating,
        )
        if bucket is not None:
            bucket.record_eval(
                exact_match=exact_match,
                predicted_is_legal=predicted_is_legal,
                solution_is_legal=solution_is_legal,
                rating=rating,
            )

    metrics: dict[str, float | int] = {}
    metrics.update(overall_stats.to_metrics("overall"))
    for bucket_name, stats in bucket_stats.items():
        metrics.update(stats.to_metrics(f"bucket/{bucket_name}"))
    return metrics


def evaluate_model_on_puzzle_file(
    *,
    model: torch.nn.Module,
    device: torch.device,
    puzzle_path: Path,
    sample_size: int | None = None,
    seed: int = 42,
    buckets: tuple[PuzzleBucket, ...] = DEFAULT_PUZZLE_BUCKETS,
) -> dict[str, float | int]:
    puzzles = load_puzzles_jsonl(puzzle_path)
    sampled_puzzles = sample_puzzles(puzzles, sample_size, seed=seed)
    metrics = evaluate_model_on_puzzles(
        model=model,
        device=device,
        puzzles=sampled_puzzles,
        buckets=buckets,
    )
    metrics["overall/source_total"] = len(puzzles)
    metrics["overall/sample_total"] = len(sampled_puzzles)
    return metrics


def _parse_rating(raw_rating: Any) -> int | None:
    if raw_rating is None:
        return None
    try:
        return int(raw_rating)
    except (TypeError, ValueError):
        return None


def _bucket_name_for_rating(
    rating: int | None,
    buckets: tuple[PuzzleBucket, ...],
) -> str | None:
    if rating is None:
        return None
    for bucket in buckets:
        if bucket.contains(rating):
            return bucket.name
    return None


def _solve_probability(model_rating: float, puzzle_rating: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((puzzle_rating - model_rating) / 400.0))


def _predict_top1_move(
    *,
    model: torch.nn.Module,
    device: torch.device,
    board: bulletchess.Board,
    fen: str,
) -> str | None:
    side_to_move = fen.split()[1]
    outcome_token = WHITE_WON_ID if side_to_move == "w" else BLACK_WON_ID
    session = InferenceSession(
        model,
        device,
        game=Game(target_outcome_token=outcome_token, board=board),
    )
    legal_ids = legal_token_ids(session.game.board)
    if not legal_ids:
        return None
    legal_probs = session.get_legal_probs()
    predicted_token = int(torch.argmax(legal_probs).item())
    return to_uci(predicted_token)
