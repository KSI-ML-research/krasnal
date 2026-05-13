from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from functools import lru_cache
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import bulletchess
import chess
import chess.pgn
import torch
from loguru import logger

from krasnal.inference import Game, InferenceSession
from krasnal.puzzle_cache import source_game_cache_path_for
from krasnal.tokens import (
    ELO_UNKNOWN_ID,
    MOVE_TO_ID,
    UNKNOWN_RESULT_ID,
    get_elo_bucket,
    legal_token_ids,
    result_to_token_id,
    to_uci,
)


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
    mrr_sum: float = 0.0
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
        mrr: float,
        rating: int | None,
    ) -> None:
        self.total += 1
        self.evaluated += 1
        self.exact_matches += int(exact_match)
        self.predicted_legal += int(predicted_is_legal)
        self.solution_legal += int(solution_is_legal)
        self.mrr_sum += float(mrr)
        if rating is not None:
            self.solved_with_rating.append((rating, int(exact_match)))

    def to_metrics(self, prefix: str) -> dict[str, float | int]:
        exact_match = self.exact_matches / self.evaluated if self.evaluated else 0.0
        predicted_legal = self.predicted_legal / self.evaluated if self.evaluated else 0.0
        solution_legal = self.solution_legal / self.evaluated if self.evaluated else 0.0
        pseudo_elo = estimate_pseudo_elo(self.solved_with_rating)
        metrics = {
            f"{prefix}/total": self.total,
            f"{prefix}/evaluated": self.evaluated,
            f"{prefix}/skipped": self.skipped,
            f"{prefix}/exact_match": exact_match,
            f"{prefix}/predicted_legal": predicted_legal,
            f"{prefix}/solution_legal": solution_legal,
            f"{prefix}/pseudo_elo": pseudo_elo,
        }
        if self.evaluated:
            metrics[f"{prefix}/mrr"] = self.mrr_sum / self.evaluated
        return metrics


@dataclass
class PuzzleEvalResult:
    overall: dict[str, float | int]
    buckets: dict[str, dict[str, float | int]] = field(default_factory=dict)

    def to_metrics(
        self,
        *,
        log_mrr: bool = False,
        log_bucket_metrics: bool = False,
        log_diagnostics: bool = False,
    ) -> dict[str, float | int]:
        metrics = {
            f"puzzle/{key}": value
            for key, value in _filter_metrics(
                self.overall,
                log_mrr=log_mrr,
                log_diagnostics=log_diagnostics,
            ).items()
        }
        if log_bucket_metrics:
            for _bucket_name, bucket_metrics in self.buckets.items():
                metrics.update(
                    {
                        f"puzzle/{key}": value
                        for key, value in _filter_metrics(
                            bucket_metrics,
                            log_mrr=log_mrr,
                            log_diagnostics=log_diagnostics,
                        ).items()
                    }
                )
        return metrics


@dataclass(frozen=True)
class SourceGameRecord:
    game_id: str
    result: str | None
    white_elo: str | None
    black_elo: str | None
    moves_uci: tuple[str, ...]

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> SourceGameRecord:
        game_id = str(payload.get("game_id") or "")
        if not game_id:
            raise ValueError("Missing game_id in source-game cache record")
        moves = tuple(str(move) for move in payload.get("moves_uci") or ())
        return cls(
            game_id=game_id,
            result=payload.get("result"),
            white_elo=payload.get("white_elo"),
            black_elo=payload.get("black_elo"),
            moves_uci=moves,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "result": self.result,
            "white_elo": self.white_elo,
            "black_elo": self.black_elo,
            "moves_uci": list(self.moves_uci),
        }

    def to_game(self, *, puzzle_fen: str) -> Game:
        target_fen = _normalized_fen(puzzle_fen)
        game = Game(
            target_outcome_token=_outcome_token_from_header(self.result),
            white_elo_token=_elo_token_from_header(self.white_elo),
            black_elo_token=_elo_token_from_header(self.black_elo),
        )
        for move_idx, move in enumerate(self.moves_uci):
            game.feed_uci(move)
            if _normalized_fen(game.board.fen()) == target_fen:
                next_move_idx = move_idx + 1
                if next_move_idx < len(self.moves_uci):
                    game.feed_uci(self.moves_uci[next_move_idx])
                return game
        raise ValueError(f"Puzzle FEN not found in cached source game: {self.game_id}")


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


def load_source_game_cache(cache_path: Path) -> dict[str, SourceGameRecord]:
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return {}

    cache: dict[str, SourceGameRecord] = {}
    try:
        with cache_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = SourceGameRecord.from_json(json.loads(line))
                cache[record.game_id] = record
    except Exception as exc:
        logger.warning("Failed to load puzzle source-game cache from {}: {}", cache_path, exc)
        return {}
    return cache


def build_source_game_cache(puzzles: list[dict[str, Any]], output_path: Path) -> int:
    output_path = Path(output_path)
    unique_game_urls: dict[str, str] = {}
    for puzzle in puzzles:
        game_url = puzzle.get("game_url")
        if not game_url:
            continue
        try:
            game_id = _extract_lichess_game_id(str(game_url))
        except ValueError as exc:
            logger.warning("Skipping puzzle with invalid game_url {}: {}", game_url, exc)
            continue
        unique_game_urls.setdefault(game_id, str(game_url))

    records: dict[str, SourceGameRecord] = {}
    for game_id in sorted(unique_game_urls):
        game_url = unique_game_urls[game_id]
        try:
            records[game_id] = _load_source_game_record(game_url)
        except Exception as exc:
            logger.warning("Skipping source-game cache entry for {}: {}", game_url, exc)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for game_id in sorted(records):
            f.write(json.dumps(records[game_id].to_json(), sort_keys=True))
            f.write("\n")
    return len(records)


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


def _extract_lichess_game_id(game_url: str) -> str:
    parsed = urlparse(game_url)
    path_parts = [part for part in parsed.path.split("/") if part]
    if not path_parts:
        raise ValueError(f"Could not extract game id from URL: {game_url}")
    game_id = path_parts[-1]
    if game_id in {"white", "black", "analysis"} and len(path_parts) >= 2:
        game_id = path_parts[-2]
    if game_id.endswith(".pgn"):
        game_id = game_id[:-4]
    if not game_id:
        raise ValueError(f"Could not extract game id from URL: {game_url}")
    return game_id


@lru_cache(maxsize=4096)
def _fetch_lichess_pgn(game_url: str) -> str:
    game_id = _extract_lichess_game_id(game_url)
    export_url = f"https://lichess.org/game/export/{game_id}.pgn"
    request = Request(export_url, headers={"User-Agent": "krasnal-puzzle-eval/1.0"})
    with urlopen(request, timeout=10) as response:
        return response.read().decode("utf-8")


def _elo_token_from_header(raw_elo: Any) -> int:
    if raw_elo is None:
        return ELO_UNKNOWN_ID
    try:
        return get_elo_bucket(int(raw_elo))
    except (TypeError, ValueError):
        return ELO_UNKNOWN_ID


def _outcome_token_from_header(raw_result: Any) -> int:
    try:
        return result_to_token_id(raw_result if raw_result is not None else UNKNOWN_RESULT_ID)
    except ValueError:
        return UNKNOWN_RESULT_ID


def _build_game_from_source_game(*, game_url: str, puzzle_fen: str) -> Game:
    source_game = _load_source_game_record(game_url)
    return source_game.to_game(puzzle_fen=puzzle_fen)


def _load_source_game_record(game_url: str) -> SourceGameRecord:
    game_id = _extract_lichess_game_id(game_url)
    pgn_text = _fetch_lichess_pgn(game_url)
    pgn_game = chess.pgn.read_game(StringIO(pgn_text))
    if pgn_game is None:
        raise ValueError(f"Could not parse PGN for {game_url}")

    return SourceGameRecord(
        game_id=game_id,
        result=pgn_game.headers.get("Result"),
        white_elo=pgn_game.headers.get("WhiteElo"),
        black_elo=pgn_game.headers.get("BlackElo"),
        moves_uci=tuple(move.uci() for move in pgn_game.mainline_moves()),
    )


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
    source_game_cache: dict[str, SourceGameRecord] | None = None,
) -> PuzzleEvalResult:
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

        source_game = _source_game_for_puzzle(puzzle, source_game_cache)
        game = _game_for_puzzle(board=board, source_game=source_game)
        legal_moves = {move.uci() for move in game.board.legal_moves()}
        if not legal_moves:
            overall_stats.record_skip()
            if bucket is not None:
                bucket.record_skip()
            continue
        predicted_move, mrr = _predict_puzzle_move(
            model=model,
            device=device,
            game=game,
            solution=solution,
        )
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
            mrr=mrr,
            rating=rating,
        )
        if bucket is not None:
            bucket.record_eval(
                exact_match=exact_match,
                predicted_is_legal=predicted_is_legal,
                solution_is_legal=solution_is_legal,
                mrr=mrr,
                rating=rating,
            )

    overall_metrics = overall_stats.to_metrics("overall")
    bucket_metrics = {
        bucket_name: stats.to_metrics(f"bucket/{bucket_name}")
        for bucket_name, stats in bucket_stats.items()
    }
    return PuzzleEvalResult(overall=overall_metrics, buckets=bucket_metrics)


def _filter_metrics(
    metrics: dict[str, float | int],
    *,
    log_mrr: bool,
    log_diagnostics: bool,
) -> dict[str, float | int]:
    filtered: dict[str, float | int] = {}
    for key, value in metrics.items():
        if key.endswith("/exact_match") or key.endswith("/pseudo_elo"):
            filtered[key] = value
            continue
        if log_mrr and key.endswith("/mrr"):
            filtered[key] = value
            continue
        if log_diagnostics and (
            key.endswith("/total")
            or key.endswith("/evaluated")
            or key.endswith("/skipped")
            or key.endswith("/predicted_legal")
            or key.endswith("/solution_legal")
            or key.endswith("/source_total")
            or key.endswith("/sample_total")
        ):
            filtered[key] = value
    return filtered


def evaluate_model_on_puzzle_file(
    *,
    model: torch.nn.Module,
    device: torch.device,
    puzzle_path: Path,
    sample_size: int | None = None,
    seed: int = 42,
    buckets: tuple[PuzzleBucket, ...] = DEFAULT_PUZZLE_BUCKETS,
    source_game_cache_path: Path | None = None,
) -> PuzzleEvalResult:
    puzzles = load_puzzles_jsonl(puzzle_path)
    sampled_puzzles = sample_puzzles(puzzles, sample_size, seed=seed)
    cache_path = (
        Path(source_game_cache_path)
        if source_game_cache_path is not None
        else source_game_cache_path_for(puzzle_path, sample_size=sample_size, seed=seed)
    )
    source_game_cache = load_source_game_cache(cache_path)
    if not source_game_cache:
        logger.warning(
            "Puzzle source-game cache not found or empty at {}. Falling back to FEN-only context. "
            "Run `just prepare-puzzles` to generate the offline cache.",
            cache_path,
        )
    result = evaluate_model_on_puzzles(
        model=model,
        device=device,
        puzzles=sampled_puzzles,
        buckets=buckets,
        source_game_cache=source_game_cache,
    )
    result.overall["overall/source_total"] = len(puzzles)
    result.overall["overall/sample_total"] = len(sampled_puzzles)
    return result


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


def _predict_puzzle_move(
    *,
    model: torch.nn.Module,
    device: torch.device,
    game: Game,
    solution: str,
) -> tuple[str | None, float]:
    session = InferenceSession(model, device, game=game)
    legal_ids = legal_token_ids(session.game.board)
    if not legal_ids:
        return None, 0.0
    legal_probs = session.get_legal_probs()
    legal_ranks = torch.argsort(legal_probs[legal_ids], descending=True)
    predicted_token = int(legal_ids[int(legal_ranks[0].item())])
    predicted_move = to_uci(predicted_token)
    solution_token = _solution_token_id(solution=solution, turn=game.board.turn)
    if solution_token is None:
        return predicted_move, 0.0

    ranked_legal_ids = [int(legal_ids[idx]) for idx in legal_ranks.tolist()]
    try:
        solution_rank = ranked_legal_ids.index(solution_token) + 1
    except ValueError:
        return predicted_move, 0.0
    mrr = 1.0 / solution_rank
    return predicted_move, mrr


def _solution_token_id(*, solution: str, turn: object) -> int | None:
    prefix = "w:" if str(turn) == "White" else "b:"
    return MOVE_TO_ID.get(prefix + solution)


def _normalized_fen(fen: str) -> str:
    return chess.Board(fen).fen(en_passant="legal")


def _source_game_for_puzzle(
    puzzle: dict[str, Any],
    source_game_cache: dict[str, SourceGameRecord] | None,
) -> SourceGameRecord | None:
    if not source_game_cache:
        return None

    game_url = puzzle.get("game_url")
    if not game_url:
        return None

    try:
        game_id = _extract_lichess_game_id(str(game_url))
    except ValueError:
        return None

    return source_game_cache.get(game_id)


def _game_for_puzzle(*, board: bulletchess.Board, source_game: SourceGameRecord | None) -> Game:
    if source_game is not None:
        try:
            return source_game.to_game(puzzle_fen=board.fen())
        except Exception as exc:
            logger.warning(
                "Puzzle source-game reconstruction failed for {}: {}; "
                "falling back to FEN-only context",
                source_game.game_id,
                exc,
            )

    return Game(
        target_outcome_token=UNKNOWN_RESULT_ID,
        white_elo_token=ELO_UNKNOWN_ID,
        black_elo_token=ELO_UNKNOWN_ID,
        board=board,
    )
