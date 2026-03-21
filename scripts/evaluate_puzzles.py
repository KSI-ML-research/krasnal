"""Evaluate a trained Krasnal model on filtered Lichess puzzles.

Loads puzzles from a JSONL file (produced by prepare_puzzles), fetches the
full game history from the Lichess API for each puzzle, and measures the
model's move prediction accuracy (Pass@1) broken down by rating bucket.

Usage:
    uv run scripts/evaluate_puzzles.py \
        --puzzles data/puzzles_filtered.jsonl \
            --model models/krasnal_base.pt
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import chess
import requests
import torch

from src.tokenizer import Tokenizer

UCI_MOVES_PATH = Path("data/all_uci_moves.txt")


@dataclass
class Puzzle:
    fen: str
    solution: str
    rating: int
    game_url: str = ""


@dataclass
class EvalResults:
    total: int = 0
    correct: int = 0
    errors: int = 0
    per_rating_bucket: dict[str, list[bool]] = field(default_factory=dict)

    @property
    def pass_at_1(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0


def load_puzzles(path: Path) -> list[Puzzle]:
    puzzles = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                puzzles.append(
                    Puzzle(
                        fen=obj["fen"],
                        solution=obj["solution"],
                        rating=int(obj["rating"]),
                        game_url=obj.get("game_url", ""),
                    )
                )
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"[WARN] Skipping malformed line {line_num}: {e}", file=sys.stderr)
    return puzzles


def load_model(checkpoint_path: str, device: torch.device):
    from inference.utils import load_model as _load_model

    model, _ = _load_model(checkpoint_path, device)
    return model


def _fen_position_matches(board: chess.Board, target_fen: str) -> bool:
    """Compare board position ignoring halfmove/fullmove counters."""
    board_parts = board.fen().split()
    target_parts = target_fen.split()
    # Compare: piece placement, side to move, castling rights, en passant
    return board_parts[:4] == target_parts[:4]


def fetch_uci_history(game_url: str, fen: str) -> list[str] | None:
    """Fetch the UCI move history up to the puzzle position via Lichess API.

    Retrieves the full game from Lichess using game_url, replays the moves,
    and returns the move list trimmed to the position matching fen.
    Returns None if the game cannot be fetched or the position is not found.
    """
    if not game_url:
        return None

    # Extract game_id from URL, e.g.:
    # "https://lichess.org/abc123XYZ"       -> "abc123XYZ"
    # "https://lichess.org/abc123XYZ/white" -> "abc123XYZ"
    # "https://lichess.org/abc123XYZ#45"    -> "abc123XYZ"
    segments = game_url.rstrip("/").split("/")
    game_id = segments[-2] if segments[-1] in ("white", "black") else segments[-1]
    game_id = game_id.split("#")[0]

    try:
        response = requests.get(
            f"https://lichess.org/game/export/{game_id}",
            headers={"Accept": "application/json"},
            params={"moves": "true", "clocks": "false", "opening": "false"},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        print(f"[WARN] Failed to fetch game {game_url}: {e}", file=sys.stderr)
        return None

    moves_str = data.get("moves", "")
    if not moves_str:
        print(f"[WARN] No moves in game {game_url}", file=sys.stderr)
        return None

    uci_moves = moves_str.split()

    # Replay moves and find the position matching the puzzle FEN
    board = chess.Board()
    for i, uci in enumerate(uci_moves):
        if _fen_position_matches(board, fen):
            return uci_moves[:i]
        try:
            board.push_uci(uci)
        except ValueError:
            print(f"[WARN] Illegal move {uci} in game {game_url}", file=sys.stderr)
            return None

    # Check position after the last move
    if _fen_position_matches(board, fen):
        return uci_moves

    print(f"[WARN] FEN not found in game {game_url}", file=sys.stderr)
    return None


def predict_next_move(
    model,
    tokenizer: Tokenizer,
    game_url: str,
    fen: str,
) -> str | None:
    """Predict the next move for a puzzle given its game context.

    Fetches the UCI history up to the puzzle position, encodes it with the
    tokenizer, runs a forward pass, and returns the predicted move in UCI
    notation. Returns None if the game history cannot be retrieved.

    TODO: implement forward pass once model loading is complete.
    """
    uci_history = fetch_uci_history(game_url, fen)
    if uci_history is None:
        return None

    # TODO: encode uci_history with tokenizer, run model forward pass
    _ = tokenizer  # noqa: F841
    _ = model  # noqa: F841
    return None


def rating_bucket(rating: int) -> str:
    low = (rating // 200) * 200
    return f"{low}-{low + 199}"


def evaluate(puzzles: list[Puzzle], model, tokenizer: Tokenizer) -> EvalResults:
    results = EvalResults()

    for puzzle in puzzles:
        results.total += 1
        bucket = rating_bucket(puzzle.rating)

        try:
            predicted = predict_next_move(model, tokenizer, puzzle.game_url, puzzle.fen)
            if predicted is None:
                results.errors += 1
                results.per_rating_bucket.setdefault(bucket, []).append(False)
                continue
            is_correct = predicted.strip() == puzzle.solution.strip()
        except Exception as e:
            print(f"[WARN] Prediction failed for puzzle '{puzzle.fen}': {e}", file=sys.stderr)
            results.errors += 1
            results.per_rating_bucket.setdefault(bucket, []).append(False)
            continue

        results.correct += int(is_correct)
        results.per_rating_bucket.setdefault(bucket, []).append(is_correct)

    return results


def print_summary(results: EvalResults, model_path: str) -> None:
    sep = "=" * 52
    print(sep)
    print("  Krasnal — Puzzle Evaluation Summary")
    print(sep)
    print(f"  Model checkpoint : {model_path}")
    print(f"  Total puzzles    : {results.total:>10,}")
    print(f"  Correct (Pass@1) : {results.correct:>10,}")
    print(f"  Errors / skipped : {results.errors:>10,}")
    print(f"  Pass@1 accuracy  : {results.pass_at_1:>10.2%}")
    print(sep)
    print("  Accuracy by rating bucket:")
    for bucket in sorted(results.per_rating_bucket.keys()):
        outcomes = results.per_rating_bucket[bucket]
        acc = sum(outcomes) / len(outcomes) if outcomes else 0.0
        print(f"    {bucket:>10}  →  {acc:.2%}  ({sum(outcomes)}/{len(outcomes)})")
    print(sep)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Krasnal chess engine on filtered Lichess puzzles."
    )
    parser.add_argument(
        "--puzzles",
        type=Path,
        default=Path("data/puzzles_filtered.jsonl"),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/krasnal_base.pt",
        metavar="CHECKPOINT",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    puzzles = load_puzzles(args.puzzles)
    if not puzzles:
        print("[ERROR] No puzzles loaded. Check the input file.", file=sys.stderr)
        sys.exit(1)

    tokenizer = Tokenizer(UCI_MOVES_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device)
    results = evaluate(puzzles, model, tokenizer)
    print_summary(results, args.model)


if __name__ == "__main__":
    main()
