import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from src.tokenizer import Tokenizer

UCI_MOVES_PATH = Path("data/all_uci_moves.txt")


@dataclass
class Puzzle:
    fen: str
    solution: str
    rating: int


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
                    )
                )
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"[WARN] Skipping malformed line {line_num}: {e}", file=sys.stderr)
    return puzzles


def load_model(checkpoint_path: str):  # noqa: ARG001
    # TODO: Instantiate KrasnalModel(config), then:
    return None


def predict_next_move(model, tokenizer: Tokenizer, fen: str) -> str:  # noqa: ARG001
    # TODO: Convert FEN to a move-sequence context, encode, run forward pass:
    return "e2e4"


def rating_bucket(rating: int) -> str:
    low = (rating // 200) * 200
    return f"{low}-{low + 199}"


def evaluate(puzzles: list[Puzzle], model, tokenizer: Tokenizer) -> EvalResults:
    results = EvalResults()

    for puzzle in puzzles:
        results.total += 1
        bucket = rating_bucket(puzzle.rating)

        try:
            predicted = predict_next_move(model, tokenizer, puzzle.fen)
            is_correct = predicted.strip() == puzzle.solution.strip()
        except Exception as e:
            print(f"[WARN] Prediction failed for FEN '{puzzle.fen}': {e}", file=sys.stderr)
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
    model = load_model(args.model)
    results = evaluate(puzzles, model, tokenizer)
    print_summary(results, args.model)


if __name__ == "__main__":
    main()
