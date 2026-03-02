"""Baseline evaluation: legal move metrics for the chess model."""

import argparse
import logging
import random
from datetime import datetime
from pathlib import Path

import chess
import chess.engine
import matplotlib.pyplot as plt
import polars as pl
import torch
from tqdm.auto import tqdm

from config import (
    DATASET_PATH,
    DRAW_ID,
    EOS_ID,
    MODEL_PATH,
    PAD_ID,
    SOS_ID,
    WIN_BLACK_ID,
    WIN_WHITE_ID,
)
from dataset import ChessDataset
from inference import InferenceSession, get_legal_token_ids, load_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate chess model on legal move metrics")
    parser.add_argument("--num-games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(MODEL_PATH),
        help="Path to the model checkpoint.",
    )
    parser.add_argument(
        "--stockfish-path",
        type=str,
        default="stockfish",
        help="Path to Stockfish binary (default: 'stockfish' from PATH).",
    )
    parser.add_argument(
        "--stockfish-time",
        type=float,
        default=0.05,
        help="Time limit (seconds) for Stockfish evaluations.",
    )
    parser.add_argument(
        "--skip-acpl",
        action="store_true",
        help="Skip Stockfish/ACPL entirely (legal move metrics only).",
    )
    return parser.parse_args()


def compute_acpl(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    model_move_uci: str,
    time_limit: float = 0.05,
) -> float:
    """Compute Centipawn Loss (CPL) for a single move."""
    try:
        limit = chess.engine.Limit(time=time_limit)
        info = engine.analyse(board, limit=limit)
        best_score = (
            info["score"].pov(board.turn).score(mate_score=10000) if info and "score" in info else 0
        )

        # if model played the best move according to stockfish, return 0
        if info.get("pv") and model_move_uci == info["pv"][0].uci():
            return 0

        model_move = chess.Move.from_uci(model_move_uci)
        if model_move in board.legal_moves:
            board_copy = board.copy()
            board_copy.push(model_move)
            score_after = (
                engine.analyse(board_copy, limit=limit)["score"]
                .pov(board.turn)
                .score(mate_score=10000)
            )
            return max(best_score - score_after, 0)
    except Exception as e:
        logger.debug(f"Error computing ACPL: {e}")
    return 0


def evaluate(
    model,
    tokenizer,
    dataset: ChessDataset,
    num_games: int,
    device: torch.device,
    stockfish_path: str = "stockfish",
    stockfish_time=0.05,
    skip_acpl=False,
):
    special_ids = {SOS_ID, EOS_ID, PAD_ID, WIN_WHITE_ID, WIN_BLACK_ID, DRAW_ID}
    block_size = model.config.block_size

    engine = None
    if not skip_acpl:
        try:
            engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        except Exception as e:
            logger.error(f"Failed to start Stockfish at '{stockfish_path}': {e}")
            logger.error("Run with --skip-acpl to evaluate without ACPL.")
            raise SystemExit(1) from None

    # sample games
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[:num_games]

    results = []
    session = InferenceSession(model, device)

    for idx in tqdm(indices, desc="Evaluating games"):
        token_ids = dataset[idx].tolist()
        outcome_token = (
            token_ids[0]
            if token_ids and token_ids[0] in {WIN_WHITE_ID, WIN_BLACK_ID, DRAW_ID, SOS_ID}
            else SOS_ID
        )

        moves = [t for t in token_ids if t not in special_ids]
        if len(moves) < 1:
            continue

        if len(moves) + 1 > block_size:
            logger.error(f"Game {idx} length ({len(moves) + 1}) exceeds {block_size=}. Skipping.")
            continue

        session.reset(outcome_token)
        board = chess.Board()

        for i, move_token in enumerate(moves):
            legal_ids = get_legal_token_ids(board, tokenizer)
            if not legal_ids:
                break

            probs = session.get_probs()

            # Basic Metrics
            top1_id = probs.argmax().item()
            top1_is_legal = top1_id in legal_ids

            # Illegal Mass
            legal_mass = probs[legal_ids].sum().item()
            illegal_mass = 1.0 - legal_mass

            # Forced-Legal ACPL
            cpl = None
            if engine and legal_ids:
                best_uci = tokenizer.id_to_move.get(
                    legal_ids[int(probs[legal_ids].argmax().item())], ""
                )
                if best_uci:
                    cpl = compute_acpl(engine, board, best_uci, stockfish_time)

            results.append(
                {
                    "move_num": i + 1,
                    "top1_legal": top1_is_legal,
                    "illegal_mass": illegal_mass,
                    "acpl": cpl,
                }
            )

            # Advance board and session with the actual move
            uci_move = tokenizer.id_to_move.get(move_token, "")
            try:
                board.push_uci(uci_move)
            except (chess.InvalidMoveError, chess.IllegalMoveError, ValueError):
                break
            session.feed(move_token)

    if engine:
        engine.quit()

    return pl.DataFrame(results) if results else pl.DataFrame()


def print_results(df):
    if df.is_empty():
        print("No positions evaluated.")
        return

    # Overall Summary
    total_n = len(df)
    legal_rate = df["top1_legal"].mean() * 100
    avg_illegal_mass = df["illegal_mass"].mean() * 100

    print("\n=== Evaluation Results ===\n")
    print(f"Positions evaluated:      {total_n:,}")
    print(f"Top-1 Legal Move Rate:    {legal_rate:.1f}%")
    print(f"Mean Illegal Prob Mass:   {avg_illegal_mass:.1f}%")

    acpl_df = df.filter(pl.col("acpl").is_not_null())
    if not acpl_df.is_empty():
        avg_acpl = acpl_df["acpl"].mean()
        print(f"Average Centipawn Loss:   {avg_acpl:.1f}")

    # Phase Breakdown
    print("\nMetrics by Phase:")
    print(f"{'Phase':<18} | {'Legal %':>8} | {'Ill. Mass %':>12} | {'ACPL':>8}\n" + "-" * 57)
    for label, (start, end) in [
        ("Opening (1-10)", (1, 10)),
        ("Middle (11-30)", (11, 30)),
        ("Endgame (31+)", (31, 999)),
    ]:
        pdf = df.filter(pl.col("move_num").is_between(start, end))
        if pdf.is_empty():
            continue
        acpl = pdf["acpl"].drop_nulls().mean()
        legal_r = pdf["top1_legal"].mean() * 100
        ill_m = pdf["illegal_mass"].mean() * 100
        acpl_s = f"{acpl:.1f}" if acpl else "N/A"
        print(f"{label:<18} | {legal_r:>7.1f}% | {ill_m:>11.1f}% | {acpl_s:>8}")

    print("\nTrend by Move Number:")
    trends = _bin_by_move_number(df)
    for row in trends.iter_rows(named=True):
        start, l_rate, acpl = row["bin_start"], row["top1_legal"] * 100, row["acpl"]
        acpl_str = f"{acpl:>5.0f}" if acpl is not None else "  N/A"
        bar = "#" * int(l_rate / 5)
        print(f"Moves {start:2}-{start + 9:<2} | Legal: {l_rate:>5.1f}% | ACPL: {acpl_str} | {bar}")


def _bin_by_move_number(df: pl.DataFrame) -> pl.DataFrame:
    """Bin moves into groups of 10 and compute mean metrics."""
    return (
        df.with_columns((((pl.col("move_num") - 1) // 10) * 10 + 1).alias("bin_start"))
        .group_by("bin_start")
        .mean()
        .sort("bin_start")
    )


def save_plot(df: pl.DataFrame, path: str) -> None:
    """Save a trend plot of Legal Rate and ACPL by move number."""
    trends = _bin_by_move_number(df)
    labels = [f"{b}–{b + 9}" for b in trends["bin_start"]]

    fig, ax_legal = plt.subplots(figsize=(10, 4))
    ax_legal.plot(labels, trends["top1_legal"] * 100, marker="o", color="steelblue")
    ax_legal.set(ylabel="Top-1 Legal Rate (%)", title="Evaluation Trends", ylim=(0, 105))
    ax_legal.grid(axis="y", alpha=0.3)

    if df["acpl"].is_not_null().any():
        ax_acpl = ax_legal.twinx()
        ax_acpl.plot(labels, trends["acpl"], marker="s", color="tomato", alpha=0.7)
        ax_acpl.set(ylabel="ACPL")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {out}")


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading model...")
    model, tokenizer = load_model(args.model_path, device)

    print("Loading dataset...")
    dataset = ChessDataset(DATASET_PATH)
    print(f"Dataset: {len(dataset)} games, evaluating {args.num_games}")

    stats = evaluate(
        model,
        tokenizer,
        dataset,
        args.num_games,
        device,
        args.stockfish_path,
        args.stockfish_time,
        args.skip_acpl,
    )
    print_results(stats)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    plot_path = results_dir / f"eval_{timestamp}.png"
    save_plot(stats, str(plot_path))

    stats_path = results_dir / f"eval_{timestamp}.csv"
    stats.write_csv(stats_path)
    print(f"Stats saved to {stats_path}")


if __name__ == "__main__":
    main()
