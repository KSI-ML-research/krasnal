"""Benchmark the production inference path used by the UCI provider.

Measures end-to-end latency of `ModelProvider.get_best_move()` using the
same persistent-session inference path as production.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import bulletchess
import torch
from tqdm import tqdm

from krasnal.inference.exceptions import NoLegalMovesError
from krasnal.tokens import WHITE_WON_ID
from krasnal.uci_engine.provider import ModelProvider


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _print_results(move_times: list[float], plies_completed: int) -> None:
    avg_time = sum(move_times) / len(move_times) * 1000
    min_time = min(move_times) * 1000
    max_time = max(move_times) * 1000
    total_time = sum(move_times)
    throughput = plies_completed / total_time if total_time > 0 else 0.0

    print()
    print("Results:")
    print(f"  Avg time/move: {avg_time:.2f}ms")
    print(f"  Min: {min_time:.2f}ms, Max: {max_time:.2f}ms")
    print(f"  Total benchmark time: {total_time:.2f}s")
    print(f"  Throughput: {throughput:.2f} moves/s")


def _run_one_game(
    provider: ModelProvider,
    *,
    moves_per_game: int,
    measure: bool,
) -> tuple[list[float], int]:
    board = bulletchess.Board()
    uci_moves: list[str] = []
    move_times: list[float] = []
    plies_completed = 0

    provider.reset_session(WHITE_WON_ID)

    for _ in range(moves_per_game):
        if not list(board.legal_moves()):
            break

        history = " ".join(uci_moves)
        _sync_device(provider.device)
        start = time.perf_counter()
        try:
            best_move = provider.get_best_move(history)
        except NoLegalMovesError:
            break
        _sync_device(provider.device)

        move = bulletchess.Move.from_uci(best_move)
        legal_moves = {candidate.uci() for candidate in board.legal_moves()}
        if best_move not in legal_moves:
            raise ValueError(
                f"Provider returned illegal move {best_move} for position {board.fen()}"
            )

        board.apply(move)
        uci_moves.append(best_move)
        plies_completed += 1

        if measure:
            move_times.append(time.perf_counter() - start)

    return move_times, plies_completed


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark production inference path")
    parser.add_argument("--directory", required=True, help="Artifact directory containing model.pt")
    parser.add_argument("--num_games", type=int, default=100, help="Number of games to benchmark")
    parser.add_argument("--moves_per_game", type=int, default=40, help="Maximum plies per game")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup games before benchmarking")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for deterministic sampling"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to run inference on",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    artifact_dir = Path(args.directory)
    device = torch.device(args.device)

    print("Loading model...")
    provider = ModelProvider.from_artifact_dir(artifact_dir, device=device)
    params_m = provider.model.get_num_params() / 1_000_000
    cfg = provider.model.config

    print()
    print("=== Production Inference Benchmark ===")
    print(f"Artifact: {artifact_dir}")
    print(f"Model: {params_m:.1f}M params")
    print(
        f"Config: layers={cfg.n_layer}, heads={cfg.n_head},"
        f" embd={cfg.n_embd}, block={cfg.block_size}"
    )
    print(f"Device: {args.device}")
    print(f"Games: {args.num_games}, Moves/game: {args.moves_per_game}")
    print()

    for _ in range(args.warmup):
        _run_one_game(provider, moves_per_game=args.moves_per_game, measure=False)

    move_times: list[float] = []
    plies_completed = 0
    for _ in tqdm(range(args.num_games), desc="Games"):
        game_times, game_plies = _run_one_game(
            provider,
            moves_per_game=args.moves_per_game,
            measure=True,
        )
        move_times.extend(game_times)
        plies_completed += game_plies

    if not move_times:
        raise RuntimeError("Benchmark completed zero plies")

    _print_results(move_times, plies_completed)


if __name__ == "__main__":
    main()
