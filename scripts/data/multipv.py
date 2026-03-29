#!/usr/bin/env python3
"""Get top N moves from Stockfish for a given position."""

import argparse
from pathlib import Path

import chess
import chess.engine


def build_board(moves: str) -> chess.Board:
    """Build board from UCI moves string."""
    board = chess.Board()
    for move in moves.split():
        board.push_uci(move)
    return board


def get_multipv_moves(
    position: str,
    depth: int,
    multipv: int,
    stockfish_path: Path,
) -> list[str]:
    """Get top N moves from Stockfish for a position.

    Args:
        position: UCI moves string (e.g., "e2e4 e7e5 g1f3")
        depth: Search depth
        multipv: Number of top moves to return
        stockfish_path: Path to Stockfish executable

    Returns:
        List of top N moves in UCI format
    """
    board = build_board(position)
    limit = chess.engine.Limit(depth=depth)

    engine = chess.engine.SimpleEngine.popen_uci(str(stockfish_path))
    try:
        infos = engine.analyse(board, limit, multipv=multipv)
        if isinstance(infos, dict):
            infos = [infos]

        moves = []
        for info in infos:
            pv = info.get("pv", [])
            if pv:
                moves.append(pv[0].uci())
        return moves
    finally:
        engine.quit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Get top N moves from Stockfish")
    parser.add_argument("--position", type=str, required=True, help="UCI moves (e.g., 'e2e4 e7e5')")
    parser.add_argument("--depth", type=int, required=True, help="Search depth")
    parser.add_argument("--multipv", type=int, default=3, help="Number of top moves to return")
    parser.add_argument("--stockfish-path", type=Path, default=Path("/opt/homebrew/bin/stockfish"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    moves = get_multipv_moves(
        position=args.position,
        depth=args.depth,
        multipv=args.multipv,
        stockfish_path=args.stockfish_path,
    )

    for i, move in enumerate(moves):
        print(f"{i + 1}: {move}")


if __name__ == "__main__":
    main()
