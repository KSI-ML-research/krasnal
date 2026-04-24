#!/usr/bin/env python3
import argparse
import datetime
import os
import sys
from pathlib import Path

import chess
import chess.engine
import chess.pgn

from krasnal.utils import find_latest_model_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a match between two UCI engines.")
    parser.add_argument(
        "--white-artifact-dir",
        type=Path,
        default=None,
        help="Krasnal artifact directory for White.",
    )
    parser.add_argument(
        "--black-artifact-dir",
        type=Path,
        default=None,
        help="Krasnal artifact directory for Black.",
    )
    parser.add_argument(
        "--white-stockfish-binary",
        default=None,
        help="Stockfish binary path for White.",
    )
    parser.add_argument(
        "--black-stockfish-binary",
        default=None,
        help="Stockfish binary path for Black.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        required=True,
        help="Required search depth passed to both engines via UCI go depth.",
    )
    parser.add_argument("--games", type=int, default=20, help="Number of games to play.")
    parser.add_argument(
        "--pgn-out",
        type=Path,
        default=None,
        help="Optional PGN output path for all games.",
    )
    return parser.parse_args()


def build_krasnal_command(artifact_dir: Path) -> tuple[list[str], dict[str, str], str]:
    env = os.environ.copy()
    env["KRASNAL_ENGINE_PROVIDER"] = "model"
    env["KRASNAL_MODEL_ARTIFACT_DIR"] = str(artifact_dir)
    label = artifact_dir.parent.name + "/" + artifact_dir.name
    return [sys.executable, "-m", "krasnal.uci_engine.run"], env, f"Krasnal {label}"


def build_side(
    side_name: str,
    *,
    artifact_dir: Path | None,
    stockfish_binary: str | None,
) -> tuple[list[str], dict[str, str] | None, str]:
    if artifact_dir is not None and stockfish_binary is not None:
        raise ValueError(f"{side_name}: choose artifact dir or stockfish binary, not both")
    if artifact_dir is None and stockfish_binary is None:
        raise ValueError(f"{side_name}: one engine must be configured")

    if artifact_dir is not None:
        return build_krasnal_command(artifact_dir)
    return [stockfish_binary], None, f"Stockfish ({stockfish_binary})"


def open_engine(
    cmd: list[str],
    env: dict[str, str] | None,
) -> chess.engine.SimpleEngine:
    if env is None:
        return chess.engine.SimpleEngine.popen_uci(cmd)
    return chess.engine.SimpleEngine.popen_uci(cmd, env=env)


def adjudicate_no_move(board: chess.Board) -> str:
    return "0-1" if board.turn == chess.WHITE else "1-0"


def score_from_white_perspective(result: str) -> float:
    if result == "1-0":
        return 1.0
    if result == "0-1":
        return 0.0
    if result == "1/2-1/2":
        return 0.5
    raise ValueError(f"Unexpected game result: {result}")


def main() -> None:
    args = parse_args()
    white_artifact = args.white_artifact_dir
    black_artifact = args.black_artifact_dir

    if (
        white_artifact is None
        and args.white_stockfish_binary is None
        and black_artifact is not None
        and args.black_stockfish_binary is None
    ):
        white_artifact = find_latest_model_artifact()
    if (
        black_artifact is None
        and args.black_stockfish_binary is None
        and white_artifact is not None
        and args.white_stockfish_binary is None
    ):
        black_artifact = find_latest_model_artifact()

    white_cmd, white_env, white_label = build_side(
        "white",
        artifact_dir=white_artifact,
        stockfish_binary=args.white_stockfish_binary,
    )
    black_cmd, black_env, black_label = build_side(
        "black",
        artifact_dir=black_artifact,
        stockfish_binary=args.black_stockfish_binary,
    )

    limit = chess.engine.Limit(depth=args.depth)
    white_engine = open_engine(white_cmd, white_env)
    black_engine = open_engine(black_cmd, black_env)

    white_wins = 0
    black_wins = 0
    draws = 0
    white_score = 0.0
    pgn_games: list[chess.pgn.Game] = []

    try:
        for game_idx in range(args.games):
            swap_colors = game_idx % 2 == 1
            current_white = black_engine if swap_colors else white_engine
            current_black = white_engine if swap_colors else black_engine
            current_white_label = black_label if swap_colors else white_label
            current_black_label = white_label if swap_colors else black_label

            board = chess.Board()
            game = chess.pgn.Game()
            game.headers["Event"] = "Engine Match"
            game.headers["Site"] = "Local"
            game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
            game.headers["Round"] = str(game_idx + 1)
            game.headers["White"] = current_white_label
            game.headers["Black"] = current_black_label
            node = game

            while not board.is_game_over():
                engine = current_white if board.turn == chess.WHITE else current_black
                result = engine.play(board, limit)
                if result.move is None:
                    game.headers["Termination"] = "engine returned no move"
                    break
                board.push(result.move)
                node = node.add_variation(result.move)

            final_result = (
                board.result(claim_draw=True) if board.is_game_over() else adjudicate_no_move(board)
            )
            game.headers["Result"] = final_result
            pgn_games.append(game)

            score_as_played = score_from_white_perspective(final_result)
            white_side_score = 1.0 - score_as_played if swap_colors else score_as_played

            white_score += white_side_score
            if white_side_score == 1.0:
                white_wins += 1
            elif white_side_score == 0.5:
                draws += 1
            else:
                black_wins += 1

            print(
                f"Game {game_idx + 1}/{args.games}: result={final_result} "
                f"white={current_white_label} black={current_black_label}"
            )
    finally:
        white_engine.quit()
        black_engine.quit()

    if args.pgn_out is not None:
        args.pgn_out.parent.mkdir(parents=True, exist_ok=True)
        with args.pgn_out.open("w") as handle:
            for game in pgn_games:
                print(game, file=handle, end="\n\n")

    print(f"depth={args.depth}")
    print(f"side_a={white_label}")
    print(f"side_b={black_label}")
    print(f"games={args.games}")
    print(f"side_a_wins={white_wins} draws={draws} side_b_wins={black_wins}")
    print(f"side_a_score={white_score / args.games:.3f}")


if __name__ == "__main__":
    main()
