#!/usr/bin/env python3
"""Play a match between two Krasnal model versions.

Usage:
    uv run scripts/evals/play_krasnal_vs_krasnal.py \
        --model-a artifacts/pretrain/model_a \
        --model-b artifacts/pretrain/model_b \
        --games 20

Both models must have matching vocab_size and it must match the global vocabulary.
"""

import argparse
import json
from pathlib import Path

import torch

from krasnal.config import GPTConfig
from krasnal.inference import Game, InferenceSession
from krasnal.inference.sampling import sample_token
from krasnal.inference.utils import load_model
from krasnal.tokens import (
    WHITE_WON_ID,
    get_vocab_size,
    legal_token_ids,
    to_uci,
)


def load_model_from_dir(artifact_dir: Path, device: torch.device):
    config_path = artifact_dir / "config.json"
    checkpoint_path = artifact_dir / "model.pt"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    with open(config_path) as f:
        cfg = json.load(f)

    gpt_cfg = GPTConfig(
        vocab_size=cfg["vocab_size"],
        block_size=cfg["block_size"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
        dropout=cfg["dropout"],
    )

    model = load_model(str(checkpoint_path), device, gpt_cfg)
    model_name = cfg.get("name", artifact_dir.name)

    return model, gpt_cfg, model_name


def verify_vocab_compatibility(cfg_a: GPTConfig, cfg_b: GPTConfig) -> None:
    global_vocab = get_vocab_size()

    if cfg_a.vocab_size != global_vocab:
        raise ValueError(
            f"Model A vocab_size ({cfg_a.vocab_size}) != global vocab ({global_vocab})"
        )
    if cfg_b.vocab_size != global_vocab:
        raise ValueError(
            f"Model B vocab_size ({cfg_b.vocab_size}) != global vocab ({global_vocab})"
        )
    if cfg_a.vocab_size != cfg_b.vocab_size:
        raise ValueError(
            f"Model A vocab_size ({cfg_a.vocab_size}) != Model B vocab_size ({cfg_b.vocab_size})"
        )


def get_model_move(session: InferenceSession) -> str:
    legal_ids = legal_token_ids(session.game.board)
    if not legal_ids:
        raise ValueError("No legal moves available")

    legal_probs = session.get_legal_probs()
    best_token = sample_token(legal_probs, temperature=0.0, top_p=1.0)
    return to_uci(best_token)


def play_game(
    white_model: torch.nn.Module, black_model: torch.nn.Module, device: torch.device
) -> str:
    import bulletchess

    board = bulletchess.Board()

    white_session = InferenceSession(
        white_model, device, game=Game(target_outcome_token=WHITE_WON_ID)
    )
    black_session = InferenceSession(
        black_model, device, game=Game(target_outcome_token=WHITE_WON_ID)
    )

    for _ in range(120):
        if not board.legal_moves():
            return "1-0" if board.turn != bulletchess.WHITE else "0-1"

        if board.turn == bulletchess.WHITE:
            move = get_model_move(white_session)
            board.apply(bulletchess.Move.from_uci(move))
            white_session.feed_uci(move)
            black_session.feed_uci(move)
        else:
            move = get_model_move(black_session)
            board.apply(bulletchess.Move.from_uci(move))
            white_session.feed_uci(move)
            black_session.feed_uci(move)

    return "1/2-1/2"


def main():
    parser = argparse.ArgumentParser(description="Play Krasnal vs Krasnal match")
    parser.add_argument("--model-a", type=Path, required=True, help="Directory for model A")
    parser.add_argument("--model-b", type=Path, required=True, help="Directory for model B")
    parser.add_argument("--games", type=int, default=20, help="Number of games to play")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    print("Loading models...")
    model_a, cfg_a, name_a = load_model_from_dir(args.model_a, device)
    model_b, cfg_b, name_b = load_model_from_dir(args.model_b, device)

    print(f"Model A: {name_a} (vocab={cfg_a.vocab_size})")
    print(f"Model B: {name_b} (vocab={cfg_b.vocab_size})")

    verify_vocab_compatibility(cfg_a, cfg_b)
    print(f"Vocab compatibility verified (global vocab={get_vocab_size()})")

    a_wins = 0
    b_wins = 0
    draws = 0

    print(f"\nPlaying {args.games} games...")
    for i in range(args.games):
        swap_colors = i % 2 == 1
        white_model = model_b if swap_colors else model_a
        black_model = model_a if swap_colors else model_b
        white_name = name_b if swap_colors else name_a
        black_name = name_a if swap_colors else name_b

        result = play_game(white_model, black_model, device)

        if result == "1-0":
            if swap_colors:
                b_wins += 1
            else:
                a_wins += 1
        elif result == "0-1":
            if swap_colors:
                a_wins += 1
            else:
                b_wins += 1
        else:
            draws += 1

        print(f"Game {i + 1}/{args.games}: {result} (white={white_name}, black={black_name})")

    print("\n=== Results ===")
    print(f"{name_a}: {a_wins} wins")
    print(f"{name_b}: {b_wins} wins")
    print(f"Draws: {draws}")
    total = args.games
    score_a = (a_wins * 1.0 + draws * 0.5) / total
    print(f"Score: {name_a}={score_a:.3f} ({a_wins}/{draws}/{b_wins})")


if __name__ == "__main__":
    main()
