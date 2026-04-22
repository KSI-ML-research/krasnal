"""Benchmark inference speed of chess language model.

Usage:
    uv run scripts/benchmarks/inference_speed.py \
        --directory <path> \
        --num_games 100 \
        --moves_per_game 20 \
        --device cpu

Measures token generation throughput (ms/token) by generating random tokens.
Does not include legal move filtering - benchmark measures pure model forward pass.
"""

import argparse
import os
import time

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from krasnal.config import GPTConfig
from krasnal.inference.session import InferenceSession
from krasnal.inference.utils import load_model
from krasnal.tokens import WHITE_WON_ID


def find_model_in_directory(directory: str):
    name = "model.pt"
    path = os.path.join(directory, name)
    if os.path.isfile(path):
        return path
    raise FileNotFoundError(f"No model.pt found in {directory}")


def find_config_in_directory(directory: str):
    for name in ["config.yaml", "config.json"]:
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"No config.yaml or config.json found in {directory}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference speed")
    parser.add_argument("--directory", help="Directory containing model.pt and config.yaml")
    parser.add_argument("--num_games", type=int, default=100, help="Number of games to benchmark")
    parser.add_argument("--moves_per_game", type=int, default=20, help="Moves to generate per game")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup games before benchmarking")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on",
    )
    args = parser.parse_args()

    if not args.directory:
        parser.error("--directory is required")

    checkpoint = find_model_in_directory(args.directory)
    config = find_config_in_directory(args.directory)

    device = torch.device(args.device)

    print("Loading model...")
    cfg = OmegaConf.load(config)
    gpt_cfg = GPTConfig(
        vocab_size=cfg.vocab_size,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
        bias=cfg.bias,
    )
    model = load_model(checkpoint, device, gpt_cfg)
    params_M = model.get_num_params() / 1_000_000

    session = InferenceSession(model, device, outcome_token=WHITE_WON_ID)

    print("Warming up...")
    for _ in range(args.warmup):
        session.reset(WHITE_WON_ID)
        for _ in range(args.moves_per_game):
            probs = session.get_raw_probs()
            token = torch.multinomial(probs, 1).item()
            session.feed_token(token)

    model_name = cfg.name if hasattr(cfg, "name") else "unknown"
    print()
    print("=== Inference Speed Benchmark ===")
    print(f"Model: {model_name} ({cfg.n_layer} layers, {cfg.n_head} heads, {cfg.n_embd} embed)")
    print(f"Parameters: {params_M:.1f}M, Vocab: {cfg.vocab_size}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Config: {config}")
    print(f"Device: {args.device}")
    print(f"Games: {args.num_games}, Moves/game: {args.moves_per_game}")
    print()

    print("Benchmarking...")
    all_times = []
    for _game_idx in tqdm(range(args.num_games), desc="Games"):
        session.reset(WHITE_WON_ID)
        for _ in range(args.moves_per_game):
            start = time.perf_counter()
            probs = session.get_raw_probs()
            end = time.perf_counter()
            all_times.append(end - start)
            token = torch.multinomial(probs, 1).item()
            session.feed_token(token)

    avg_time = sum(all_times) / len(all_times) * 1000
    min_time = min(all_times) * 1000
    max_time = max(all_times) * 1000
    total_time = sum(all_times)

    print()
    print("Results:")
    print(f"  Avg time/token: {avg_time:.2f}ms")
    print(f"  Min: {min_time:.2f}ms, Max: {max_time:.2f}ms")
    print(f"  Total benchmark time: {total_time:.2f}s")


if __name__ == "__main__":
    main()
