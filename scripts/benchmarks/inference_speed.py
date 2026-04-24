"""Benchmark inference speed of chess language model.

Usage:
    uv run scripts/benchmarks/inference_speed.py \
        --checkpoint <path> \
        --config <path> \
        --num_games 100 \
        --moves_per_game 20 \
        --device cpu \
        --mode both

Measures token generation throughput (ms/token) by generating random tokens.
Does not include legal move filtering - benchmark measures pure model forward pass.
"""

import argparse
import time

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from krasnal.config import GPTConfig
from krasnal.inference.kv_cache import KVCache
from krasnal.inference.session import InferenceSession
from krasnal.inference.utils import load_model
from krasnal.tokens import ELO_UNKNOWN_ID, GAME_START_ID, WHITE_WON_ID


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def _build_kv_cache(model: torch.nn.Module, device: torch.device) -> KVCache:
    cfg = model.config
    return KVCache(
        batch_size=1,
        num_layers=cfg.n_layer,
        num_heads=cfg.n_head,
        head_dim=cfg.n_embd // cfg.n_head,
        max_seq_len=cfg.block_size,
        device=device,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )


def _sample_next_token(probs: torch.Tensor) -> int:
    return int(torch.multinomial(probs, 1).item())


def _benchmark_no_kv(
    model: torch.nn.Module,
    device: torch.device,
    num_games: int,
    moves_per_game: int,
    warmup: int,
) -> list[float]:
    session = InferenceSession(model, device, outcome_token=WHITE_WON_ID)

    for _ in range(warmup):
        session.reset(WHITE_WON_ID)
        for _ in range(moves_per_game):
            probs = session.get_raw_probs()
            session.feed_token(_sample_next_token(probs))

    token_times: list[float] = []
    for _ in tqdm(range(num_games), desc="Games (no-kv)"):
        session.reset(WHITE_WON_ID)
        for _ in range(moves_per_game):
            _sync_if_cuda(device)
            start = time.perf_counter()
            probs = session.get_raw_probs()
            _sync_if_cuda(device)
            token_times.append(time.perf_counter() - start)
            session.feed_token(_sample_next_token(probs))

    return token_times


def _benchmark_kv(
    model: torch.nn.Module,
    device: torch.device,
    num_games: int,
    moves_per_game: int,
    warmup: int,
) -> list[float]:
    def run_one_game(measure: bool) -> list[float]:
        kv_cache = _build_kv_cache(model, device)
        context = [GAME_START_ID, WHITE_WON_ID, ELO_UNKNOWN_ID, ELO_UNKNOWN_ID]
        times: list[float] = []

        x = torch.tensor([context], dtype=torch.long, device=device)
        logits, _ = model(x, past_kv=kv_cache)

        for _ in range(moves_per_game):
            probs = torch.softmax(logits[0, -1], dim=-1)
            token = _sample_next_token(probs)
            context.append(token)

            x = torch.tensor([[token]], dtype=torch.long, device=device)
            _sync_if_cuda(device)
            start = time.perf_counter()
            logits, _ = model(x, past_kv=kv_cache)
            _sync_if_cuda(device)
            if measure:
                times.append(time.perf_counter() - start)

        return times

    for _ in range(warmup):
        run_one_game(measure=False)

    token_times: list[float] = []
    for _ in tqdm(range(num_games), desc="Games (kv)"):
        token_times.extend(run_one_game(measure=True))

    return token_times


def _print_results(mode: str, token_times: list[float]) -> None:
    avg_time = sum(token_times) / len(token_times) * 1000
    min_time = min(token_times) * 1000
    max_time = max(token_times) * 1000
    total_time = sum(token_times)
    print()
    print(f"Results ({mode}):")
    print(f"  Avg time/token: {avg_time:.2f}ms")
    print(f"  Min: {min_time:.2f}ms, Max: {max_time:.2f}ms")
    print(f"  Total benchmark time: {total_time:.2f}s")


def _print_comparison(no_kv_times: list[float], kv_times: list[float]) -> None:
    no_kv_avg = sum(no_kv_times) / len(no_kv_times)
    kv_avg = sum(kv_times) / len(kv_times)
    speedup = no_kv_avg / kv_avg if kv_avg > 0 else float("inf")
    print()
    print("Comparison:")
    print(f"  no-kv avg: {no_kv_avg * 1000:.2f}ms/token")
    print(f"  kv avg:    {kv_avg * 1000:.2f}ms/token")
    print(f"  speedup:   {speedup:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference speed")
    parser.add_argument("--config", required=True, help="Path to model config YAML")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--num_games", type=int, default=100, help="Number of games to benchmark")
    parser.add_argument("--moves_per_game", type=int, default=20, help="Moves to generate per game")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup games before benchmarking")
    parser.add_argument(
        "--mode",
        choices=["no-kv", "kv", "both"],
        default="both",
        help="Benchmark mode: no-kv, kv, or both",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    print("Loading model...")
    cfg = OmegaConf.load(args.config)
    gpt_cfg = GPTConfig(
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
        bias=cfg.bias,
    )
    model = load_model(args.checkpoint, device, gpt_cfg)
    params_M = model.get_num_params() / 1_000_000

    model_name = cfg.name if hasattr(cfg, "name") else "unknown"
    print()
    print("=== Inference Speed Benchmark ===")
    print(f"Model: {model_name} ({cfg.n_layer} layers, {cfg.n_head} heads, {cfg.n_embd} embed)")
    print(f"Parameters: {params_M:.1f}M")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Device: {args.device}")
    print(f"Mode: {args.mode}")
    print(f"Games: {args.num_games}, Moves/game: {args.moves_per_game}")
    print()

    no_kv_times: list[float] | None = None
    kv_times: list[float] | None = None

    if args.mode in {"no-kv", "both"}:
        print("Benchmarking no-kv...")
        no_kv_times = _benchmark_no_kv(
            model,
            device,
            num_games=args.num_games,
            moves_per_game=args.moves_per_game,
            warmup=args.warmup,
        )
        _print_results("no-kv", no_kv_times)

    if args.mode in {"kv", "both"}:
        print("Benchmarking kv...")
        kv_times = _benchmark_kv(
            model,
            device,
            num_games=args.num_games,
            moves_per_game=args.moves_per_game,
            warmup=args.warmup,
        )
        _print_results("kv", kv_times)

    if args.mode == "both" and no_kv_times is not None and kv_times is not None:
        _print_comparison(no_kv_times, kv_times)


if __name__ == "__main__":
    main()
