#!/usr/bin/env -S uv run python
"""Linear board-state probes for comparing checkpoints.

The probe freezes one or more Krasnal checkpoints, extracts hidden states at real
game positions, and trains a small linear classifier to predict the piece on
each board square. This is a lightweight diagnostic for the question: "is board
state linearly available in the model's hidden state?"

Examples:

    cd krasnal && uv run python scripts/diagnostics/board_state_probe.py \\
        --artifact-dir artifacts/pretrain/NO_QA \\
        --artifact-dir artifacts/pretrain/WITH_QA

    cd krasnal && uv run python scripts/diagnostics/board_state_probe.py \\
        --artifact-dir artifacts/pretrain/NO_QA \\
        --artifact-dir artifacts/pretrain/WITH_QA \\
        --max-train-positions 8192 --probe-steps 600
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bulletchess
import torch
import torch.nn.functional as F
import wandb
from torch.nn.utils.rnn import pad_sequence

from krasnal.config import CLOCK_IGNORE_ID, EVAL_DATASET_PATH, GPTConfig
from krasnal.dataset import ChessDataset
from krasnal.eval.parsers import parse_row_to_game_tokens
from krasnal.eval.replayer import replay_game_tokens
from krasnal.inference import load_model
from krasnal.inference.utils import create_amp_context
from krasnal.model import GPT
from krasnal.tokens import (
    COLORED_PIECE_TOKENS,
    EMPTY_ID,
    ID_TO_MOVE,
    PAD_ID,
    load_move_vocab,
    whats_on_answer_token_id,
)
from krasnal.utils import gpt_config_from_artifact_dict, read_model_config_json

LABEL_TOKEN_IDS: tuple[int, ...] = (EMPTY_ID, *sorted(COLORED_PIECE_TOKENS.values()))
LABEL_TOKEN_TO_CLASS = {token_id: i for i, token_id in enumerate(LABEL_TOKEN_IDS)}
LABEL_NAMES = tuple(ID_TO_MOVE[token_id] for token_id in LABEL_TOKEN_IDS)
SQUARES = tuple(f"{chr(97 + file_i)}{rank_i + 1}" for rank_i in range(8) for file_i in range(8))


@dataclass(frozen=True)
class ProbeExample:
    sequence: tuple[int, ...]
    active_clock_ids: tuple[int, ...]
    opponent_clock_ids: tuple[int, ...]
    labels: tuple[int, ...]


@dataclass(frozen=True)
class Artifact:
    name: str
    path: Path
    raw_config: dict[str, Any]
    config: GPTConfig


def _pick_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name in ("cpu", "cuda", "mps"):
        return torch.device(name)
    print("device must be auto|cpu|cuda|mps", file=sys.stderr)
    sys.exit(1)


def _read_artifact(path: Path) -> Artifact:
    for rel in ("config.json", "model.pt", "move_vocab.json"):
        p = path / rel
        if not p.is_file():
            print(f"Missing inference file: {p}", file=sys.stderr)
            sys.exit(1)
    raw = read_model_config_json(path / "config.json")
    return Artifact(
        name=path.name,
        path=path,
        raw_config=raw,
        config=gpt_config_from_artifact_dict(raw),
    )


def _install_artifact_vocab(artifact: Artifact) -> None:
    load_move_vocab(
        artifact.path / "move_vocab.json",
        piece_aware_moves=bool(artifact.raw_config.get("piece_aware_moves", False)),
        side_prefixed_moves=bool(artifact.raw_config.get("side_prefixed_moves", True)),
    )


def _board_labels(fen: str) -> tuple[int, ...]:
    board = bulletchess.Board.from_fen(fen)
    return tuple(
        LABEL_TOKEN_TO_CLASS[whats_on_answer_token_id(board, square)] for square in SQUARES
    )


def _clock_value(value: int | None) -> int:
    return CLOCK_IGNORE_ID if value is None else int(value)


def _example_from_context(ctx) -> ProbeExample | None:
    if ctx.sequence is None or ctx.actual_token is None:
        return None
    if ctx.active_clock_sequence is None or ctx.opponent_clock_sequence is None:
        return None
    if len(ctx.sequence) != len(ctx.active_clock_sequence):
        return None
    if len(ctx.sequence) != len(ctx.opponent_clock_sequence):
        return None

    actual_active = _clock_value(ctx.active_clock_seconds)
    actual_opponent = _clock_value(ctx.opponent_clock_seconds)

    if ctx.post_move_fen is None:
        return None
    sequence = [*ctx.sequence, ctx.actual_token]
    active = [*ctx.active_clock_sequence[1:], actual_active, CLOCK_IGNORE_ID]
    opponent = [*ctx.opponent_clock_sequence[1:], actual_opponent, CLOCK_IGNORE_ID]
    labels = _board_labels(ctx.post_move_fen)

    if not sequence:
        return None
    if len(sequence) != len(active) or len(sequence) != len(opponent):
        raise RuntimeError("probe sequence/clock length mismatch")
    return ProbeExample(
        sequence=tuple(int(x) for x in sequence),
        active_clock_ids=tuple(int(x) for x in active),
        opponent_clock_ids=tuple(int(x) for x in opponent),
        labels=labels,
    )


def _sample_game_contexts(contexts: list, positions_per_game: int, rng: random.Random) -> list:
    if positions_per_game <= 0 or len(contexts) <= positions_per_game:
        out = list(contexts)
        rng.shuffle(out)
        return out
    return rng.sample(contexts, positions_per_game)


def _collect_examples(
    *,
    eval_parquet: Path,
    include_elo: bool,
    block_size: int,
    max_train_positions: int,
    max_eval_positions: int,
    positions_per_game: int,
    max_games: int,
    seed: int,
) -> tuple[list[ProbeExample], list[ProbeExample]]:
    dataset = ChessDataset(eval_parquet, include_elo=include_elo)
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    if max_games > 0:
        indices = indices[:max_games]

    train: list[ProbeExample] = []
    eval_: list[ProbeExample] = []
    filling_train = True

    for idx in indices:
        row = dataset[idx]
        game_tokens = parse_row_to_game_tokens(row)
        if game_tokens is None:
            continue
        if len(game_tokens.initial_context) + len(game_tokens.body_tokens) > block_size:
            continue

        contexts = _sample_game_contexts(
            replay_game_tokens(game_tokens),
            positions_per_game=positions_per_game,
            rng=rng,
        )
        examples = [
            ex
            for ctx in contexts
            if (ex := _example_from_context(ctx)) is not None and len(ex.sequence) <= block_size
        ]
        if not examples:
            continue

        target = train if filling_train else eval_
        remaining = (
            max_train_positions - len(train) if filling_train else max_eval_positions - len(eval_)
        )
        target.extend(examples[:remaining])

        if filling_train and len(train) >= max_train_positions:
            filling_train = False
        if not filling_train and len(eval_) >= max_eval_positions:
            break

    if not train:
        raise ValueError("collected zero train probe examples")
    if not eval_:
        raise ValueError("collected zero eval probe examples")
    return train, eval_


def _labels_tensor(examples: list[ProbeExample]) -> torch.Tensor:
    return torch.tensor([ex.labels for ex in examples], dtype=torch.long)


def _layer_names(config: GPTConfig) -> tuple[str, ...]:
    return ("emb", *(f"block_{i}" for i in range(config.n_layer)), "final")


def _extract_hidden_states(
    model: GPT,
    examples: list[ProbeExample],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    names = _layer_names(model.config)
    chunks: dict[str, list[torch.Tensor]] = {name: [] for name in names}
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def capture(name: str):
        def hook(_module, _inputs, output):
            captured[name] = output

        return hook

    handles.append(model.transformer.drop.register_forward_hook(capture("emb")))
    for i, block in enumerate(model.transformer.h):
        handles.append(block.register_forward_hook(capture(f"block_{i}")))
    handles.append(model.transformer.ln_f.register_forward_hook(capture("final")))

    amp_ctx = create_amp_context(device)
    use_clock = model.config.use_clock_encodings
    try:
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            token_rows = [torch.tensor(ex.sequence, dtype=torch.long) for ex in batch]
            x = pad_sequence(token_rows, batch_first=True, padding_value=PAD_ID).to(device)
            lengths = torch.tensor(
                [len(ex.sequence) for ex in batch], dtype=torch.long, device=device
            )
            last = lengths - 1

            kwargs = {}
            if use_clock:
                active_rows = [torch.tensor(ex.active_clock_ids, dtype=torch.long) for ex in batch]
                opponent_rows = [
                    torch.tensor(ex.opponent_clock_ids, dtype=torch.long) for ex in batch
                ]
                kwargs["active_clock_ids"] = pad_sequence(
                    active_rows,
                    batch_first=True,
                    padding_value=CLOCK_IGNORE_ID,
                ).to(device)
                kwargs["opponent_clock_ids"] = pad_sequence(
                    opponent_rows,
                    batch_first=True,
                    padding_value=CLOCK_IGNORE_ID,
                ).to(device)

            captured.clear()
            with torch.inference_mode(), amp_ctx:
                model(x, return_all_logits=False, **kwargs)

            batch_idx = torch.arange(len(batch), device=device)
            for name in names:
                if name not in captured:
                    raise RuntimeError(f"missing hidden capture for {name}")
                hidden = captured[name][batch_idx, last].detach().float().cpu()
                chunks[name].append(hidden)

    finally:
        for handle in handles:
            handle.remove()

    return {name: torch.cat(parts, dim=0) for name, parts in chunks.items()}


def _majority_baseline(train_y: torch.Tensor, eval_y: torch.Tensor) -> dict[str, float]:
    square_majority = []
    for sq in range(train_y.size(1)):
        counts = torch.bincount(train_y[:, sq], minlength=len(LABEL_TOKEN_IDS))
        square_majority.append(int(torch.argmax(counts)))
    pred = torch.tensor(square_majority, dtype=torch.long).unsqueeze(0).expand_as(eval_y)
    return _score_predictions(pred, eval_y)


def _score_predictions(pred: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    correct = pred.eq(labels)
    occupied = labels.ne(0)
    pred_occupied = pred.ne(0)
    empty_vs_occupied = pred_occupied.eq(occupied)

    return {
        "acc": float(correct.float().mean().item()),
        "occupied_acc": float(correct[occupied].float().mean().item()) if occupied.any() else 0.0,
        "empty_vs_occupied_acc": float(empty_vs_occupied.float().mean().item()),
    }


def _train_linear_board_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    eval_x: torch.Tensor,
    eval_y: torch.Tensor,
    *,
    steps: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    n_features = train_x.size(1)
    n_classes = len(LABEL_TOKEN_IDS)
    head = torch.nn.Linear(n_features, 64 * n_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    train_x_d = ((train_x - mean) / std).to(device)
    train_y_d = train_y.to(device)
    eval_x_d = ((eval_x - mean) / std).to(device)
    eval_y_d = eval_y.to(device)

    for _step in range(steps):
        idx = torch.randint(
            low=0,
            high=train_x.size(0),
            size=(min(batch_size, train_x.size(0)),),
            generator=generator,
        ).to(device)
        logits = head(train_x_d[idx]).view(-1, 64, n_classes)
        loss = F.cross_entropy(logits.reshape(-1, n_classes), train_y_d[idx].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.inference_mode():
        logits = head(eval_x_d).view(-1, 64, n_classes)
        pred = logits.argmax(dim=-1).cpu()
        metrics = _score_predictions(pred, eval_y)
        metrics["loss"] = float(
            F.cross_entropy(logits.reshape(-1, n_classes), eval_y_d.reshape(-1)).item()
        )
    return metrics


def _load_checkpoint(artifact: Artifact, device: torch.device) -> GPT:
    _install_artifact_vocab(artifact)
    return load_model(str(artifact.path / "model.pt"), device, artifact.config)


def _run_artifact(
    artifact: Artifact,
    train_examples: list[ProbeExample],
    eval_examples: list[ProbeExample],
    *,
    device: torch.device,
    hidden_batch_size: int,
    probe_batch_size: int,
    probe_steps: int,
    probe_lr: float,
    probe_weight_decay: float,
    seed: int,
) -> dict[str, Any]:
    model = _load_checkpoint(artifact, device)
    train_y = _labels_tensor(train_examples)
    eval_y = _labels_tensor(eval_examples)
    baseline = _majority_baseline(train_y, eval_y)

    print(f"\n{artifact.name}: extracting train hidden states ({len(train_examples)} positions)")
    train_h = _extract_hidden_states(
        model,
        train_examples,
        device=device,
        batch_size=hidden_batch_size,
    )
    print(f"{artifact.name}: extracting eval hidden states ({len(eval_examples)} positions)")
    eval_h = _extract_hidden_states(
        model,
        eval_examples,
        device=device,
        batch_size=hidden_batch_size,
    )

    layer_metrics: dict[str, dict[str, float]] = {}
    for i, name in enumerate(_layer_names(artifact.config)):
        metrics = _train_linear_board_probe(
            train_h[name],
            train_y,
            eval_h[name],
            eval_y,
            steps=probe_steps,
            lr=probe_lr,
            weight_decay=probe_weight_decay,
            batch_size=probe_batch_size,
            seed=seed + i,
            device=device,
        )
        layer_metrics[name] = metrics
        print(
            f"  {name:>7s}  acc={metrics['acc']:.4f}  "
            f"occupied_acc={metrics['occupied_acc']:.4f}  "
            f"empty_vs_occ={metrics['empty_vs_occupied_acc']:.4f}"
        )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "artifact": str(artifact.path),
        "name": artifact.name,
        "baseline": baseline,
        "layers": layer_metrics,
    }


def _print_summary(results: list[dict[str, Any]]) -> None:
    print("\nSummary (linear board-state probe)")
    print("model                         layer        acc  occupied  empty/occ  d_acc_vs_base")
    print("-" * 82)
    for result in results:
        base_acc = result["baseline"]["acc"]
        best_layer, best = max(result["layers"].items(), key=lambda item: item[1]["acc"])
        print(
            f"{result['name']:<29s} {best_layer:>7s}  "
            f"{best['acc']:9.4f} {best['occupied_acc']:9.4f} "
            f"{best['empty_vs_occupied_acc']:9.4f} {best['acc'] - base_acc:14.4f}"
        )
    print("\nMajority-per-square baseline:")
    for result in results:
        b = result["baseline"]
        print(
            f"  {result['name']}: acc={b['acc']:.4f} "
            f"occupied_acc={b['occupied_acc']:.4f} "
            f"empty_vs_occ={b['empty_vs_occupied_acc']:.4f}"
        )


def _wandb_payload(payload: dict[str, Any]) -> dict[str, float | int | str]:
    out: dict[str, float | int | str] = {
        "probe/type": "board_state",
        "probe/train_positions": int(payload["train_positions"]),
        "probe/eval_positions": int(payload["eval_positions"]),
    }
    for result in payload["results"]:
        model = result["name"]
        for metric, value in result["baseline"].items():
            out[f"probe/board/{model}/baseline/{metric}"] = float(value)

        best_layer, best = max(result["layers"].items(), key=lambda item: item[1]["acc"])
        out[f"probe/board/{model}/best_layer"] = best_layer
        for metric, value in best.items():
            out[f"probe/board/{model}/best/{metric}"] = float(value)

        for layer, metrics in result["layers"].items():
            for metric, value in metrics.items():
                out[f"probe/board/{model}/{layer}/{metric}"] = float(value)
    return out


def _log_to_wandb(
    payload: dict[str, Any],
    *,
    project: str,
    name: str | None,
    group: str | None,
    json_out: Path | None,
) -> None:
    run = wandb.init(
        project=project,
        name=name,
        group=group,
        tags=["diagnostic", "probe", "board-state"],
        config={
            "probe_type": "board_state",
            "train_positions": payload["train_positions"],
            "eval_positions": payload["eval_positions"],
            "artifacts": [result["artifact"] for result in payload["results"]],
        },
    )
    metrics = _wandb_payload(payload)
    wandb.log(metrics)
    for key, value in metrics.items():
        run.summary[key] = value
    if json_out is not None and json_out.is_file():
        artifact = wandb.Artifact(json_out.stem, type="probe-results")
        artifact.add_file(str(json_out))
        wandb.log_artifact(artifact)
    wandb.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        action="append",
        required=True,
        help="Checkpoint artifact directory. Pass twice to compare no-QA vs QA.",
    )
    parser.add_argument(
        "--eval-parquet",
        type=Path,
        default=EVAL_DATASET_PATH,
        help=f"Eval Parquet path (default: {EVAL_DATASET_PATH}).",
    )
    parser.add_argument("--max-train-positions", type=int, default=4096)
    parser.add_argument("--max-eval-positions", type=int, default=2048)
    parser.add_argument(
        "--positions-per-game",
        type=int,
        default=4,
        help="Sample this many positions per game; <=0 uses all positions from selected games.",
    )
    parser.add_argument("--max-games", type=int, default=0, help="0 means scan until quotas fill.")
    parser.add_argument("--hidden-batch-size", type=int, default=64)
    parser.add_argument("--probe-batch-size", type=int, default=512)
    parser.add_argument("--probe-steps", type=int, default=300)
    parser.add_argument("--probe-lr", type=float, default=3e-3)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", help="cpu | cuda | mps | auto")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--wandb", action="store_true", help="Log probe metrics to W&B.")
    parser.add_argument("--wandb-project", default="krasnal")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    args = parser.parse_args()

    if args.max_train_positions <= 0 or args.max_eval_positions <= 0:
        print("max train/eval positions must be positive", file=sys.stderr)
        sys.exit(1)
    if args.probe_steps <= 0:
        print("--probe-steps must be positive", file=sys.stderr)
        sys.exit(1)

    artifacts = [_read_artifact(path) for path in args.artifact_dir]
    first = artifacts[0]
    _install_artifact_vocab(first)
    include_elo = bool(first.raw_config.get("include_elo", True))
    block_size = min(artifact.config.block_size for artifact in artifacts)
    device = _pick_device(args.device)

    print(
        f"Collecting post-move board labels from {args.eval_parquet} "
        f"(train={args.max_train_positions}, eval={args.max_eval_positions}, "
        f"positions_per_game={args.positions_per_game}, block_size={block_size})"
    )
    train_examples, eval_examples = _collect_examples(
        eval_parquet=args.eval_parquet,
        include_elo=include_elo,
        block_size=block_size,
        max_train_positions=args.max_train_positions,
        max_eval_positions=args.max_eval_positions,
        positions_per_game=args.positions_per_game,
        max_games=args.max_games,
        seed=args.seed,
    )
    print(
        f"Collected {len(train_examples)} train and {len(eval_examples)} eval positions "
        f"on device={device}. Labels={LABEL_NAMES}"
    )

    results = [
        _run_artifact(
            artifact,
            train_examples,
            eval_examples,
            device=device,
            hidden_batch_size=args.hidden_batch_size,
            probe_batch_size=args.probe_batch_size,
            probe_steps=args.probe_steps,
            probe_lr=args.probe_lr,
            probe_weight_decay=args.probe_weight_decay,
            seed=args.seed,
        )
        for artifact in artifacts
    ]
    payload = {
        "train_positions": len(train_examples),
        "eval_positions": len(eval_examples),
        "label_names": LABEL_NAMES,
        "results": results,
    }
    _print_summary(results)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote {args.json_out}")

    if args.wandb:
        _log_to_wandb(
            payload,
            project=args.wandb_project,
            name=args.wandb_name,
            group=args.wandb_group,
            json_out=args.json_out,
        )


if __name__ == "__main__":
    main()
