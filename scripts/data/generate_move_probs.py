"""Generate per-move model probabilities or entropy for selected games only.

This script is intentionally scoped to a small set of games (for example the
100 games under analysis). It can either:
    - export full legal probability vectors per move, or
    - export per-move entropy directly from the provider.

The output is limited to the selected rows, so downstream builders can run only
on the games you want to inspect.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from krasnal.inference.batch import StatelessBatchInferenceSession
from krasnal.inference.game import Game
from krasnal.inference.utils import load_model
from krasnal.tokens import DRAW_ID
from krasnal.uci_engine.provider import ModelProvider
from krasnal.utils import build_gpt_config_from_artifact, resolve_runtime_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate per-move probs from model")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Optional explicit path to model checkpoint (.pt). Defaults to artifact-dir/model.pt.",
    )
    p.add_argument(
        "--artifact-dir",
        type=Path,
        required=True,
        help="Directory containing model artifact config.json (used to build GPTConfig)",
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument(
        "--use-provider",
        action="store_true",
        help="Use `ModelProvider` (from artifact dir) to compute move analysis (entropy) per move.",
    )
    p.add_argument(
        "--lichess-ids",
        nargs="*",
        default=None,
        help="Optional explicit lichess_id list. If provided, only these games are processed.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit after filtering (useful for the first 100 games).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_runtime_device()
    print(f"Using device: {device}")

    cfg = build_gpt_config_from_artifact(args.artifact_dir)
    model_path = args.model or (args.artifact_dir / "model.pt")
    if not model_path.exists():
        raise SystemExit(f"Checkpoint not found: {model_path}")
    model = load_model(str(model_path), device, cfg)
    batcher = StatelessBatchInferenceSession(model, device)

    provider: ModelProvider | None = None
    if args.use_provider:
        if args.artifact_dir is None:
            raise SystemExit("--artifact-dir is required when using --use-provider")
        provider = ModelProvider.from_artifact_dir(args.artifact_dir, device)

    df = pl.read_parquet(args.input)
    if args.lichess_ids:
        if "lichess_id" not in df.columns:
            raise SystemExit(
                "Input parquet has no lichess_id column, cannot filter by --lichess-ids"
            )
        df = df.filter(pl.col("lichess_id").cast(str).is_in([str(x) for x in args.lichess_ids]))
    if args.limit is not None:
        df = df.head(args.limit)

    if df.is_empty():
        raise SystemExit("No rows left after filtering")

    if args.use_provider and provider is not None:
        prob_rows: list[list[list[float]]] = []
        for uci_raw in df.get_column("uci_moves"):
            if uci_raw is None:
                prob_rows.append([])
                continue
            if isinstance(uci_raw, str):
                moves = uci_raw.split()
            elif isinstance(uci_raw, list):
                moves = [str(x) for x in uci_raw]
            else:
                prob_rows.append([])
                continue

            provider.reset_session(DRAW_ID)
            row_probs: list[list[float]] = []
            for move_idx in range(len(moves)):
                prefix = " ".join(moves[: move_idx + 1])
                probs = provider.get_legal_probs(prefix)
                row_probs.append([float(x) for x in probs.tolist()])
            prob_rows.append(row_probs)

        df = df.with_columns(pl.Series("model_move_probs", prob_rows))
        df.write_parquet(args.output)
        print(f"Wrote selected rows to {args.output} with model_move_probs")
        return

    # Build game states and full legal probability vectors.
    # Rebuild each prefix Game instead of copying Game objects,
    # because bulletchess.Board can't be deep-copied/pickled safely.
    games: list[Game] = []
    mapping: list[tuple[int, int]] = []
    for row_idx, uci_raw in enumerate(df.get_column("uci_moves")):
        if uci_raw is None:
            continue
        if isinstance(uci_raw, str):
            moves = uci_raw.split()
        elif isinstance(uci_raw, list):
            moves = [str(x) for x in uci_raw]
        else:
            continue

        for move_idx in range(len(moves)):
            prefix_game = Game()
            for prev_uci in moves[:move_idx]:
                prefix_game.feed_uci(prev_uci)
            games.append(prefix_game)
            mapping.append((row_idx, move_idx))

    if not games:
        df = df.with_columns(pl.Series("model_move_probs", [[] for _ in range(df.height)]))
        df.write_parquet(args.output)
        print(f"Wrote selected rows to {args.output} with empty model_move_probs")
        return

    probs_tensor = batcher.get_legal_probs_batch(games, batch_size=args.batch_size)
    probs_cpu = probs_tensor.cpu().numpy()
    per_row: list[list[list[float]]] = [[] for _ in range(df.height)]
    for (row_idx, _move_idx), prob_vec in zip(mapping, probs_cpu, strict=False):
        per_row[row_idx].append([float(x) for x in prob_vec.tolist()])

    df = df.with_columns(pl.Series("model_move_probs", per_row))
    df.write_parquet(args.output)
    print(f"Wrote selected rows to {args.output}; added model_move_probs")


if __name__ == "__main__":
    main()
