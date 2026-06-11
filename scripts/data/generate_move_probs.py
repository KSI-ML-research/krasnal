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
import torch

from krasnal.config import CLOCK_IGNORE_ID
from krasnal.inference.batch import StatelessBatchInferenceSession
from krasnal.inference.game import Game
from krasnal.inference.utils import load_model
from krasnal.tokens import DRAW_ID, MOVE_VOCAB_PATH, load_move_vocab
from krasnal.uci_engine.provider import ModelProvider
from krasnal.utils import gpt_config_from_artifact_dict, resolve_runtime_device


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

    import json
    cfg_path = args.artifact_dir / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"config.json not found in {args.artifact_dir}")
    with open(cfg_path) as f:
        cfg_dict = json.load(f)
    cfg = gpt_config_from_artifact_dict(cfg_dict)

    piece_aware = bool(cfg_dict.get("piece_aware_moves", False))
    side_prefixed = bool(cfg_dict.get("side_prefixed_moves", False))

    vocab_candidates = [
        args.artifact_dir / "move_vocab.json",
        args.artifact_dir / "vocab.json",
    ]
    vocab_path = next((p for p in vocab_candidates if p.exists()), None)
    if vocab_path is None:
        vocab_path = MOVE_VOCAB_PATH
    load_move_vocab(vocab_path, piece_aware_moves=piece_aware, side_prefixed_moves=side_prefixed)
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

    # Build game states, clock sequences, and full legal probability vectors.
    # Rebuild each prefix Game instead of copying Game objects,
    # because bulletchess.Board can't be deep-copied/pickled safely.
    games: list[Game] = []
    mapping: list[tuple[int, int]] = []
    active_sequences: list[list[int]] = []
    opponent_sequences: list[list[int]] = []

    cw_col = df["clocks_white"].to_list()
    cb_col = df["clocks_black"].to_list()
    ti_col = df["time_initial"].to_list()

    for row_idx, uci_raw in enumerate(df.get_column("uci_moves")):
        if uci_raw is None:
            continue
        if isinstance(uci_raw, str):
            moves = uci_raw.split()
        elif isinstance(uci_raw, list):
            moves = [str(x) for x in uci_raw]
        else:
            continue

        ti = int(ti_col[row_idx]) if ti_col[row_idx] is not None else CLOCK_IGNORE_ID
        cw = [float(v) for v in cw_col[row_idx]] if cw_col[row_idx] is not None else []
        cb = [float(v) for v in cb_col[row_idx]] if cb_col[row_idx] is not None else []

        for move_idx in range(len(moves)):
            prefix_game = Game()
            for prev_uci in moves[:move_idx]:
                prefix_game.feed_uci(prev_uci)
            games.append(prefix_game)
            mapping.append((row_idx, move_idx))

            k = move_idx  # predicting move k

            # Compute clock state at decision point
            if k % 2 == 0:  # white to move
                if k == 0:
                    active_clock, opponent_clock = ti, ti
                else:
                    w_idx = (k - 2) // 2
                    b_idx = (k - 2) // 2
                    active_clock = int(cw[w_idx]) if w_idx < len(cw) else CLOCK_IGNORE_ID
                    opponent_clock = int(cb[b_idx]) if b_idx < len(cb) else CLOCK_IGNORE_ID
            else:  # black to move
                w_idx = (k - 1) // 2
                if k <= 1:
                    active_clock = ti
                else:
                    b_idx = (k - 3) // 2
                    active_clock = int(cb[b_idx]) if b_idx < len(cb) else CLOCK_IGNORE_ID
                opponent_clock = int(cw[w_idx]) if w_idx < len(cw) else CLOCK_IGNORE_ID

            ctx_tokens = prefix_game.context_tokens()
            seq_len = len(ctx_tokens)
            act_seq = [ti] * seq_len
            opp_seq = [ti] * seq_len
            act_seq[-1] = active_clock
            opp_seq[-1] = opponent_clock
            active_sequences.append(act_seq)
            opponent_sequences.append(opp_seq)

    if not games:
        df = df.with_columns(pl.Series("model_move_probs", [[] for _ in range(df.height)]))
        df.write_parquet(args.output)
        print(f"Wrote selected rows to {args.output} with empty model_move_probs")
        return
    legal_logits = batcher.get_legal_logits_batch(
        games,
        active_clock_sequences=active_sequences,
        opponent_clock_sequences=opponent_sequences,
        batch_size=args.batch_size,
    )
    legal_probs = torch.softmax(legal_logits, dim=-1).cpu().numpy()
    per_row: list[list[list[float]]] = [[] for _ in range(df.height)]
    for (row_idx, _move_idx), prob_vec in zip(mapping, legal_probs, strict=True):
        probs_only = prob_vec[prob_vec > 0].tolist()
        per_row[row_idx].append(probs_only)

    df = df.with_columns(pl.Series("model_move_probs", per_row))
    df.write_parquet(args.output)
    print(f"Wrote selected rows to {args.output}; added model_move_probs")


if __name__ == "__main__":
    main()
