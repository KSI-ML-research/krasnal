#!/usr/bin/env -S uv run python
"""I/O clock probe on eval Parquet: replay real positions, vary only UCI leaf clocks.

Samples rows from ``eval.parquet`` (default ``data/2_tokenized/eval.parquet``), rebuilds the
board from stored ``token_ids`` (and optional clock columns for history), then measures how much
the **legal-move** distribution shifts when only ``wtime``/``btime`` change. Also counts **legal
top-1 flips** across a small clock grid vs a 600s/600s baseline.

Examples::

    cd krasnal && uv run python scripts/diagnostics/clock_io_probe.py \\
        --artifact-dir artifacts/pretrain/RUN
    cd krasnal && uv run python scripts/diagnostics/clock_io_probe.py --mock
    cd krasnal && uv run python scripts/diagnostics/clock_io_probe.py \\
        --artifact-dir artifacts/pretrain/RUN --eval-rows 64 --eval-parquet data/other.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl
import torch
import torch.nn.functional as F

from krasnal.config import CLOCK_IGNORE_ID, EVAL_DATASET_PATH, MOVE_VOCAB_PATH, GPTConfig
from krasnal.inference import Game, InferenceSession, load_model
from krasnal.model import GPT
from krasnal.tokens import (
    ELO_ABOVE_2200_ID,
    GAME_START_ID,
    SPECIAL_TOKENS,
    TC_TOKENS,
    TC_UNKNOWN_ID,
    WHITE_WON_ID,
    get_move_clock_pairs,
    get_moves_only,
    get_vocab_size,
    legal_token_ids,
    load_move_vocab,
    to_uci,
)
from krasnal.uci_engine.go_params import GoParams
from krasnal.utils import (
    gpt_config_from_artifact_dict,
    read_model_config_json,
    resolve_runtime_device,
)

_TC_IDS = frozenset(TC_TOKENS.values())
_SPECIAL_IDS = frozenset(SPECIAL_TOKENS.values())
_CLOCK_GRID = (300, 60, 10, 5, 1)


def _pick_device(name: str) -> torch.device:
    if name == "auto":
        return resolve_runtime_device()
    if name in ("cpu", "cuda", "mps"):
        return torch.device(name)
    print("device must be auto|cpu|cuda|mps", file=sys.stderr)
    sys.exit(1)


def _require_inference_artifact(artifact_dir: Path) -> None:
    for rel in ("config.json", "model.pt", "move_vocab.json"):
        p = artifact_dir / rel
        if not p.is_file():
            print(f"Missing inference file: {p}", file=sys.stderr)
            sys.exit(1)


def _legal_simplex(session: InferenceSession) -> tuple[list[int], torch.Tensor]:
    legal_ids = legal_token_ids(session.game.board)
    if not legal_ids:
        raise RuntimeError("No legal moves in current position.")
    logits = session.get_legal_logits()
    probs = F.softmax(logits, dim=-1)
    p = probs[legal_ids].to(torch.float64)
    p = p / p.sum()
    return legal_ids, p


def _tv(p: torch.Tensor, q: torch.Tensor) -> float:
    return float(0.5 * torch.abs(p - q).sum())


def _set_leaf_clocks(session: InferenceSession, w_s: int, b_s: int) -> None:
    session.prepare_go_clocks(GoParams(wtime_ms=int(w_s) * 1000, btime_ms=int(b_s) * 1000))


def _install_move_vocab(*, artifact_dir: Path | None, raw_config: dict | None) -> None:
    if artifact_dir is not None:
        vocab_path = artifact_dir / "move_vocab.json"
        load_move_vocab(
            vocab_path,
            piece_aware_moves=bool(raw_config.get("piece_aware_moves", False)),
            side_prefixed_moves=bool(raw_config.get("side_prefixed_moves", True)),
        )
    else:
        load_move_vocab(
            MOVE_VOCAB_PATH,
            piece_aware_moves=True,
            side_prefixed_moves=True,
        )


def _load_model_and_config(
    *,
    artifact_dir: Path | None,
    mock: bool,
    device: torch.device,
) -> tuple[GPT, GPTConfig]:
    if mock:
        _install_move_vocab(artifact_dir=None, raw_config=None)
        cfg = GPTConfig(
            block_size=256,
            vocab_size=get_vocab_size(),
            n_layer=4,
            n_head=4,
            n_embd=256,
            use_time_conditioning=True,
            time_conditioning_hidden=64,
        )
        model = GPT(cfg).to(device).eval()
        return model, cfg

    assert artifact_dir is not None
    _require_inference_artifact(artifact_dir)
    raw = read_model_config_json(artifact_dir / "config.json")
    _install_move_vocab(artifact_dir=artifact_dir, raw_config=raw)
    cfg = gpt_config_from_artifact_dict(raw)
    if not cfg.use_time_conditioning:
        print(
            "This checkpoint has use_time_conditioning=False; "
            "the I/O clock probe is not applicable.",
            file=sys.stderr,
        )
        sys.exit(2)
    model = load_model(str(artifact_dir / "model.pt"), device, cfg)
    return model, cfg


def _fresh_session(model: GPT, device: torch.device) -> InferenceSession:
    game = Game(
        white_elo_token=ELO_ABOVE_2200_ID,
        black_elo_token=ELO_ABOVE_2200_ID,
        time_control_token=TC_UNKNOWN_ID,
        target_outcome_token=WHITE_WON_ID,
    )
    return InferenceSession(model, device, game=game)


def _eval_first_move_index(token_ids: list[int]) -> int:
    """Index of first move token (after prefix), mirroring ``get_moves_only`` think skipping."""

    for i, t in enumerate(token_ids):
        if t not in _SPECIAL_IDS:
            return i
    return len(token_ids)


def _game_from_eval_prefix(prefix: list[int]) -> Game:
    if not prefix or prefix[0] != GAME_START_ID:
        raise ValueError("eval row prefix must start with GAME_START_ID")
    i = 1
    if i < len(prefix) and prefix[i] in _TC_IDS:
        tc_token = int(prefix[i])
        i += 1
    else:
        tc_token = TC_UNKNOWN_ID
    if i + 2 >= len(prefix):
        raise ValueError(f"eval prefix too short ({len(prefix)} tokens): {prefix!r}")
    outcome = int(prefix[i])
    white_elo = int(prefix[i + 1])
    black_elo = int(prefix[i + 2])
    return Game(
        white_elo_token=white_elo,
        black_elo_token=black_elo,
        time_control_token=tc_token,
        target_outcome_token=outcome,
    )


def _replay_eval_prefix(
    session: InferenceSession,
    token_ids: list[int],
    active_clock_ids: list[int] | None,
    opponent_clock_ids: list[int] | None,
    *,
    ply: int,
) -> None:
    """Rebuild ``Game`` from row metadata and apply the first ``ply`` moves."""
    p0 = _eval_first_move_index(token_ids)
    prefix = list(token_ids[:p0])
    game = _game_from_eval_prefix(prefix)
    session.new_game(game)

    moves = get_moves_only(token_ids)
    if not moves:
        raise ValueError("no move tokens in row")
    pairs = get_move_clock_pairs(token_ids, active_clock_ids, opponent_clock_ids)
    use_pairs = pairs is not None and len(pairs) == len(moves)
    ply_use = min(max(1, ply), len(moves))
    for j in range(ply_use):
        uci = to_uci(moves[j])
        if uci is None:
            raise ValueError(f"to_uci returned None for move index {j}, id={moves[j]}")
        if use_pairs:
            ca, co = pairs[j]
            ca_n = None if ca == CLOCK_IGNORE_ID else int(ca)
            co_n = None if co == CLOCK_IGNORE_ID else int(co)
            session.feed_uci(uci, clock_active=ca_n, clock_opponent=co_n)
        else:
            session.feed_uci(uci)


def _probe_clock_grid(
    session: InferenceSession,
    *,
    baseline_w: int = 600,
    baseline_b: int = 600,
) -> tuple[float, tuple[int, int], int, int, int]:
    """TV/arg vs baseline; grid top-1 flip count; flip at max-TV cell (0/1)."""
    _set_leaf_clocks(session, baseline_w, baseline_b)
    legal_ids, p0 = _legal_simplex(session)
    baseline_tid = int(legal_ids[int(torch.argmax(p0))])

    max_tv = 0.0
    arg_wb = (baseline_w, baseline_b)
    flip_at_max_tv = 0
    grid_flip_count = 0

    for w_s in _CLOCK_GRID:
        for b_s in _CLOCK_GRID:
            _set_leaf_clocks(session, w_s, b_s)
            ids2, p = _legal_simplex(session)
            if p.shape != p0.shape or ids2 != legal_ids:
                continue
            tv = _tv(p0, p)
            top_tid = int(legal_ids[int(torch.argmax(p))])
            if top_tid != baseline_tid:
                grid_flip_count += 1
            if tv > max_tv:
                max_tv = tv
                arg_wb = (w_s, b_s)
                flip_at_max_tv = 1 if top_tid != baseline_tid else 0

    return max_tv, arg_wb, baseline_tid, grid_flip_count, flip_at_max_tv


def _read_eval_sample(path: Path, n: int, seed: int) -> pl.DataFrame:
    if not path.is_file():
        print(f"Parquet not found: {path}", file=sys.stderr)
        sys.exit(1)
    schema = pl.scan_parquet(path).collect_schema()
    cols = ["token_ids"]
    for c in ("active_clock_ids", "opponent_clock_ids"):
        if c in schema:
            cols.append(c)
    df = pl.read_parquet(path, columns=cols)
    if len(df) == 0:
        print("Parquet has zero rows.", file=sys.stderr)
        sys.exit(1)
    return df.sample(n=min(int(n), len(df)), seed=seed, shuffle=True)


def _run_eval_parquet_probe(
    model: GPT,
    device: torch.device,
    path: Path,
    *,
    rows: int,
    seed: int,
    ply_frac: float,
) -> None:
    df = _read_eval_sample(path, rows, seed)
    n_grid = len(_CLOCK_GRID) ** 2
    print(
        f"eval rows={len(df)} from {path} (seed={seed}, ply_frac={ply_frac}) "
        f"device={device} use_time_conditioning={model.config.use_time_conditioning} "
        f"clock_grid={list(_CLOCK_GRID)} ({n_grid} pairs vs 600s baseline; baseline not on grid)",
    )

    max_tvs: list[float] = []
    grid_flips: list[int] = []
    worst_flips: list[int] = []

    hdr = "row  ply  moves legal    max_tv      arg_wb  gflips wst  top1_uci  board_fen"
    print(f"\n{hdr}")
    print("-" * 100)
    for i, row in enumerate(df.iter_rows(named=True)):
        token_ids = [int(x) for x in row["token_ids"]]
        act = [int(x) for x in row["active_clock_ids"]] if "active_clock_ids" in row else None
        opp = [int(x) for x in row["opponent_clock_ids"]] if "opponent_clock_ids" in row else None
        moves = get_moves_only(token_ids)
        if not moves:
            print(f"{i:4d} skip (no moves)")
            continue
        ply = max(1, min(len(moves), int(len(moves) * float(ply_frac))))
        session = _fresh_session(model, device)
        try:
            _replay_eval_prefix(session, token_ids, act, opp, ply=ply)
        except (ValueError, RuntimeError) as e:
            print(f"{i:4d} skip ({e})")
            continue
        max_tv, arg, baseline_tid, gflips, wflip = _probe_clock_grid(session)
        max_tvs.append(max_tv)
        grid_flips.append(gflips)
        worst_flips.append(wflip)
        fen0 = session.game.board.fen().split()[0]
        n_legal = len(legal_token_ids(session.game.board))
        top1_uci = to_uci(baseline_tid) or "?"
        print(
            f"{i:4d} {ply:4d} {len(moves):6d} {n_legal:5d} {max_tv:9.5f} {arg!s:>12}  "
            f"{gflips:5d} {wflip:3d}  {top1_uci:8s}  {fen0}",
        )

    if max_tvs:
        t = torch.tensor(max_tvs, dtype=torch.float64)
        gf = torch.tensor(grid_flips, dtype=torch.float64)
        wf = torch.tensor(worst_flips, dtype=torch.float64)
        pct_any_grid = 100.0 * float((gf > 0).float().mean())
        pct_worst = 100.0 * float(wf.mean())
        print(
            f"\nAggregated max_tv: mean={float(t.mean()):.5f} "
            f"p90={float(t.quantile(0.9)):.5f} max={float(t.max()):.5f} (n={len(max_tvs)})",
        )
        print(
            f"Legal top-1 flips across {n_grid} clock pairs vs (600s,600s): "
            f"mean_flips={float(gf.mean()):.2f} p90_flips={float(gf.quantile(0.9)):.0f} "
            f"pct_rows_any_flip={pct_any_grid:.1f}% "
            f"pct_rows_flip_at_max_tv_pair={pct_worst:.1f}%",
        )
        if float(t.max()) < 0.01 and pct_worst < 1.0:
            print(
                "\nInterpretation: leaf clocks barely move the legal distribution and almost "
                "never change the legal argmax on this grid.",
            )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Training artifact directory (config.json + model.pt + move_vocab.json).",
    )
    p.add_argument(
        "--mock",
        action="store_true",
        help="Random-init small GPT with time conditioning (no checkpoint).",
    )
    p.add_argument(
        "--eval-parquet",
        type=Path,
        default=EVAL_DATASET_PATH,
        help=f"Eval Parquet path (default: {EVAL_DATASET_PATH}).",
    )
    p.add_argument("--eval-rows", type=int, default=32, help="Rows to sample from eval Parquet.")
    p.add_argument("--eval-seed", type=int, default=0, help="RNG seed for Parquet row sampling.")
    p.add_argument(
        "--eval-ply-frac",
        type=float,
        default=0.75,
        help="Replay first floor(frac * num_moves) moves (at least 1).",
    )
    p.add_argument("--device", type=str, default="auto", help="cpu | cuda | auto")
    args = p.parse_args()

    if args.mock and args.artifact_dir is not None:
        print("Use either --mock or --artifact-dir, not both.", file=sys.stderr)
        sys.exit(1)
    if not args.mock and args.artifact_dir is None:
        print("Pass --artifact-dir DIR or --mock.", file=sys.stderr)
        sys.exit(1)

    if args.eval_ply_frac <= 0 or args.eval_ply_frac > 1:
        print("--eval-ply-frac must be in (0, 1].", file=sys.stderr)
        sys.exit(1)

    device = _pick_device(args.device)
    if args.mock:
        print(
            "(--mock) Random weights: metrics are not diagnostic of training; "
            "use --artifact-dir for a real checkpoint.",
        )

    model, _cfg = _load_model_and_config(
        artifact_dir=args.artifact_dir,
        mock=args.mock,
        device=device,
    )

    _run_eval_parquet_probe(
        model,
        device,
        args.eval_parquet,
        rows=args.eval_rows,
        seed=args.eval_seed,
        ply_frac=args.eval_ply_frac,
    )


if __name__ == "__main__":
    main()
