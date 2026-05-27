"""QA probe evaluation: is_check and what_is_on inference passes."""

from __future__ import annotations

from typing import Any

import bulletchess
import plotly.graph_objects as go
import torch

import wandb
from krasnal.config import CLOCK_IGNORE_ID
from krasnal.eval.what_is_on_baseline import WhatIsOnBaselineCounts
from krasnal.inference import StatelessBatchInferenceSession
from krasnal.tokens import (
    COLORED_PIECE_TOKENS,
    EMPTY_ID,
    IS_CHECK_ID,
    NO_CHECK_ID,
    YES_CHECK_ID,
    whats_on_probe_labels,
)

from .metrics.context import EvalContext

_WHAT_IS_ON_LABEL_IDS: tuple[int, ...] = (EMPTY_ID, *sorted(COLORED_PIECE_TOKENS.values()))


def probe_clock_sequences(ctx: EvalContext, probe_len: int) -> tuple[list[int], list[int]]:
    """Build clock sequences for a probe: base + actual_token + probe_token(s).

    ``probe_len`` is the number of tokens added after ``ctx.sequence``
    (typically 2: actual_token + probe marker).
    """
    base_active = ctx.active_clock_sequence or []
    base_opponent = ctx.opponent_clock_sequence or []
    tail_active = [
        CLOCK_IGNORE_ID if ctx.active_clock_seconds is None else ctx.active_clock_seconds
    ]
    tail_opponent = [
        CLOCK_IGNORE_ID if ctx.opponent_clock_seconds is None else ctx.opponent_clock_seconds
    ]
    extra_active = [CLOCK_IGNORE_ID] * (probe_len - 1)
    extra_opponent = [CLOCK_IGNORE_ID] * (probe_len - 1)
    return (
        [*base_active, *tail_active, *extra_active],
        [*base_opponent, *tail_opponent, *extra_opponent],
    )


def compute_binary_f1_metrics(
    *,
    tp: int,
    fp: int,
    fn: int,
) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    check_f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "qa/is_check/precision": precision,
        "qa/is_check/recall": recall,
        "qa/is_check/f1": check_f1,
    }


def build_what_is_on_heatmap(square_accs: dict[str, float]) -> wandb.Plotly:
    files = ["a", "b", "c", "d", "e", "f", "g", "h"]
    ranks = ["8", "7", "6", "5", "4", "3", "2", "1"]
    z = [[square_accs.get(f"{file}{rank}", 0.0) for file in files] for rank in ranks]
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=files,
            y=ranks,
            colorscale="Viridis",
            reversescale=False,
            colorbar={"title": "Accuracy"},
            hovertemplate="file=%{x}<br>rank=%{y}<br>acc=%{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="what_is_on accuracy per square",
        xaxis_title="File",
        yaxis_title="Rank",
    )
    return wandb.Plotly(fig)


def evaluate_is_check_probe_counts(
    contexts: list[EvalContext],
    model: torch.nn.Module,
    device: torch.device,
    *,
    batch_size: int = 64,
) -> dict[str, int]:
    """Run a batched is_check probe and return additive confusion counts."""
    probe_sequences: list[list[int]] = []
    probe_active: list[list[int]] = []
    probe_opponent: list[list[int]] = []
    labels: list[int] = []
    block_size = model.config.block_size

    for ctx in contexts:
        if ctx.sequence is None or ctx.actual_token is None or ctx.gives_check is None:
            continue
        probe = [*ctx.sequence, ctx.actual_token, IS_CHECK_ID]
        if len(probe) > block_size:
            continue
        act_seq, opp_seq = probe_clock_sequences(ctx, probe_len=2)
        probe_sequences.append(probe)
        probe_active.append(act_seq)
        probe_opponent.append(opp_seq)
        labels.append(1 if ctx.gives_check else 0)

    if not probe_sequences:
        return {"tp": 0, "fp": 0, "tn": 0, "fn": 0}

    batch_session = StatelessBatchInferenceSession(model, device)
    logits = batch_session.get_raw_logits_batch(
        probe_sequences,
        active_clock_sequences=probe_active,
        opponent_clock_sequences=probe_opponent,
        batch_size=batch_size,
    )
    preds: list[int] = []
    for logit in logits:
        preds.append(1 if logit[YES_CHECK_ID] >= logit[NO_CHECK_ID] else 0)

    tp = sum(1 for pred, label in zip(preds, labels, strict=True) if pred == 1 and label == 1)
    fp = sum(1 for pred, label in zip(preds, labels, strict=True) if pred == 1 and label == 0)
    tn = sum(1 for pred, label in zip(preds, labels, strict=True) if pred == 0 and label == 0)
    fn = sum(1 for pred, label in zip(preds, labels, strict=True) if pred == 0 and label == 1)

    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def empty_what_is_on_probe_stats() -> dict[str, Any]:
    return {
        "correct": 0,
        "total": 0,
        "baseline_correct": 0,
        "square_correct": {},
        "square_count": {},
        "square_baseline_correct": {},
    }


def evaluate_what_is_on_probe_stats(
    contexts: list[EvalContext],
    model: torch.nn.Module,
    device: torch.device,
    eval_seed: int,
    *,
    baseline: WhatIsOnBaselineCounts | None = None,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Run a batched what_is_on probe and return additive accuracy counts."""
    probe_sequences: list[list[int]] = []
    probe_active: list[list[int]] = []
    probe_opponent: list[list[int]] = []
    labels: list[int] = []
    sq_strs: list[str] = []
    plies: list[int] = []
    block_size = model.config.block_size

    for ctx in contexts:
        if ctx.sequence is None or ctx.actual_token is None or ctx.post_move_fen is None:
            continue

        game_key = ctx.what_is_on_game_key or ""
        ply = ctx.what_is_on_ply if ctx.what_is_on_ply is not None else 0
        board = bulletchess.Board.from_fen(ctx.post_move_fen)
        sq_str, whats_on_token_id, ans_id = whats_on_probe_labels(
            board,
            post_move_fen=ctx.post_move_fen,
            game_key=game_key,
            ply=ply,
            seed=eval_seed,
        )

        probe = [*ctx.sequence, ctx.actual_token, whats_on_token_id]
        if len(probe) > block_size:
            continue

        act_seq, opp_seq = probe_clock_sequences(ctx, probe_len=2)
        probe_sequences.append(probe)
        probe_active.append(act_seq)
        probe_opponent.append(opp_seq)
        labels.append(ans_id)
        sq_strs.append(sq_str)
        plies.append(int(ply))

    if not probe_sequences:
        return empty_what_is_on_probe_stats()

    batch_session = StatelessBatchInferenceSession(model, device)
    logits = batch_session.get_raw_logits_batch(
        probe_sequences,
        active_clock_sequences=probe_active,
        opponent_clock_sequences=probe_opponent,
        batch_size=batch_size,
    )
    preds: list[int] = []

    for logit in logits:
        pred_id = max(_WHAT_IS_ON_LABEL_IDS, key=lambda pid: float(logit[pid]))
        preds.append(pred_id)

    correct = sum(1 for pred, label in zip(preds, labels, strict=True) if pred == label)
    total = len(labels)

    baseline_preds: list[int] | None = None
    baseline_correct = 0
    if baseline is not None:
        baseline_preds = [baseline.predict(sq, pl) for sq, pl in zip(sq_strs, plies, strict=True)]
        baseline_correct = sum(
            1 for bp, label in zip(baseline_preds, labels, strict=True) if bp == label
        )

    stats = empty_what_is_on_probe_stats()
    stats["correct"] = correct
    stats["total"] = total
    stats["baseline_correct"] = baseline_correct
    for i, sq in enumerate(sq_strs):
        stats["square_count"][sq] = stats["square_count"].get(sq, 0) + 1
        if preds[i] == labels[i]:
            stats["square_correct"][sq] = stats["square_correct"].get(sq, 0) + 1
        if baseline_preds is not None and baseline_preds[i] == labels[i]:
            key = "square_baseline_correct"
            stats[key][sq] = stats[key].get(sq, 0) + 1
    return stats


def finalize_what_is_on_probe_stats(
    stats: dict[str, Any],
    *,
    include_per_square: bool = False,
    include_baseline: bool = False,
) -> dict[str, Any]:
    total = stats["total"]
    metrics: dict[str, Any] = {
        "qa/what_is_on/acc": stats["correct"] / total if total > 0 else 0.0,
    }
    if include_baseline:
        metrics["qa/what_is_on/acc_baseline"] = (
            stats["baseline_correct"] / total if total > 0 else 0.0
        )
    if not include_per_square:
        return metrics

    square_acc: dict[str, float] = {}
    square_acc_bl: dict[str, float] = {}
    for file_i in range(8):
        for rank_i in range(8):
            sq = f"{chr(97 + file_i)}{1 + rank_i}"
            count = stats["square_count"].get(sq, 0)
            square_acc[sq] = stats["square_correct"].get(sq, 0) / count if count else 0.0
            square_acc_bl[sq] = (
                stats["square_baseline_correct"].get(sq, 0) / count if count else 0.0
            )

    metrics["qa/what_is_on/acc_matrix"] = build_what_is_on_heatmap(square_acc)
    if include_baseline:
        metrics["qa/what_is_on/acc_matrix_baseline"] = build_what_is_on_heatmap(square_acc_bl)
    return metrics
