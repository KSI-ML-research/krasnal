import random
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import torch
from loguru import logger
from omegaconf import OmegaConf

import wandb
from krasnal.config import CLOCK_IGNORE_ID
from krasnal.dataset import ChessDataset
from krasnal.eval.parsers import parse_row_to_game_tokens
from krasnal.eval.replayer import replay_games
from krasnal.eval.what_is_on_baseline import (
    WhatIsOnBaselineCounts,
)
from krasnal.inference import StatelessBatchInferenceSession
from krasnal.sampling import whats_on_square_index
from krasnal.tokens import (
    COLORED_PIECE_TOKENS,
    EMPTY_ID,
    IS_CHECK_ID,
    NO_CHECK_ID,
    WHATS_ON_SQUARE,
    YES_CHECK_ID,
    to_uci,
)
from krasnal.utils import set_seed

from .metrics import METRIC_REGISTRY
from .metrics.context import EvalContext
from .metrics.core import AccuracyCore, CoreMetric, IllegalMassCore, MRRCore, Top1LegalCore
from .metrics.filtered import WhenLowTimeMetric

_WHAT_IS_ON_LABEL_IDS: tuple[int, ...] = (EMPTY_ID, *sorted(COLORED_PIECE_TOKENS.values()))

LOW_TIME_METRICS: dict[str, type[CoreMetric]] = {
    "acc_when_low_time": AccuracyCore,
    "top1_legal_when_low_time": Top1LegalCore,
    "illegal_mass_when_low_time": IllegalMassCore,
    "mrr_when_low_time": MRRCore,
}


class ChessEvaluator:
    """Evaluates chess model on legal move metrics using batched inference."""

    def __init__(
        self,
        metrics: list[str] | None = None,
        seed: int | None = None,
        qa_config: dict[str, Any] | None = None,
        low_time_seconds: int = 30,
    ):
        if metrics is None:
            raise ValueError("ChessEvaluator requires an explicit metrics list")

        self.requested_metrics = metrics
        self.seed = seed
        self.low_time_seconds = int(low_time_seconds)

        qa_cfg = qa_config or {}
        check_cfg = qa_cfg.get("check", {})
        self.enable_qa_check_metrics = bool(check_cfg.get("enabled", True))
        self.enable_qa_check_confusion_matrix_metrics = bool(
            check_cfg.get("confusion_matrix", False)
        )

        what_is_on_cfg = qa_cfg.get("what_is_on", {})
        self.enable_what_is_on_probe_metrics = bool(what_is_on_cfg.get("enabled", True))
        self.enable_what_is_on_accuracy_per_square_metrics = bool(
            what_is_on_cfg.get("accuracy_per_square", False)
        )
        self.what_is_on_baseline: WhatIsOnBaselineCounts | None = None
        raw_baseline = what_is_on_cfg.get("baseline_counts_path")
        if raw_baseline:
            bp = Path(raw_baseline)
            if bp.is_file():
                self.what_is_on_baseline = WhatIsOnBaselineCounts.load(bp)
            else:
                logger.warning("what_is_on baseline_counts_path is not a file: {}", bp)

        self.metrics = self._init_metrics()

    def _init_metrics(self) -> dict[str, Any]:
        metrics = {}
        for name in self.requested_metrics:
            if name in LOW_TIME_METRICS:
                metrics[name] = WhenLowTimeMetric(LOW_TIME_METRICS[name](), self.low_time_seconds)
                continue
            if name in METRIC_REGISTRY:
                metrics[name] = METRIC_REGISTRY[name]()
        return metrics

    def evaluate(
        self,
        model: torch.nn.Module,
        dataset: ChessDataset,
        num_games: int,
        device: torch.device,
        seed: int | None = None,
    ) -> dict[str, Any]:
        seed = seed if seed is not None else self.seed
        if seed is not None:
            set_seed(seed)

        self.metrics = self._init_metrics()

        block_size = model.config.block_size
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        indices = indices[:num_games]

        game_tokens_list = []
        for idx in indices:
            row = dataset[idx]
            game_tokens = parse_row_to_game_tokens(row)
            if game_tokens is not None:
                game_tokens_list.append(game_tokens)

        contexts = replay_games(game_tokens_list, block_size)

        if not contexts:
            return {name: 0.0 for name in self.metrics}

        eval_seed = seed if seed is not None else (self.seed if self.seed is not None else 0)
        return self._infer_and_aggregate(contexts, model, device, eval_seed)

    def _infer_and_aggregate(
        self,
        contexts: list[EvalContext],
        model: torch.nn.Module,
        device: torch.device,
        eval_seed: int,
    ) -> dict[str, float]:
        all_positions = [ctx.sequence for ctx in contexts]
        all_active_clock = [ctx.active_clock_sequence for ctx in contexts]
        all_opponent_clock = [ctx.opponent_clock_sequence for ctx in contexts]
        batch_session = StatelessBatchInferenceSession(model, device)
        probs = batch_session.get_raw_probs_batch(
            all_positions,
            active_clock_sequences=all_active_clock,
            opponent_clock_sequences=all_opponent_clock,
        )

        for ctx, prob in zip(contexts, probs, strict=True):
            ctx.probs = prob

        results: dict[str, list[float]] = {
            name: [] for name, m in self.metrics.items() if not hasattr(m, "finalize")
        }

        for ctx in contexts:
            if ctx.legal_ids:
                self._compute_top1_fen(ctx)

            for metric in self.metrics.values():
                result = metric.compute(ctx)
                for k, v in result.items():
                    results[k].append(v)

        final = self._aggregate_results(results)
        if self.enable_qa_check_metrics:
            final.update(self._evaluate_is_check_probe(contexts, model, device))
        if self.enable_what_is_on_probe_metrics:
            final.update(self._evaluate_what_is_on_probe(contexts, model, device, eval_seed))
        return final

    def _evaluate_is_check_probe(
        self,
        contexts: list[EvalContext],
        model: torch.nn.Module,
        device: torch.device,
    ) -> dict[str, float]:
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
            act_seq, opp_seq = self._probe_clock_sequences(ctx, probe_len=2)
            probe_sequences.append(probe)
            probe_active.append(act_seq)
            probe_opponent.append(opp_seq)
            labels.append(1 if ctx.gives_check else 0)

        if not probe_sequences:
            return {
                "qa/is_check/precision": 0.0,
                "qa/is_check/recall": 0.0,
                "qa/is_check/f1": 0.0,
            }

        batch_session = StatelessBatchInferenceSession(model, device)
        probs = batch_session.get_raw_probs_batch(
            probe_sequences,
            active_clock_sequences=probe_active,
            opponent_clock_sequences=probe_opponent,
        )
        preds: list[int] = []
        for prob in probs:
            preds.append(1 if prob[YES_CHECK_ID] >= prob[NO_CHECK_ID] else 0)

        tp = sum(1 for pred, label in zip(preds, labels, strict=True) if pred == 1 and label == 1)
        fp = sum(1 for pred, label in zip(preds, labels, strict=True) if pred == 1 and label == 0)
        tn = sum(1 for pred, label in zip(preds, labels, strict=True) if pred == 0 and label == 0)
        fn = sum(1 for pred, label in zip(preds, labels, strict=True) if pred == 0 and label == 1)

        metrics = self._compute_binary_f1_metrics(tp=tp, fp=fp, fn=fn)
        if self.enable_qa_check_confusion_matrix_metrics:
            metrics["qa/is_check/confusion_matrix/tp"] = float(tp)
            metrics["qa/is_check/confusion_matrix/fp"] = float(fp)
            metrics["qa/is_check/confusion_matrix/tn"] = float(tn)
            metrics["qa/is_check/confusion_matrix/fn"] = float(fn)
        return metrics

    def _evaluate_what_is_on_probe(
        self,
        contexts: list[EvalContext],
        model: torch.nn.Module,
        device: torch.device,
        eval_seed: int,
    ) -> dict[str, Any]:
        import bulletchess

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
            sq_idx = whats_on_square_index(
                post_move_fen=ctx.post_move_fen,
                game_key=game_key,
                ply=ply,
                seed=eval_seed,
            )

            file_char = chr(97 + (sq_idx % 8))
            rank_char = str(1 + (sq_idx // 8))
            sq_str = f"{file_char}{rank_char}"
            whats_on_token_id = WHATS_ON_SQUARE[f"<whats_on_{sq_str}>"]

            piece = board[bulletchess.Square.from_str(sq_str)]

            if piece is None:
                ans_id = EMPTY_ID
            else:
                color_str = "w" if str(piece.color) == "White" else "b"
                piece_str = str(piece.piece_type).lower()
                ans_id = COLORED_PIECE_TOKENS[f"<{color_str}:{piece_str}>"]

            probe = [*ctx.sequence, ctx.actual_token, whats_on_token_id]
            if len(probe) > block_size:
                continue

            act_seq, opp_seq = self._probe_clock_sequences(ctx, probe_len=2)
            probe_sequences.append(probe)
            probe_active.append(act_seq)
            probe_opponent.append(opp_seq)
            labels.append(ans_id)
            sq_strs.append(sq_str)
            plies.append(int(ply))

        metrics: dict[str, Any] = {
            "qa/what_is_on/acc": 0.0,
        }
        if self.enable_what_is_on_accuracy_per_square_metrics:
            empty_acc = {f"{f}{r}": 0.0 for f in "abcdefgh" for r in range(1, 9)}
            metrics["qa/what_is_on/acc_matrix"] = self._build_what_is_on_heatmap(empty_acc)

        if not probe_sequences:
            return metrics

        batch_session = StatelessBatchInferenceSession(model, device)
        probs = batch_session.get_raw_probs_batch(
            probe_sequences,
            active_clock_sequences=probe_active,
            opponent_clock_sequences=probe_opponent,
        )
        preds: list[int] = []

        for prob in probs:
            pred_id = max(_WHAT_IS_ON_LABEL_IDS, key=lambda pid: float(prob[pid]))
            preds.append(pred_id)

        correct = sum(1 for pred, label in zip(preds, labels, strict=True) if pred == label)
        total = len(labels)

        metrics["qa/what_is_on/acc"] = correct / total if total > 0 else 0.0

        baseline_preds: list[int] | None = None
        if self.what_is_on_baseline is not None:
            baseline_preds = [
                self.what_is_on_baseline.predict(sq, pl)
                for sq, pl in zip(sq_strs, plies, strict=True)
            ]
            baseline_correct = sum(
                1 for bp, label in zip(baseline_preds, labels, strict=True) if bp == label
            )
            metrics["qa/what_is_on/acc_baseline"] = baseline_correct / total if total > 0 else 0.0

        if self.enable_what_is_on_accuracy_per_square_metrics:
            square_acc: dict[str, float] = {}
            for file_i in range(8):
                for rank_i in range(8):
                    sq = f"{chr(97 + file_i)}{1 + rank_i}"
                    sq_indices = [i for i, s in enumerate(sq_strs) if s == sq]
                    if not sq_indices:
                        square_acc[sq] = 0.0
                        continue
                    square_acc[sq] = sum(1 for i in sq_indices if preds[i] == labels[i]) / len(
                        sq_indices
                    )

            metrics["qa/what_is_on/acc_matrix"] = self._build_what_is_on_heatmap(square_acc)

            if baseline_preds is not None:
                square_acc_bl: dict[str, float] = {}
                for file_i in range(8):
                    for rank_i in range(8):
                        sq = f"{chr(97 + file_i)}{1 + rank_i}"
                        sq_indices = [i for i, s in enumerate(sq_strs) if s == sq]
                        if not sq_indices:
                            square_acc_bl[sq] = 0.0
                            continue
                        square_acc_bl[sq] = sum(
                            1 for i in sq_indices if baseline_preds[i] == labels[i]
                        ) / len(sq_indices)
                metrics["qa/what_is_on/acc_matrix_baseline"] = self._build_what_is_on_heatmap(
                    square_acc_bl
                )

        return metrics

    @staticmethod
    def _probe_clock_sequences(ctx: EvalContext, probe_len: int) -> tuple[list[int], list[int]]:
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

    @staticmethod
    def _build_what_is_on_heatmap(square_accs: dict[str, float]) -> wandb.Plotly:
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

    @staticmethod
    def _compute_binary_f1_metrics(
        *,
        tp: int,
        fp: int,
        fn: int,
    ) -> dict[str, float]:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        check_f1 = (
            2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        )

        return {
            "qa/is_check/precision": precision,
            "qa/is_check/recall": recall,
            "qa/is_check/f1": check_f1,
        }

    def _compute_top1_fen(self, ctx: EvalContext) -> None:
        import bulletchess

        if ctx.probs is None or ctx.fen is None or ctx.legal_ids is None:
            return

        legal_probs = [(tid, ctx.probs[tid].item()) for tid in ctx.legal_ids]
        top1_token = max(legal_probs, key=lambda x: x[1])[0]
        uci_move_raw = to_uci(top1_token)
        if uci_move_raw:
            ctx.top1_move_uci = uci_move_raw
            try:
                board = bulletchess.Board.from_fen(ctx.fen)
                top1_move = bulletchess.Move.from_uci(uci_move_raw)
                board.apply(top1_move)
                ctx.top1_fen = board.fen()
            except Exception as e:
                logger.warning(
                    f"Failed to compute top1_fen: {uci_move_raw} on {ctx.fen[:30]}...: {e}"
                )

    def _aggregate_results(self, results: dict[str, list[float]]) -> dict[str, float]:
        final_results: dict[str, float] = {}
        for k, v in results.items():
            final_results[k] = sum(v) / len(v) if v else 0.0
        for metric in self.metrics.values():
            if hasattr(metric, "finalize"):
                for k, v in metric.finalize().items():
                    final_results[k] = v
        return final_results


def chess_evaluator_from_config(cfg: Any, *, metrics: list[str]) -> ChessEvaluator:
    return ChessEvaluator(
        metrics=metrics,
        seed=cfg.seed,
        qa_config=OmegaConf.to_container(cfg.eval.qa, resolve=True),
        low_time_seconds=int(cfg.eval.get("low_time_seconds", 30)),
    )
