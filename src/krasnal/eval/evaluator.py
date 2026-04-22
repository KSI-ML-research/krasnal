import random
from typing import Any

import torch
from loguru import logger

from krasnal.dataset import ChessDataset
from krasnal.eval.cot import extract_think_tokens, is_valid_cot_sequence, parse_cot_sample
from krasnal.eval.parsers import parse_game_tokens
from krasnal.eval.replayer import replay_games
from krasnal.inference import InferenceSession, StatelessBatchInferenceSession
from krasnal.tokens import (
    GAME_END_ID,
    IS_CHECK_ID,
    NO_CHECK_ID,
    THINK_END_ID,
    YES_CHECK_ID,
    get_moves_only,
    to_uci,
)
from krasnal.utils import set_seed

from .metrics import COT_METRICS, DEFAULT_METRICS, METRIC_REGISTRY
from .metrics.context import EvalContext
from .stockfish import StockfishClient


class ChessEvaluator:
    """Evaluates chess model on legal move metrics using batched inference."""

    DEFAULT_METRICS = DEFAULT_METRICS

    def __init__(
        self,
        metrics: list[str] | None = None,
        cot: bool = False,
        cot_max_tokens: int = 128,
        seed: int | None = None,
        stockfish: StockfishClient | None = None,
        acpl_sample_size: int = 100,
    ):
        self.requested_metrics = metrics or (COT_METRICS if cot else self.DEFAULT_METRICS)
        self.cot = cot
        self.cot_max_tokens = cot_max_tokens
        self.seed = seed
        self.stockfish = stockfish
        self.acpl_sample_size = acpl_sample_size
        self.metrics = self._init_metrics()

    def _init_metrics(self) -> dict[str, Any]:
        metrics = {}
        for name in self.requested_metrics:
            if name in METRIC_REGISTRY:
                if name in {"acpl", "stockfish_top1", "blunder_rate"}:
                    metrics[name] = METRIC_REGISTRY[name](
                        stockfish=self.stockfish, sample_size=self.acpl_sample_size
                    )
                else:
                    metrics[name] = METRIC_REGISTRY[name]()
        return metrics

    def evaluate(
        self,
        model: torch.nn.Module,
        dataset: ChessDataset,
        num_games: int,
        device: torch.device,
        seed: int | None = None,
        stockfish: StockfishClient | None = None,
    ) -> dict[str, Any]:
        seed = seed if seed is not None else self.seed
        if seed is not None:
            set_seed(seed)

        if stockfish is not None:
            self.stockfish = stockfish
            self.metrics = self._init_metrics()

        if self.cot:
            return self.evaluate_cot(model, dataset, num_games, device, seed)

        block_size = model.config.block_size
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        indices = indices[:num_games]

        game_tokens_list = []
        for idx in indices:
            token_ids = dataset[idx].tolist()
            game_tokens = parse_game_tokens(token_ids)
            if game_tokens is None:
                continue

            moves = get_moves_only(token_ids)
            game_tokens.move_tokens = moves
            game_tokens_list.append(game_tokens)

        contexts = replay_games(game_tokens_list, block_size)

        if not contexts:
            return {name: 0.0 for name in self.metrics}

        return self._infer_and_aggregate(contexts, model, device)

    def _infer_and_aggregate(
        self,
        contexts: list[EvalContext],
        model: torch.nn.Module,
        device: torch.device,
    ) -> dict[str, float]:
        all_positions = [ctx.sequence for ctx in contexts]
        batch_session = StatelessBatchInferenceSession(model, device)
        probs = batch_session.get_raw_probs_batch(all_positions)

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
        final.update(self._evaluate_is_check_probe(contexts, model, device))
        return final

    def _evaluate_is_check_probe(
        self,
        contexts: list[EvalContext],
        model: torch.nn.Module,
        device: torch.device,
    ) -> dict[str, float]:
        probe_sequences: list[list[int]] = []
        labels: list[int] = []
        block_size = model.config.block_size

        for ctx in contexts:
            if ctx.sequence is None or ctx.actual_token is None or ctx.gives_check is None:
                continue
            probe = [*ctx.sequence, ctx.actual_token, IS_CHECK_ID]
            if len(probe) > block_size:
                continue
            probe_sequences.append(probe)
            labels.append(1 if ctx.gives_check else 0)

        if not probe_sequences:
            return {
                "check_tp": 0.0,
                "check_fp": 0.0,
                "check_tn": 0.0,
                "check_fn": 0.0,
                "check_precision": 0.0,
                "check_recall": 0.0,
                "check_f1": 0.0,
                "check_macro_f1": 0.0,
                "check_f1_always_no": 0.0,
            }

        batch_session = StatelessBatchInferenceSession(model, device)
        probs = batch_session.get_raw_probs_batch(probe_sequences)
        preds: list[int] = []
        for prob in probs:
            preds.append(1 if prob[YES_CHECK_ID] >= prob[NO_CHECK_ID] else 0)

        tp = sum(1 for pred, label in zip(preds, labels, strict=True) if pred == 1 and label == 1)
        fp = sum(1 for pred, label in zip(preds, labels, strict=True) if pred == 1 and label == 0)
        tn = sum(1 for pred, label in zip(preds, labels, strict=True) if pred == 0 and label == 0)
        fn = sum(1 for pred, label in zip(preds, labels, strict=True) if pred == 0 and label == 1)

        metrics = self._compute_binary_f1_metrics(tp=tp, fp=fp, tn=tn, fn=fn)
        fn_always_no = sum(1 for label in labels if label == 1)
        tn_always_no = sum(1 for label in labels if label == 0)
        baseline = self._compute_binary_f1_metrics(tp=0, fp=0, tn=tn_always_no, fn=fn_always_no)
        metrics["check_f1_always_no"] = baseline["check_f1"]
        return metrics

    @staticmethod
    def _compute_binary_f1_metrics(
        *,
        tp: int,
        fp: int,
        tn: int,
        fn: int,
    ) -> dict[str, float]:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        check_f1 = (
            2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        )

        no_precision = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        no_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        no_f1 = (
            2.0 * no_precision * no_recall / (no_precision + no_recall)
            if (no_precision + no_recall) > 0
            else 0.0
        )

        return {
            "check_tp": float(tp),
            "check_fp": float(fp),
            "check_tn": float(tn),
            "check_fn": float(fn),
            "check_precision": precision,
            "check_recall": recall,
            "check_f1": check_f1,
            "check_macro_f1": (check_f1 + no_f1) / 2.0,
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
                final_results.update(metric.finalize())
        return final_results

    def evaluate_cot(
        self,
        model: torch.nn.Module,
        dataset: ChessDataset,
        num_games: int,
        device: torch.device,
        seed: int | None = None,
    ) -> dict[str, float]:
        """Evaluate model using chain-of-thought format."""
        seed = seed if seed is not None else self.seed
        if seed is not None:
            set_seed(seed)

        indices = list(range(len(dataset)))
        random.shuffle(indices)
        indices = indices[:num_games]

        results: dict[str, list[float]] = {name: [] for name in self.metrics}
        for idx in indices:
            sample = parse_cot_sample(dataset[idx].tolist())
            if sample is None:
                continue

            generated_tokens, post_think_probs = self._generate_cot_continuation(
                model=model,
                device=device,
                prompt_tokens=sample["prompt_tokens"],
            )
            full_sequence = [*sample["prompt_tokens"], *generated_tokens]
            context = EvalContext(
                cot_format_valid=is_valid_cot_sequence(full_sequence),
                cot_post_think_probs=post_think_probs,
                cot_post_think_actual_token=sample["post_think_actual_token"],
                cot_post_think_legal_ids=sample["post_think_legal_ids"],
                target_think_tokens=sample["target_think_tokens"],
                generated_think_tokens=extract_think_tokens(generated_tokens),
            )

            for metric in self.metrics.values():
                metric_result = metric.compute(context)
                for key, value in metric_result.items():
                    results[key].append(float(value))

        return {
            key: (sum(values) / len(values) if values else 0.0) for key, values in results.items()
        }

    def _generate_cot_continuation(
        self,
        *,
        model: torch.nn.Module,
        device: torch.device,
        prompt_tokens: list[int],
    ) -> tuple[list[int], torch.Tensor | None]:
        """Generate CoT continuation tokens."""
        session = InferenceSession(model, device, outcome_token=prompt_tokens[0])
        for token_id in prompt_tokens[1:]:
            session.feed_token(token_id)

        generated_tokens: list[int] = []
        post_think_probs = None
        seen_think_end = False

        for _ in range(self.cot_max_tokens):
            probs = session.get_raw_probs()
            token_id = int(torch.argmax(probs).item())
            generated_tokens.append(token_id)
            session.feed_token(token_id)
            if seen_think_end and post_think_probs is None:
                post_think_probs = probs.detach().cpu()
            if token_id == THINK_END_ID:
                seen_think_end = True
            if token_id == GAME_END_ID:
                break

        return generated_tokens, post_think_probs

    @staticmethod
    def _is_valid_cot_sequence(tokens: list[int]) -> bool:
        return is_valid_cot_sequence(tokens)

    @staticmethod
    def _extract_generated_think_tokens(tokens: list[int]) -> list[int]:
        return extract_think_tokens(tokens)
