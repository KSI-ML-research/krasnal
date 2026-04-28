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
    BISHOP_ID,
    GAME_END_ID,
    IS_CHECK_ID,
    KING_ID,
    KNIGHT_ID,
    NO_CHECK_ID,
    PAWN_ID,
    QUEEN_ID,
    ROOK_ID,
    THINK_END_ID,
    WHAT_PIECE_ID,
    YES_CHECK_ID,
    get_moves_only,
    to_uci,
)
from krasnal.utils import set_seed

from .metrics import METRIC_REGISTRY
from .metrics.context import EvalContext
from .stockfish import StockfishClient


class ChessEvaluator:
    """Evaluates chess model on legal move metrics using batched inference."""

    def __init__(
        self,
        metrics: list[str] | None = None,
        cot: bool = False,
        cot_max_tokens: int = 128,
        seed: int | None = None,
        stockfish: StockfishClient | None = None,
        acpl_sample_size: int = 100,
        enable_qa_check_metrics: bool = True,
        enable_piece_probe_metrics: bool = True,
        enable_piece_f1_breakdown_metrics: bool = False,
        enable_qa_check_confusion_matrix_metrics: bool = False,
        enable_piece_confusion_matrix_metrics: bool = False,
    ):
        if metrics is None:
            raise ValueError("ChessEvaluator requires an explicit metrics list")

        self.requested_metrics = metrics
        self.cot = cot
        self.cot_max_tokens = cot_max_tokens
        self.seed = seed
        self.stockfish = stockfish
        self.acpl_sample_size = acpl_sample_size
        self.enable_qa_check_metrics = enable_qa_check_metrics
        self.enable_piece_probe_metrics = enable_piece_probe_metrics
        self.enable_piece_f1_breakdown_metrics = enable_piece_f1_breakdown_metrics
        self.enable_qa_check_confusion_matrix_metrics = enable_qa_check_confusion_matrix_metrics
        self.enable_piece_confusion_matrix_metrics = enable_piece_confusion_matrix_metrics
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
        if self.enable_qa_check_metrics:
            final.update(self._evaluate_is_check_probe(contexts, model, device))
        if self.enable_piece_probe_metrics:
            final.update(self._evaluate_piece_probe(contexts, model, device))
        return final

    def _evaluate_piece_probe(
        self,
        contexts: list[EvalContext],
        model: torch.nn.Module,
        device: torch.device,
    ) -> dict[str, float]:
        piece_type_to_token = {
            1: PAWN_ID,
            2: KNIGHT_ID,
            3: BISHOP_ID,
            4: ROOK_ID,
            5: QUEEN_ID,
            6: KING_ID,
        }
        piece_ids = [PAWN_ID, KNIGHT_ID, BISHOP_ID, ROOK_ID, QUEEN_ID, KING_ID]
        piece_names = {
            PAWN_ID: "pawn",
            KNIGHT_ID: "knight",
            BISHOP_ID: "bishop",
            ROOK_ID: "rook",
            QUEEN_ID: "queen",
            KING_ID: "king",
        }

        probe_sequences: list[list[int]] = []
        labels: list[int] = []
        block_size = model.config.block_size

        for ctx in contexts:
            if ctx.sequence is None or ctx.actual_token is None or ctx.piece_type is None:
                continue
            true_piece_token = piece_type_to_token.get(ctx.piece_type)
            if true_piece_token is None:
                continue
            probe = [*ctx.sequence, ctx.actual_token, WHAT_PIECE_ID]
            if len(probe) > block_size:
                continue
            probe_sequences.append(probe)
            labels.append(true_piece_token)

        default_metrics: dict[str, float] = {
            "piece_acc": 0.0,
            "qa_piece_f1": 0.0,
        }
        if self.enable_piece_f1_breakdown_metrics:
            for piece_id in piece_ids:
                default_metrics[f"piece_f1_{piece_names[piece_id]}"] = 0.0
        if self.enable_piece_confusion_matrix_metrics:
            for true_id in piece_ids:
                for pred_id in piece_ids:
                    default_metrics[f"piece_cm_{piece_names[true_id]}_{piece_names[pred_id]}"] = 0.0

        if not probe_sequences:
            return default_metrics

        batch_session = StatelessBatchInferenceSession(model, device)
        probs = batch_session.get_raw_probs_batch(probe_sequences)

        preds: list[int] = []
        for prob in probs:
            pred_piece = max(piece_ids, key=lambda pid: float(prob[pid]))
            preds.append(pred_piece)

        confusion = {(true_id, pred_id): 0 for true_id in piece_ids for pred_id in piece_ids}
        for true_id, pred_id in zip(labels, preds, strict=True):
            confusion[(true_id, pred_id)] += 1

        total = len(labels)
        correct = sum(confusion[(piece_id, piece_id)] for piece_id in piece_ids)
        acc = correct / total if total > 0 else 0.0

        f1_sum = 0.0
        piece_f1_scores: dict[int, float] = {}
        for piece_id in piece_ids:
            tp = confusion[(piece_id, piece_id)]
            fp = sum(confusion[(other, piece_id)] for other in piece_ids if other != piece_id)
            fn = sum(confusion[(piece_id, other)] for other in piece_ids if other != piece_id)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            )
            piece_f1_scores[piece_id] = f1
            f1_sum += f1
        macro_f1 = f1_sum / len(piece_ids)

        metrics: dict[str, float] = {
            "piece_acc": acc,
            "qa_piece_f1": macro_f1,
        }
        if self.enable_piece_f1_breakdown_metrics:
            for piece_id in piece_ids:
                metrics[f"piece_f1_{piece_names[piece_id]}"] = piece_f1_scores[piece_id]
        if self.enable_piece_confusion_matrix_metrics:
            for true_id in piece_ids:
                for pred_id in piece_ids:
                    metrics[f"piece_cm_{piece_names[true_id]}_{piece_names[pred_id]}"] = float(
                        confusion[(true_id, pred_id)]
                    )
        return metrics

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
                "qa_check_precision": 0.0,
                "qa_check_recall": 0.0,
                "qa_check_f1": 0.0,
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

        metrics = self._compute_binary_f1_metrics(tp=tp, fp=fp, fn=fn)
        if self.enable_qa_check_confusion_matrix_metrics:
            metrics["qa_check_tp"] = float(tp)
            metrics["qa_check_fp"] = float(fp)
            metrics["qa_check_tn"] = float(tn)
            metrics["qa_check_fn"] = float(fn)
        return metrics

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
            "qa_check_precision": precision,
            "qa_check_recall": recall,
            "qa_check_f1": check_f1,
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
