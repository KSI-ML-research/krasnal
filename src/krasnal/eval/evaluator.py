import random
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from omegaconf import OmegaConf

from krasnal.dataset import ChessDataset
from krasnal.eval.parsers import parse_row_to_game_tokens
from krasnal.eval.qa_probes import (
    evaluate_is_check_probe,
    evaluate_what_is_on_probe,
)
from krasnal.eval.replayer import replay_games
from krasnal.eval.what_is_on_baseline import (
    WhatIsOnBaselineCounts,
)
from krasnal.inference import StatelessBatchInferenceSession
from krasnal.utils import set_seed

from .metrics import METRIC_REGISTRY
from .metrics.context import EvalContext
from .metrics.core import AccuracyCore, CoreMetric, IllegalMassCore, MRRCore, Top1LegalCore
from .metrics.filtered import WhenLowTimeMetric

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
            for metric in self.metrics.values():
                result = metric.compute(ctx)
                for k, v in result.items():
                    results[k].append(v)

        final = self._aggregate_results(results)
        if self.enable_qa_check_metrics:
            final.update(
                evaluate_is_check_probe(
                    contexts,
                    model,
                    device,
                    include_confusion_matrix=self.enable_qa_check_confusion_matrix_metrics,
                )
            )
        if self.enable_what_is_on_probe_metrics:
            final.update(
                evaluate_what_is_on_probe(
                    contexts,
                    model,
                    device,
                    eval_seed,
                    include_per_square=self.enable_what_is_on_accuracy_per_square_metrics,
                    baseline=self.what_is_on_baseline,
                )
            )
        return final

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
