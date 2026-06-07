import random
from pathlib import Path
from typing import Any

import bulletchess
import torch
from loguru import logger
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence

from krasnal.config import CLOCK_IGNORE_ID
from krasnal.dataset import ChessDataset
from krasnal.eval.parsers import GameTokens, parse_row_to_game_tokens
from krasnal.eval.qa_probes import (
    compute_binary_f1_metrics,
    evaluate_is_check_probe_counts,
    evaluate_what_is_on_probe_stats,
    finalize_what_is_on_probe_stats,
)
from krasnal.eval.replayer import replay_games
from krasnal.eval.what_is_on_baseline import (
    WhatIsOnBaselineCounts,
)
from krasnal.inference import StatelessBatchInferenceSession
from krasnal.tokens import ELO_BUCKETS, PAD_ID, is_move_token_id, move_token_id_for_turn
from krasnal.utils import set_seed

from .metrics.context import EvalContext

EVAL_GAME_CHUNK_SIZE = 256
BASE_MOVE_METRICS = {"acc", "mrr", "top1_legal"}
ACC_FILTERS = {
    "acc_opening": "opening",
    "acc_middlegame": "middlegame",
    "acc_endgame": "endgame",
    "acc_when_gives_check": "when_gives_check",
    "acc_when_in_check": "when_in_check",
}
ELO_BUCKET_TO_TOKEN = {bucket_name: token for token, bucket_name in ELO_BUCKETS.items()}


class _MoveMetricAccumulator:
    def __init__(self, metric_names: list[str]):
        self.metric_names = metric_names
        self.sums: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        for name in metric_names:
            key = self._output_key(name)
            if key is not None:
                self.sums[key] = 0.0
                self.counts[key] = 0

    def update(self, contexts: list[EvalContext], logits: torch.Tensor) -> None:
        if not contexts:
            return

        metric_values = self._metric_values(contexts, logits)

        for name in self.metric_names:
            source = self._source_metric(name)
            key = self._output_key(name)
            if source is None or key is None:
                continue
            mask = self._filter_for_metric(name, contexts, logits.device)
            values = metric_values[source] if mask is None else metric_values[source][mask]
            if values.numel() == 0:
                continue
            self.sums[key] += float(values.float().sum().item())
            self.counts[key] += int(values.numel())

    def finalize(self) -> dict[str, float]:
        return {
            key: self.sums[key] / self.counts[key] if self.counts[key] else 0.0 for key in self.sums
        }

    def _metric_values(
        self,
        contexts: list[EvalContext],
        logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        device = logits.device
        actual = torch.tensor(
            [ctx.actual_token for ctx in contexts],
            dtype=torch.long,
            device=device,
        )
        top1 = logits.argmax(dim=1)
        top1_cpu = top1.detach().cpu().tolist()
        rank = logits.gt(logits.gather(1, actual[:, None])).sum(dim=1) + 1

        acc = top1.eq(actual).float()
        mrr = rank.float().reciprocal()
        top1_legal = torch.tensor(
            [
                1.0 if _is_legal_token_in_position(ctx, top_id) else 0.0
                for top_id, ctx in zip(top1_cpu, contexts, strict=True)
            ],
            dtype=torch.float32,
            device=device,
        )

        return {
            "acc": acc,
            "mrr": mrr,
            "top1_legal": top1_legal,
        }

    def _source_metric(self, name: str) -> str | None:
        if name in BASE_MOVE_METRICS:
            return name
        if name in ACC_FILTERS:
            return "acc"
        if name.startswith("acc_elo_"):
            return "acc"
        return None

    def _output_key(self, name: str) -> str | None:
        if name.startswith("acc_elo_"):
            bucket_name = name.removeprefix("acc_elo_")
            if bucket_name in ELO_BUCKET_TO_TOKEN:
                return f"acc/acc_elo_{bucket_name}"
            return None
        if self._source_metric(name) is not None:
            return name
        return None

    def _filter_for_metric(
        self,
        name: str,
        contexts: list[EvalContext],
        device: torch.device,
    ) -> torch.Tensor | None:
        if name.startswith("acc_elo_"):
            bucket_name = name.removeprefix("acc_elo_")
            elo_token = ELO_BUCKET_TO_TOKEN[bucket_name]
            return torch.tensor(
                [ctx.player_elo_token == elo_token for ctx in contexts],
                dtype=torch.bool,
                device=device,
            )
        if name in ACC_FILTERS:
            filter_name = ACC_FILTERS[name]
            if filter_name == "when_gives_check":
                values = [ctx.gives_check is True for ctx in contexts]
            elif filter_name == "when_in_check":
                values = [ctx.in_check is True for ctx in contexts]
            else:
                values = [ctx.phase == filter_name for ctx in contexts]
            return torch.tensor(values, dtype=torch.bool, device=device)
        return None


def _is_legal_token_in_position(ctx: EvalContext, token_id: int) -> bool:
    if ctx.fen is None:
        return False

    board = bulletchess.Board.from_fen(ctx.fen)
    for move in board.legal_moves():
        piece = board[move.origin]
        if piece is None:
            continue
        if move_token_id_for_turn(move.uci(), board.turn, piece.piece_type) == token_id:
            return True
    return False


def _update_context_sample(
    *,
    sample: list[EvalContext],
    seen: int,
    contexts: list[EvalContext],
    limit: int,
    rng: random.Random,
) -> int:
    if limit <= 0:
        sample.extend(contexts)
        return seen + len(contexts)

    for ctx in contexts:
        seen += 1
        if len(sample) < limit:
            sample.append(ctx)
            continue
        replacement_idx = rng.randrange(seen)
        if replacement_idx < limit:
            sample[replacement_idx] = ctx
    return seen


def _compact_clock_sequences(game_tokens: GameTokens) -> tuple[list[int], list[int]]:
    if (
        game_tokens.prefix_active_seconds is not None
        and game_tokens.prefix_opponent_seconds is not None
    ):
        prefix_active = game_tokens.prefix_active_seconds
        prefix_opponent = game_tokens.prefix_opponent_seconds
    else:
        prefix_active = CLOCK_IGNORE_ID
        prefix_opponent = CLOCK_IGNORE_ID

    active = [prefix_active] * len(game_tokens.initial_context)
    opponent = [prefix_opponent] * len(game_tokens.initial_context)
    active.extend(game_tokens.body_active_seconds or [])
    opponent.extend(game_tokens.body_opponent_seconds or [])
    return active, opponent


class ChessEvaluator:
    """Evaluates chess model on legal move metrics using batched inference."""

    def __init__(
        self,
        metrics: list[str] | None = None,
        seed: int | None = None,
        qa_config: dict[str, Any] | None = None,
        inference_batch_size: int = 64,
        min_ply: int = 0,
        min_active_clock: int | None = None,
    ):
        if metrics is None:
            raise ValueError("ChessEvaluator requires an explicit metrics list")

        self.requested_metrics = metrics
        self.seed = seed
        self.inference_batch_size = int(inference_batch_size)
        self.min_ply = int(min_ply)
        self.min_active_clock = min_active_clock

        qa_cfg = qa_config or {}
        self.qa_max_positions = int(qa_cfg.get("max_positions", 0))
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

        block_size = model.config.block_size
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        if num_games > 0:
            indices = indices[:num_games]

        eval_seed = seed if seed is not None else (self.seed if self.seed is not None else 0)
        final = self._evaluate_indices(
            model=model,
            dataset=dataset,
            indices=indices,
            block_size=block_size,
            device=device,
            eval_seed=eval_seed,
        )
        if final is not None:
            return final

        return _MoveMetricAccumulator(self.requested_metrics).finalize()

    def _evaluate_indices(
        self,
        model: torch.nn.Module,
        dataset: ChessDataset,
        indices: list[int],
        block_size: int,
        device: torch.device,
        eval_seed: int,
    ) -> dict[str, Any] | None:
        batch_session = StatelessBatchInferenceSession(model, device)
        move_metrics = _MoveMetricAccumulator(self.requested_metrics)
        qa_contexts: list[EvalContext] = []
        qa_contexts_seen = 0
        qa_rng = random.Random(eval_seed)
        sample_qa = self.enable_qa_check_metrics or self.enable_what_is_on_probe_metrics
        total_contexts = 0

        for start in range(0, len(indices), EVAL_GAME_CHUNK_SIZE):
            chunk_indices = indices[start : start + EVAL_GAME_CHUNK_SIZE]
            game_tokens_list = []
            for idx in chunk_indices:
                row = dataset[idx]
                game_tokens = parse_row_to_game_tokens(row)
                if game_tokens is not None:
                    game_tokens_list.append(game_tokens)

            contexts = replay_games(game_tokens_list, block_size)
            if not contexts:
                continue

            keep_indices = [idx for idx, ctx in enumerate(contexts) if self._include_context(ctx)]
            if not keep_indices:
                continue
            contexts = [contexts[idx] for idx in keep_indices]

            total_contexts += len(contexts)
            if sample_qa:
                qa_contexts_seen = _update_context_sample(
                    sample=qa_contexts,
                    seen=qa_contexts_seen,
                    contexts=contexts,
                    limit=self.qa_max_positions,
                    rng=qa_rng,
                )
            logits = self._infer_game_move_logits(game_tokens_list, batch_session)
            logits = logits[torch.tensor(keep_indices, device=logits.device)]
            if logits.size(0) != len(contexts):
                raise RuntimeError(
                    f"full-game logits/context mismatch: {logits.size(0)} != {len(contexts)}"
                )
            move_metrics.update(contexts, logits)

        if total_contexts == 0:
            return None

        final: dict[str, Any] = move_metrics.finalize()
        if self.enable_qa_check_metrics:
            check_counts = evaluate_is_check_probe_counts(
                qa_contexts,
                model,
                device,
                batch_size=self.inference_batch_size,
            )
            final.update(
                compute_binary_f1_metrics(
                    tp=check_counts["tp"],
                    fp=check_counts["fp"],
                    fn=check_counts["fn"],
                )
            )
            if self.enable_qa_check_confusion_matrix_metrics:
                final["qa/is_check/confusion_matrix/tp"] = float(check_counts["tp"])
                final["qa/is_check/confusion_matrix/fp"] = float(check_counts["fp"])
                final["qa/is_check/confusion_matrix/tn"] = float(check_counts["tn"])
                final["qa/is_check/confusion_matrix/fn"] = float(check_counts["fn"])
        if self.enable_what_is_on_probe_metrics:
            what_is_on_stats = evaluate_what_is_on_probe_stats(
                qa_contexts,
                model,
                device,
                eval_seed,
                baseline=self.what_is_on_baseline,
                batch_size=self.inference_batch_size,
            )
            final.update(
                finalize_what_is_on_probe_stats(
                    what_is_on_stats,
                    include_per_square=self.enable_what_is_on_accuracy_per_square_metrics,
                    include_baseline=self.what_is_on_baseline is not None,
                )
            )
        return final

    def _include_context(self, ctx: EvalContext) -> bool:
        if ctx.what_is_on_ply is not None and ctx.what_is_on_ply < self.min_ply:
            return False
        return not (
            self.min_active_clock is not None
            and ctx.active_clock_seconds is not None
            and ctx.active_clock_seconds < self.min_active_clock
        )

    def _infer_game_move_logits(
        self,
        game_tokens_list: list[GameTokens],
        batch_session: StatelessBatchInferenceSession,
    ) -> torch.Tensor:
        all_logits = []
        for start in range(0, len(game_tokens_list), self.inference_batch_size):
            chunk = [
                game_tokens
                for game_tokens in game_tokens_list[start : start + self.inference_batch_size]
                if len(game_tokens.body_tokens) + len(game_tokens.initial_context)
                <= batch_session.model.config.block_size
            ]
            if not chunk:
                continue

            token_rows = [
                torch.tensor(
                    [*game_tokens.initial_context, *game_tokens.body_tokens],
                    dtype=torch.long,
                )
                for game_tokens in chunk
            ]
            clock_rows = [_compact_clock_sequences(game_tokens) for game_tokens in chunk]
            active_rows = [
                torch.tensor(active, dtype=torch.long) for active, _opponent in clock_rows
            ]
            opponent_rows = [
                torch.tensor(opponent, dtype=torch.long) for _active, opponent in clock_rows
            ]

            x_rows = [tokens[:-1] for tokens in token_rows]
            active_x_rows = [active[1:] for active in active_rows]
            opponent_x_rows = [opponent[1:] for opponent in opponent_rows]
            x = pad_sequence(x_rows, batch_first=True, padding_value=PAD_ID).to(
                batch_session.device
            )
            active_x = pad_sequence(
                active_x_rows,
                batch_first=True,
                padding_value=CLOCK_IGNORE_ID,
            ).to(batch_session.device)
            opponent_x = pad_sequence(
                opponent_x_rows,
                batch_first=True,
                padding_value=CLOCK_IGNORE_ID,
            ).to(batch_session.device)

            with torch.inference_mode(), batch_session._amp_ctx:
                logits, _ = batch_session.model(
                    x,
                    active_clock_ids=active_x,
                    opponent_clock_ids=opponent_x,
                    return_all_logits=True,
                )

            for row_idx, game_tokens in enumerate(chunk):
                prefix_len = len(game_tokens.initial_context)
                positions = [
                    prefix_len + body_idx
                    for body_idx, token in enumerate(game_tokens.body_tokens)
                    if is_move_token_id(token)
                ]
                logit_positions = torch.tensor(
                    [pos - 1 for pos in positions],
                    device=batch_session.device,
                )
                all_logits.append(logits[row_idx, logit_positions])

        if not all_logits:
            raise ValueError("no game logits to concatenate")
        return torch.cat(all_logits, dim=0)


def chess_evaluator_from_config(cfg: Any, *, metrics: list[str]) -> ChessEvaluator:
    eval_batch_size = cfg.eval.get("inference_batch_size")
    if eval_batch_size is None:
        train_cfg = cfg.get("train", {})
        eval_batch_size = train_cfg.get("batch_size", 64)
    return ChessEvaluator(
        metrics=metrics,
        seed=cfg.seed,
        qa_config=OmegaConf.to_container(cfg.eval.qa, resolve=True),
        inference_batch_size=int(eval_batch_size),
        min_ply=int(cfg.eval.get("min_ply", 0)),
        min_active_clock=cfg.eval.get("min_active_clock"),
    )
