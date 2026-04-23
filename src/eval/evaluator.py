import logging
import random
from typing import Any

import bulletchess
import torch

from config import DRAW_ID, EOS_ID, SOS_ID, WIN_BLACK_ID, WIN_WHITE_ID
from dataset import ChessDataset
from inference import BatchInferenceSession, get_legal_token_ids
from tokenizer import Tokenizer
from utils import set_seed

from .metrics import DEFAULT_METRICS, METRIC_REGISTRY
from .metrics.context import EvalContext
from .stockfish import StockfishClient

logger = logging.getLogger(__name__)

PIECE_TYPE_TO_INT = {pt: i + 1 for i, pt in enumerate(bulletchess.PIECE_TYPES)}


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
        self.requested_metrics = metrics or self.DEFAULT_METRICS
        self.cot = cot
        self.cot_max_tokens = cot_max_tokens
        self.seed = seed
        self.stockfish = stockfish
        self.acpl_sample_size = acpl_sample_size
        self.metrics = self._init_metrics()

    def _init_metrics(self) -> dict[str, Any]:
        metrics = {}
        phase_metric_suffixes = {"_opening", "_middlegame", "_endgame"}
        
        for name in self.requested_metrics:
            if name in METRIC_REGISTRY:
                if name == "acpl":
                    metrics[name] = METRIC_REGISTRY[name](
                        stockfish=self.stockfish, sample_size=self.acpl_sample_size
                    )
                elif any(name.endswith(suffix) for suffix in phase_metric_suffixes):
                    # Phase-based metrics: extract phase from metric name
                    phase = name.rsplit("_", 1)[1]
                    metrics[name] = METRIC_REGISTRY[name](phase=phase)
                else:
                    metrics[name] = METRIC_REGISTRY[name]()
        return metrics

    def evaluate(
        self,
        model: torch.nn.Module,
        tokenizer: Tokenizer,
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

        block_size = model.config.block_size
        special_ids = {SOS_ID, EOS_ID, WIN_WHITE_ID, WIN_BLACK_ID, DRAW_ID}

        indices = list(range(len(dataset)))
        random.shuffle(indices)
        indices = indices[:num_games]

        contexts: list[EvalContext] = []

        for _game_idx, idx in enumerate(indices):
            token_ids = dataset[idx].tolist()

            outcome_token = (
                token_ids[0]
                if token_ids and token_ids[0] in {WIN_WHITE_ID, WIN_BLACK_ID, DRAW_ID, SOS_ID}
                else SOS_ID
            )

            moves = [t for t in token_ids if t not in special_ids]
            if len(moves) < 1:
                continue

            if len(moves) + 1 > block_size:
                logger.warning(
                    f"Game {idx} length ({len(moves) + 1}) exceeds {block_size=}. Skipping."
                )
                continue

            board = bulletchess.Board()
            context = [outcome_token]
            ply_count = 0

            for move_idx, move_token in enumerate(moves):
                legal_ids = get_legal_token_ids(board, tokenizer)
                if not legal_ids:
                    break

                in_check = board in bulletchess.CHECK
                if ply_count < 20:
                    phase = "opening"
                elif ply_count < 80:
                    phase = "middlegame"
                else:
                    phase = "endgame"

                uci_move = tokenizer.id_to_move.get(move_token)
                if not uci_move:
                    logger.warning(
                        "Stopping game %s at move %s: unknown token_id=%s",
                        idx,
                        move_idx + 1,
                        move_token,
                    )
                    break
                try:
                    move = bulletchess.Move.from_uci(uci_move)
                except Exception:
                    logger.warning(
                        "Stopping game %s at move %s: invalid replay move=%s token_id=%s",
                        idx,
                        move_idx + 1,
                        uci_move,
                        move_token,
                    )
                    break

                piece = board[move.origin]
                piece_type = PIECE_TYPE_TO_INT.get(piece.piece_type, 0) if piece else 0

                fen = board.fen()

                board.apply(move)
                gives_check = board in bulletchess.CHECK

                contexts.append(
                    EvalContext(
                        probs=None,
                        legal_ids=legal_ids,
                        sequence=context.copy(),
                        piece_type=piece_type,
                        actual_token=move_token,
                        in_check=in_check,
                        phase=phase,
                        gives_check=gives_check,
                        fen=fen,
                        top1_fen=None,
                    )
                )

                context.append(move_token)
                ply_count += 1

        if not contexts:
            return {name: 0.0 for name in self.metrics}

        all_positions = [ctx.sequence for ctx in contexts]
        batch_session = BatchInferenceSession(model, device)
        probs = batch_session.get_probs_batch(all_positions)

        for ctx, prob in zip(contexts, probs, strict=True):
            ctx.probs = prob

        results: dict[str, list[float]] = {
            name: [] for name, m in self.metrics.items() if not hasattr(m, "finalize")
        }

        for ctx in contexts:
            if ctx.legal_ids:
                legal_probs = [(tid, ctx.probs[tid].item()) for tid in ctx.legal_ids]
                top1_token = max(legal_probs, key=lambda x: x[1])[0]
                uci_move = tokenizer.id_to_move.get(top1_token)
                if uci_move:
                    try:
                        board = bulletchess.Board.from_fen(ctx.fen)
                        top1_move = bulletchess.Move.from_uci(uci_move)
                        board.apply(top1_move)
                        ctx.top1_fen = board.fen()
                    except Exception as e:
                        logger.warning(
                            f"Failed to compute top1_fen: {uci_move} on {ctx.fen[:30]}...: {e}"
                        )

            for metric in self.metrics.values():
                result = metric.compute(ctx)
                for k, v in result.items():
                    results[k].append(v)

        final_results: dict[str, float] = {}
        for k, v in results.items():
            final_results[k] = sum(v) / len(v) if v else 0.0
        for metric in self.metrics.values():
            if hasattr(metric, "finalize"):
                final_results.update(metric.finalize())
        return final_results
