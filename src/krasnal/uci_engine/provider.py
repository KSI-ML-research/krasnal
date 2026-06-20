from __future__ import annotations

import os
import random
import time as _time
from abc import ABC, abstractmethod
from pathlib import Path

import bulletchess
import torch
import xgboost as xgb
from loguru import logger

from krasnal.inference import Game, InferenceSession, load_model
from krasnal.inference.exceptions import NoLegalMovesError
from krasnal.inference.sampling import sample_token
from krasnal.move_time.xgboost import predict_single
from krasnal.tokens import (
    legal_token_ids,
    load_move_vocab,
    normalize_history_uci_moves,
    to_uci,
)
from krasnal.uci_engine.go_params import GoParams
from krasnal.utils import (
    gpt_config_from_artifact_dict,
    read_model_config_json,
    resolve_runtime_device,
)


class ModelProviderError(ValueError):
    """Raised when the model provider encounters an unrecoverable error."""


class ChessModelProvider(ABC):
    """
    Interface for the chess engine (UCI bridge).
    """

    @abstractmethod
    def reset_session(self) -> None:
        """Reset for a new game (``ucinewgame``)."""

    @abstractmethod
    def get_best_move(self, uci_moves: str) -> str:
        """Return best move UCI for the given move list string."""

    def think_and_move(self, uci_moves: str, _go: GoParams) -> str:
        """Return best move after waiting for the predicted thinking time."""
        # Default: no delay
        return self.get_best_move(uci_moves)


class RandomMockProvider(ChessModelProvider):
    """
    Chess model returning random legal move. Use for testing, not actual games ;)
    """

    def reset_session(self) -> None:
        return None

    @staticmethod
    def _apply_uci_move_list(board: bulletchess.Board, uci_moves: str) -> None:
        for uci in normalize_history_uci_moves(uci_moves):
            move = bulletchess.Move.from_uci(uci)
            board.apply(move)

    def get_best_move(self, uci_moves: str) -> str:
        board = bulletchess.Board()
        if uci_moves.strip():
            self._apply_uci_move_list(board, uci_moves)

        legal_moves = list(board.legal_moves())

        if not legal_moves:
            raise NoLegalMovesError(f"No legal moves in position: {board.fen()}")

        chosen_move = random.choice(legal_moves)
        return chosen_move.uci()


class ModelProvider(ChessModelProvider):
    """Choose the highest-probability legal move from a loaded artifact."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        artifact_dir: Path | None = None,
        artifact_config: dict | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.artifact_dir = artifact_dir
        self.engine_name = (
            f"Krasnal Model ({artifact_dir.parent.name}/{artifact_dir.name})"
            if artifact_dir is not None
            else "Krasnal Model"
        )
        self.temperature = float(os.environ.get("KRASNAL_TEMPERATURE", "0.0"))
        self.top_p = float(os.environ.get("KRASNAL_TOP_P", "1.0"))
        self.session: InferenceSession | None = None
        self.artifact_config = artifact_config or {}
        self.xgb_model: xgb.Booster | None = None
        self._time_initial_seconds: int | None = None

    @classmethod
    def from_artifact_dir(
        cls,
        artifact_dir: Path,
        device: torch.device | None = None,
    ) -> ModelProvider:
        cfg_path = artifact_dir / "config.json"
        artifact_config = read_model_config_json(cfg_path)
        unsupported = []
        if bool(artifact_config.get("outcome_conditioning_enabled", False)):
            unsupported.append("outcome_conditioning")
        if bool(artifact_config["use_clock_encodings"]):
            unsupported.append("clock_encodings")
        if bool(artifact_config.get("time_control_token_enabled", True)):
            unsupported.append("time_control_token")
        if bool(artifact_config.get("opponent_material_enabled", False)):
            unsupported.append("opponent_material")
        if bool(artifact_config.get("include_elo", True)):
            unsupported.append("elo_tokens")
        if unsupported:
            raise ModelProviderError(
                "Artifacts with " + ", ".join(unsupported) + " are not supported for inference"
            )
        if not (artifact_dir / "model.pt").is_file():
            raise ValueError(
                f"Missing model.pt under {artifact_dir}. "
                "Training may still be running, or no checkpoint was saved yet.",
            )
        load_move_vocab(
            artifact_dir / "move_vocab.json",
            piece_aware_moves=bool(artifact_config.get("piece_aware_moves", False)),
            side_prefixed_moves=bool(artifact_config.get("side_prefixed_moves", True)),
        )
        gpt_config = gpt_config_from_artifact_dict(artifact_config)
        runtime_device = device or resolve_runtime_device()
        model = load_model(str(artifact_dir / "model.pt"), runtime_device, gpt_config)

        provider = cls(
            model=model,
            device=runtime_device,
            artifact_dir=artifact_dir,
            artifact_config=artifact_config,
        )

        xgb_path = artifact_dir / "xgboost_baseline.json"
        if xgb_path.is_file():
            xgb_model = xgb.Booster()
            xgb_model.load_model(str(xgb_path))
            provider.xgb_model = xgb_model
            logger.info("Loaded XGBoost move-time model from {}", xgb_path)
        else:
            logger.warning("No XGBoost model found at {}", xgb_path)

        return provider

    def _build_game(self) -> Game:
        return Game(
            elo_tokens_enabled=False,
            time_control_token_enabled=False,
        )

    def reset_session(self) -> None:
        """Reset for a new game."""
        self.session = InferenceSession(
            self.model,
            self.device,
            game=self._build_game(),
        )
        self._time_initial_seconds = None

    def _sync_session_history(self, move_list: list[str]) -> InferenceSession:
        if self.session is None:
            self.reset_session()

        session = self.session
        assert session is not None
        current_moves = session.game.moves_uci

        if move_list[: len(current_moves)] != current_moves:
            self.reset_session()
            session = self.session
            assert session is not None
            current_moves = session.game.moves_uci

        for move_str in move_list[len(current_moves) :]:
            try:
                session.feed_uci(move_str)
            except ValueError as exc:
                raise ModelProviderError(f"Invalid move in history: {move_str}") from exc

        return session

    def _pick_best_uci(self, session: InferenceSession) -> str:
        if not list(session.game.board.legal_moves()):
            raise NoLegalMovesError("No legal moves available")

        legal_ids = legal_token_ids(session.game.board)
        if not legal_ids:
            raise ModelProviderError("No legal move tokens available for current position")

        legal_probs = session.get_legal_probs()
        if torch.isnan(legal_probs).any():
            raise ModelProviderError("Could not produce legal move probabilities")

        best_token = sample_token(legal_probs, temperature=self.temperature, top_p=self.top_p)
        best_move = to_uci(best_token)
        if not best_move:
            raise ModelProviderError(f"Sampled token {best_token} is not a move")
        return best_move

    def get_best_move(self, uci_moves: str) -> str:
        try:
            move_list = normalize_history_uci_moves(uci_moves)
            session = self._sync_session_history(move_list)
            return self._pick_best_uci(session)
        except NoLegalMovesError:
            raise
        except ModelProviderError:
            raise
        except Exception as exc:
            logger.exception("ModelProvider.get_best_move failed")
            raise ModelProviderError(f"{type(exc).__name__}: {exc}") from exc

    def get_move_time(self, uci_moves: str, go: GoParams) -> float:
        """Predict thinking time in seconds using XGBoost (falls back to 0.0)."""
        if self.xgb_model is None:
            return 0.0

        move_list = normalize_history_uci_moves(uci_moves)
        session = self._sync_session_history(move_list)

        w_ms = go.wtime_ms
        b_ms = go.btime_ms
        if w_ms is None or b_ms is None:
            return 0.0

        remaining = max(w_ms, b_ms)
        if self._time_initial_seconds is None:
            self._time_initial_seconds = remaining // 1000

        time_initial = self._time_initial_seconds or 300
        turn = session.game.board.turn
        side_seconds = (w_ms // 1000) if turn == bulletchess.WHITE else (b_ms // 1000)

        ply = len(session.game.moves_uci)
        prev_clock_seconds = side_seconds
        clock_fraction_left = (prev_clock_seconds / time_initial) if time_initial > 0 else 0.0
        is_in_check = int(session.game.board in bulletchess.CHECK)
        fen_pieces = session.game.board.fen().split()[0]
        total_pieces = sum(1 for c in fen_pieces if c.isalpha())
        num_legal_moves = len(session.game.board.legal_moves())

        return predict_single(
            model=self.xgb_model,
            ply=ply,
            time_initial=time_initial,
            prev_clock_seconds=prev_clock_seconds,
            clock_fraction_left=clock_fraction_left,
            is_in_check_before_move=is_in_check,
            total_pieces=total_pieces,
            num_legal_moves=num_legal_moves,
        )

    def think_and_move(self, uci_moves: str, go: GoParams) -> str:
        """Return best move after waiting for the predicted thinking time."""
        if go.movetime_ms is not None:
            delay = go.movetime_ms / 1000.0
        else:
            delay = self.get_move_time(uci_moves, go)
        if delay > 0:
            _time.sleep(delay)
        return self.get_best_move(uci_moves)
