from __future__ import annotations

import os
import random
from abc import ABC, abstractmethod
from pathlib import Path

import bulletchess
import torch
from loguru import logger

from krasnal.inference import Game, InferenceSession, load_model
from krasnal.inference.exceptions import NoLegalMovesError
from krasnal.inference.sampling import sample_token
from krasnal.time_conditioning import uniform_clock_pair
from krasnal.tokens import (
    ELO_ABOVE_2200_ID,
    TC_UNKNOWN_ID,
    get_elo_bucket,
    get_time_control_bucket,
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
    def reset_session(self, outcome_token: int) -> None:
        """Reset for a new game (``ucinewgame``)."""

    @abstractmethod
    def get_best_move(self, uci_moves: str) -> str:
        """Return best move UCI for the given move list string."""

    def set_go_params(self, params: GoParams | None) -> None:
        """Latest ``go`` line (clocks); default no-op for mocks."""
        _ = params

    def apply_setoption(self, name: str, value: str) -> None:
        """Optional ``setoption`` (Elo / TC metadata); default no-op."""
        _ = (name, value)


class RandomMockProvider(ChessModelProvider):
    """
    Chess model returning random legal move. Use for testing, not actual games ;)
    """

    def reset_session(self, outcome_token: int) -> None:
        _ = outcome_token

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


def _parse_spin_option(value: str) -> int | None:
    v = value.strip()
    if not v:
        return None
    try:
        n = int(v)
    except ValueError:
        return None
    return None if n < 0 else n


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
        self.outcome_token: int | None = None
        self.session: InferenceSession | None = None
        self.artifact_config = artifact_config or {}
        self.outcome_conditioning_enabled = bool(
            self.artifact_config.get("outcome_conditioning_enabled", True)
        )
        self._last_go: GoParams | None = None
        self._white_elo: int | None = None
        self._black_elo: int | None = None
        self._tc_initial_sec: int | None = None
        self._tc_inc_sec: int | None = None

    @classmethod
    def from_artifact_dir(
        cls,
        artifact_dir: Path,
        device: torch.device | None = None,
    ) -> ModelProvider:
        cfg_path = artifact_dir / "config.json"
        artifact_config = read_model_config_json(cfg_path)
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
        return cls(
            model=model,
            device=runtime_device,
            artifact_dir=artifact_dir,
            artifact_config=artifact_config,
        )

    def _white_elo_token(self) -> int:
        return get_elo_bucket(self._white_elo) if self._white_elo is not None else ELO_ABOVE_2200_ID

    def _black_elo_token(self) -> int:
        return get_elo_bucket(self._black_elo) if self._black_elo is not None else ELO_ABOVE_2200_ID

    def _time_control_token(self) -> int:
        if self._tc_initial_sec is None or self._tc_inc_sec is None:
            return TC_UNKNOWN_ID
        return get_time_control_bucket(self._tc_initial_sec, self._tc_inc_sec)

    def _build_game(self, outcome_token: int) -> Game:
        return Game(
            white_elo_token=self._white_elo_token(),
            black_elo_token=self._black_elo_token(),
            time_control_token=self._time_control_token(),
            target_outcome_token=outcome_token,
            outcome_conditioning_enabled=self.outcome_conditioning_enabled,
        )

    def _clock_initial_seconds_for_session(self) -> int | None:
        if not self.model.config.use_time_conditioning:
            return None
        if self._tc_initial_sec is not None:
            return self._tc_initial_sec
        raw = self.artifact_config.get("time_initial")
        if raw is not None:
            return int(raw)
        raise ModelProviderError(
            "krasnalInitialSeconds setoption (or time_initial in model config) is required "
            "when use_time_conditioning is enabled"
        )

    def reset_session(self, outcome_token: int) -> None:
        """Reset for a new game; clears ``go`` metadata but keeps setoption TC/Elo."""
        self.outcome_token = outcome_token
        self._last_go = None
        self.session = InferenceSession(
            self.model,
            self.device,
            game=self._build_game(outcome_token),
            clock_initial_seconds=self._clock_initial_seconds_for_session(),
        )

    def set_go_params(self, params: GoParams | None) -> None:
        self._last_go = params

    def apply_setoption(self, name: str, value: str) -> None:
        key = name.strip()
        val = _parse_spin_option(value)
        match key.lower():
            case "krasnalwhiteelo":
                self._white_elo = val
            case "krasnalblackelo":
                self._black_elo = val
            case "krasnalinitialseconds":
                self._tc_initial_sec = val
            case "krasnalincrementseconds":
                self._tc_inc_sec = val
            case _:
                return
        self._sync_session_conditioning()

    def _sync_session_conditioning(self) -> None:
        if self.session is None:
            return
        g = self.session.game
        g.white_elo_token = self._white_elo_token()
        g.black_elo_token = self._black_elo_token()
        g.time_control_token = self._time_control_token()
        self.session.sync_prefix_tokens_from_game()

    def _sync_session_history(self, move_list: list[str]) -> InferenceSession:
        if self.session is None or self.outcome_token is None:
            raise ModelProviderError("Session not initialized. Call reset_session first.")

        session = self.session
        current_moves = session.game.moves_uci

        if move_list[: len(current_moves)] != current_moves:
            self.reset_session(self.outcome_token)
            session = self.session
            assert session is not None
            current_moves = session.game.moves_uci

        for move_str in move_list[len(current_moves) :]:
            try:
                session.feed_uci(
                    move_str,
                    *uniform_clock_pair(self._clock_initial_seconds_for_session()),
                )
            except ValueError as exc:
                raise ModelProviderError(f"Invalid move in history: {move_str}") from exc

        return session

    def _pick_best_uci(self, session: InferenceSession) -> str:
        if not list(session.game.board.legal_moves()):
            raise NoLegalMovesError("No legal moves available")

        session.prepare_go_clocks(self._last_go)

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
