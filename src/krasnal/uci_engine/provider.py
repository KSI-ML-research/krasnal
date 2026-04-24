from __future__ import annotations

import os
import random
from pathlib import Path

import bulletchess
import torch
from loguru import logger

from krasnal.inference import Game, InferenceSession, load_model
from krasnal.inference.exceptions import NoLegalMovesError
from krasnal.inference.sampling import sample_token
from krasnal.tokens import get_vocab_size, legal_token_ids, to_uci
from krasnal.utils import (
    build_gpt_config_from_artifact,
    resolve_runtime_device,
)


class ModelProviderError(ValueError):
    """Raised when the model provider encounters an unrecoverable error."""


class ChessModelProvider:
    """
    Interface (Protocol) for the chess engine.
    All implementations (mock, PyTorch model, web API)
    must satisfy this contract.
    """

    def reset_session(self, outcome_token: int) -> None:
        """Reset the inference session with outcome token. Called once per game."""
        ...

    def get_best_move(self, uci_moves: str) -> str:
        """
        Returns the best move in UCI notation based on the provided move history.

        Args:
            uci_moves: String representing current moves in the game in UCI format,
                       e.g., "e2e4 e7e5 g1f3".

        Returns:
            str: String in UCI notation representing the chosen move (e.g., "b8c6").

        Raises:
            ModelProviderError: If the model provider encounters an unrecoverable error.
            NoLegalMovesError: If the position has no legal moves.
        """
        ...


class RandomMockProvider(ChessModelProvider):
    """
    Chess model returning random legal move. Use for testing, not actual games ;)
    """

    def reset_session(self, outcome_token: int) -> None:
        pass  # No session needed for random provider

    def get_best_move(self, uci_moves: str) -> str:
        board = bulletchess.Board()

        if uci_moves.strip():
            for move_str in uci_moves.strip().split():
                try:
                    move = bulletchess.Move.from_uci(move_str)
                    board.apply(move)
                except Exception as e:
                    logger.error(f"Error parsing move '{move_str}': {e}")

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

    @classmethod
    def from_artifact_dir(
        cls,
        artifact_dir: Path,
        device: torch.device | None = None,
    ) -> ModelProvider:
        gpt_config = build_gpt_config_from_artifact(
            artifact_dir,
            vocab_size=get_vocab_size(),
        )
        runtime_device = device or resolve_runtime_device()
        model = load_model(str(artifact_dir / "model.pt"), runtime_device, gpt_config)
        return cls(
            model=model,
            device=runtime_device,
            artifact_dir=artifact_dir,
        )

    def reset_session(self, outcome_token: int) -> None:
        """Reset the inference session with outcome token. Called once per game."""
        self.outcome_token = outcome_token
        self.session = InferenceSession(
            self.model,
            self.device,
            game=Game(target_outcome_token=outcome_token),
        )

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
                session.feed_uci(move_str)
            except ValueError as exc:
                raise ModelProviderError(f"Invalid move in history: {move_str}") from exc

        return session

    def get_best_move(self, uci_moves: str) -> str:
        move_list = list(filter(None, uci_moves.split()))
        session = self._sync_session_history(move_list)

        legal_ids = legal_token_ids(session.game.board)
        if not legal_ids:
            raise NoLegalMovesError("No legal moves available")

        legal_probs = session.get_legal_probs()
        if torch.isnan(legal_probs).any():
            raise ModelProviderError("Could not produce legal move probabilities")

        best_token = sample_token(legal_probs, temperature=self.temperature, top_p=self.top_p)
        return to_uci(best_token)
