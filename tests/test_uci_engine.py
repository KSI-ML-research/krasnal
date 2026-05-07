import os
import sys

import chess
import chess.engine
import pytest

from krasnal.uci_engine.provider import ChessModelProvider, ModelProviderError
from krasnal.uci_engine.uci_parser import UCIParser


class FailingProvider(ChessModelProvider):
    def reset_session(self, outcome_token: int) -> None:
        self.outcome_token = outcome_token

    def get_best_move(self, _uci_moves: str) -> str:
        raise ModelProviderError("model crashed")


def test_uci_resigns_on_unrecoverable_model_provider_error(capsys):
    parser = UCIParser(FailingProvider())

    parser._process_command("ucinewgame")
    parser._process_command("go")

    output = capsys.readouterr().out.splitlines()
    assert "info string ModelProvider error: model crashed" in output
    assert "bestmove resign" in output


@pytest.mark.asyncio
async def test_uci_integration():
    # Run the engine as a separate process with mock provider
    env = os.environ.copy()
    env["KRASNAL_ENGINE_PROVIDER"] = "mock"
    _, engine = await chess.engine.popen_uci(
        [sys.executable, "-m", "krasnal.uci_engine.run"], env=env
    )

    try:
        # popen_uci automatically executes 'uci' and 'isready' during initialization

        # Testing move generation from the starting position
        board = chess.Board()
        result = await engine.play(board, chess.engine.Limit(time=0.1))
        assert result.move is not None
        assert result.move in board.legal_moves

        # Testing move generation after a move history
        board.push(result.move)
        board.push(chess.Move.from_uci("e7e5"))  # Example black move
        result = await engine.play(board, chess.engine.Limit(time=0.1))
        assert result.move is not None
        assert result.move in board.legal_moves
    finally:
        # Properly close the engine (send 'quit')
        await engine.quit()
