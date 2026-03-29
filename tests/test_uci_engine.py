import os
import sys

import chess
import chess.engine
import pytest


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
