import asyncio
import sys

import chess.engine


async def run_uci_test():
    """Logic for the UCI engine integration test."""
    # Run the engine as a separate process, simulating Lichess/Arena environment.
    # Add 'src' to PYTHONPATH so that 'from engine import ...' imports work.
    _, engine = await chess.engine.popen_uci(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, 'src'); from engine import run; run.main()",
        ]
    )

    try:
        # Testing the handshake (uci, isready)
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


def test_krasnal_uci_integration():
    """Integration test to verify UCI engine communication and move generation."""
    asyncio.run(run_uci_test())
