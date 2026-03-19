from typing import Protocol


class ChessModelProvider(Protocol):
    """
    Interface (Protocol) for the chess engine.
    All implementations (mock, PyTorch model, web API)
    must satisfy this contract.
    """

    def get_best_move(self, uci_moves: str) -> str:
        """
        Returns the best move in UCI notation based on the provided move history.

        Args:
            uci_moves: String representing current moves in the game in UCI format,
                       e.g., "e2e4 e7e5 g1f3".

        Returns:
            str: String in UCI notation representing the chosen move (e.g., "b8c6").
        """
        ...
