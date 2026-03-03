import logging
import random
import chess
from engine.provider import ChessModelProvider

logger = logging.getLogger(__name__)

class RandomMockProvider(ChessModelProvider):
    """
    Chess model returning random legal move. Use for testing, not acutal games ;)
    """

    def get_best_move(self, uci_moves: str) -> str:
        board = chess.Board()
        
        # Recreate the board state based on move history
        if uci_moves.strip():
            for move_str in uci_moves.strip().split():
                try:
                    move = chess.Move.from_uci(move_str)
                    board.push(move)
                except ValueError as e:
                    logger.error(f"Error parsing move '{move_str}': {e}")
        
        # Get a list of legal moves in the current position
        legal_moves = list(board.legal_moves)
        
        if not legal_moves:
            # In case of checkmate or stalemate, there are no legal moves, return the null move.
            return "0000" 
            
        # Choose a random legal move from the list
        chosen_move = random.choice(legal_moves)
        
        # Convert the chosen Move object back to a UCI string (e.g., "e2e4")
        return chosen_move.uci()
