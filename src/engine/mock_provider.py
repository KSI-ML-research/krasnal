import random
import chess
from engine.provider import ChessModelProvider


class RandomMockProvider(ChessModelProvider):
    """
    Mockowa implementacja Providera - uderza na "sucho" bez ML.
    Wybiera losowy, ale legalny ruch, korzystając z biblioteki python-chess.
    """

    def get_best_move(self, uci_moves: str) -> str:
        """
        Zwraca losowy, ale *legalny* ruch na podstawie danej historii.
        """
        board = chess.Board()
        
        # Odtworzenie stanu planszy na podstawie historii
        if uci_moves.strip():
            for move_str in uci_moves.strip().split():
                try:
                    move = chess.Move.from_uci(move_str)
                    board.push(move)
                except ValueError as e:
                    print(f"DEBUG: Błąd parsowania ruchu '{move_str}': {e}")
        
        # Pobranie listy legalnych ruchów w aktualnej pozycji
        legal_moves = list(board.legal_moves)
        
        if not legal_moves:
            # W przypadku mata lub pata brakuje legalnych ruchów, zwracamy rzut oznaczający koniec (tzw. null move).
            return "0000" 
            
        # Wylosowanie legalnego ruchu z listy
        chosen_move = random.choice(legal_moves)
        
        # Konwersja wybranego obiektu Move z powrotem na string UCI (np. "e2e4")
        return chosen_move.uci()
