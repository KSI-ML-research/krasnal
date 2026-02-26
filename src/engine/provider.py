from typing import Protocol


class ChessModelProvider(Protocol):
    """
    Interfejs (Protocol) dla silnika szachowego.
    Wszystkie implementacje (mock, model PyTorch, API sieciowe)
    muszą spełniać ten kontrakt.
    """

    def get_best_move(self, uci_moves: str) -> str:
        """
        Zwraca najlepszy ruch w notacji UCI na podstawie podanej historii ruchów.

        Args:
            uci_moves: Ciąg znaków reprezentujący dotychczasowe ruchy w grze w formacie UCI,
                       np. "e2e4 e7e5 g1f3".

        Returns:
            str: Ciąg znaków w notacji UCI reprezentujący wybrany ruch (np. "b8c6").
        """
        # TODO: Zostawiamy pusty kontrakt (to interfejs).
        ...
