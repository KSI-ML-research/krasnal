class RandomMockProvider():
    """
    Mockowa implementacja Providera - uderza na "sucho" bez ML.
    Służy wyłącznie do testów integracji `lichess-bot` i interfejsu UCI.
    """

    def get_best_move(self, uci_moves: str) -> str:
        """
        Zwraca losowy, ale *legalny* ruch na podstawie danej historii.

        Zależność do przyszłej implementacji: `python-chess`
        W przyszłości ten model odtworzy stan planszy z `uci_moves`,
        sprawdzi listę legalnych ruchów i wylosuje jeden z nich.
        """
        # TODO: Zaimplementować z `python-chess` lub napisać na razie stały ciąg znaków (np. "e2e4").
        # Zastąpić to właściwą logiką.
        
        print(f"DEBUG: Otrzymano historię ruchów: '{uci_moves}'")
        return "e2e4"  # Stała zwracana wartość na ten moment.
