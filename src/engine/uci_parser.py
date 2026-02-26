import sys
from engine.provider import ChessModelProvider


class UCIParser:
    """
    Pętla nasłuchująca komunikaty na wejściu i przesyłająca
    stan gry do Providera w celu uzyskania ruchu.
    """

    def __init__(self, provider: ChessModelProvider):
        self.provider = provider
        # Aktualny stan planszy / historii w formacie UCI
        self.current_moves: str = ""

    def run(self):
        """
        Główna nieskończona pętla nasłuchująca poleceń standardowego wejścia.
        """
        for line in sys.stdin:
            command = line.strip()
            
            # Pusta linijka (np. EOF lub zwykły enter)
            if not command:
                continue

            self._process_command(command)

    def _process_command(self, command: str):
        """
        Główna funkcja parsowania komend. Obsługuje
        komendy wymagane przez protokół UCI.
        """
        
        # 1. Rozpoznanie bota przez lichess-bota (przedstawienie się)
        if command == "uci":
            # TODO: Wypisać "id name Krasnal" na standardowe wyjście (stdout).
            # TODO: Wypisać "id author Zespol Krasnal"
            # TODO: Wypisać "uciok", kończące handshake'a uci.
            pass

        # 2. Silnik otrzymał komendę gotowości - powinien odpowiedzieć, że jest OK.
        elif command == "isready":
            # TODO: Wypisać "readyok".
            pass

        # 3. Zaktualizowanie stanu gry / nowa partia / restart gry
        elif command == "ucinewgame":
            # TODO: Wyzerować stan i przygotować nowe okienko.
            pass

        # 4. Ustala historię gry lub pozycję startową
        elif command.startswith("position"):
            # TODO: Parsować string z historią `uci_moves` po `position startpos moves `
            # Zrobić `self.current_moves = wyłapane ruchy`.
            pass

        # 5. Silnik wywoływany przez bota lichess-a o podanie ruchu z ustalonym stanem.
        elif command.startswith("go"):
            # TODO: Zapytać Providera (`self.provider.get_best_move(self.current_moves)`).
            # Otrzymany wynik wypisać jako `bestmove {nasz_ruch}` np. "bestmove e2e4"
            pass

        # 6. Silnik wyłącza działanie
        elif command == "quit":
            sys.exit(0)
