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
            self._send("id name Krasnal Mock")
            self._send("id author Zespol Krasnal")
            self._send("uciok")

        # 2. Silnik otrzymał komendę gotowości - powinien odpowiedzieć, że jest OK.
        elif command == "isready":
            self._send("readyok")

        # 3. Zaktualizowanie stanu gry / nowa partia / restart gry
        elif command == "ucinewgame":
            self.current_moves = ""

        # 4. Ustala historię gry lub pozycję startową
        # Przykład wejścia: "position startpos moves e2e4 e7e5 g1f3"
        elif command.startswith("position"):
            parts = command.split("moves")
            if len(parts) > 1:
                # Bierzemy wszystko po słowie 'moves' i usuwamy białe znaki
                self.current_moves = parts[1].strip()
            else:
                # Jeśli słowa 'moves' nie było, znaczy że jesteśmy na starcie (bez historii)
                self.current_moves = ""

        # 5. Silnik wywoływany przez bota lichess-a o podanie ruchu z ustalonym stanem.
        elif command.startswith("go"):
            best_move = self.provider.get_best_move(self.current_moves)
            self._send(f"bestmove {best_move}")

        # 6. Silnik wyłącza działanie
        elif command == "quit":
            sys.exit(0)

    def _send(self, msg: str):
        """
        Wysyła komunikat na standardowe wyjście i wymusza opróżnienie bufora.
        Jest to niezbędne, żeby lichess-bot otrzymał wiadomość natychmiast, a nie
        dopiero po zapełnieniu całego bufora stdout!
        """
        print(msg)
        sys.stdout.flush()
