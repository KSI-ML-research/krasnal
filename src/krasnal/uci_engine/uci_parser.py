import sys

from krasnal.tokens import BLACK_WON_ID, WHITE_WON_ID
from krasnal.uci_engine.provider import ChessModelProvider, ModelProviderError


class UCIParser:
    """
    Loop listening for input messages and sending
    the game state to the Provider to obtain a move.
    """

    def __init__(self, provider: ChessModelProvider, engine_name: str = "Krasnal"):
        self.provider = provider
        self.engine_name = engine_name
        # Current board state / history in UCI format
        self.current_moves: str = ""
        # Engine side for lichess-bot: "white", "black", or "both" (default)
        self.engine_side: str = "both"

    def _get_outcome_token(self) -> int:
        """Derive outcome token based on engine side and move count."""
        move_count = len(self.current_moves.split()) if self.current_moves else 0

        if self.engine_side == "white":
            return WHITE_WON_ID
        elif self.engine_side == "black":
            return BLACK_WON_ID
        else:
            # "both" - derive from move count (even = white to move, odd = black to move)
            if move_count % 2 == 0:
                return WHITE_WON_ID
            else:
                return BLACK_WON_ID

    def run(self):
        """
        Main infinite loop listening for standard input commands.
        """
        for line in sys.stdin:
            command = line.strip()

            # Empty line (e.g., EOF or simple enter)
            if not command:
                continue

            self._process_command(command)

    def _process_command(self, command: str):
        """
        Main command parsing function. Handles
        commands required by the UCI protocol.
        """

        # 1. Handshake (engine identification)
        if command == "uci":
            self._send(f"id name {self.engine_name}")
            self._send("id author KSI UWr")
            self._send("uciok")

        # 2. Engine receives a readiness command - should respond that it is OK.
        elif command == "isready":
            self._send("readyok")

        # 3. Update game state / new game / game restart
        elif command == "ucinewgame":
            self.current_moves = ""
            outcome_token = self._get_outcome_token()
            self.provider.reset_session(outcome_token)

        # 4. Sets game history or starting position
        # Input example: "position startpos moves e2e4 e7e5 g1f3"
        elif command.startswith("position"):
            parts = command.split("moves")
            if len(parts) > 1:
                # We take everything after the word 'moves' and strip whitespace
                self.current_moves = parts[1].strip()
            else:
                # If 'moves' word was missing, it means we are at the start (no history)
                self.current_moves = ""

        # 5. Engine called to provide a move with current state.
        elif command.startswith("go"):
            try:
                best_move = self.provider.get_best_move(self.current_moves)
            except ModelProviderError as e:
                self._send(f"info string ModelProvider error: {e}")
                raise
            self._send(f"bestmove {best_move}")

        # 6. Engine exits
        elif command == "quit":
            sys.exit(0)

    def _send(self, msg: str):
        """
        Sends a message to standard output and forces a buffer flush.
        This is necessary so the client (like lichess-bot) receives the message immediately,
        rather than after the entire stdout buffer is full!
        """
        print(msg)
        sys.stdout.flush()
