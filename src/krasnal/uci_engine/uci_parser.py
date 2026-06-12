import os
import sys

from loguru import logger

from krasnal.inference.exceptions import NoLegalMovesError
from krasnal.uci_engine.go_params import parse_go_rest
from krasnal.uci_engine.provider import ChessModelProvider, ModelProviderError


def moves_from_position_command(command: str) -> str:
    parts = command.split("moves", 1)
    return parts[1].strip() if len(parts) > 1 else ""


class UCIParser:
    """
    Loop listening for input messages and sending
    the game state to the Provider to obtain a move.

    With ``lazy_start=True`` (default for ``python -m krasnal.uci_engine.run``), the UCI
    handshake answers immediately and the heavy model load runs on ``isready`` or first use.

    Lichess inference uses a simple token stream: game start plus move tokens.
    """

    def __init__(
        self,
        provider: ChessModelProvider | None = None,
        *,
        engine_name: str = "Krasnal",
        lazy_start: bool = False,
    ):
        self._lazy_start = bool(lazy_start) and provider is None
        self.provider = provider
        self.engine_name = engine_name
        self._startup_error: str | None = None
        self.current_moves: str = ""

    def _ensure_provider(self) -> None:
        if not self._lazy_start:
            return
        if self.provider is not None or self._startup_error is not None:
            return
        from krasnal.uci_engine.bootstrap import build_provider

        try:
            self.provider, self.engine_name = build_provider()
        except Exception as exc:
            logger.exception("krasnal-uci: build_provider failed")
            self._startup_error = f"{type(exc).__name__}: {exc}"
            self.provider = None
            self.engine_name = os.environ.get("KRASNAL_UCI_ID_NAME", "Krasnal (model not loaded)")

    def _send_info_chunks(self, message: str, *, chunk: int = 220) -> None:
        """Emit UCI ``info string`` lines, chunked (some GUIs truncate one long line)."""
        collapsed = message.replace("\r", " ").replace("\n", " | ")
        for i in range(0, len(collapsed), chunk):
            self._send(f"info string {collapsed[i : i + chunk]}")

    def _emit_bestmove_none(self, info_body: str) -> None:
        self._send_info_chunks(info_body)
        self._send("bestmove (none)")

    def _handle_cmd_go(self, rest: str) -> None:
        self._ensure_provider()
        if self._startup_error is not None:
            logger.error("go failed (startup): {}", self._startup_error)
            self._emit_bestmove_none(f"krasnal-uci startup: {self._startup_error}")
            return
        if self.provider is None:
            logger.error("go failed: no provider")
            self._emit_bestmove_none("krasnal-uci internal error (no provider)")
            return

        go = parse_go_rest(rest)
        try:
            best = self.provider.think_and_move(self.current_moves, go)
        except ModelProviderError as e:
            logger.error("go failed (ModelProviderError): {}", e)
            if e.__cause__ is not None:
                logger.error("  cause: {}", e.__cause__)
            self._emit_bestmove_none(f"krasnal-uci ModelProviderError: {e}")
        except NoLegalMovesError as e:
            logger.error("go failed (NoLegalMovesError): {}", e)
            self._emit_bestmove_none(f"krasnal-uci NoLegalMovesError: {e}")
        except Exception as e:
            logger.exception("go failed ({})", type(e).__name__)
            self._emit_bestmove_none(f"krasnal-uci {type(e).__name__}: {e}")
        else:
            self._send(f"bestmove {best}")

    def run(self) -> None:
        """Main infinite loop listening for standard input commands."""
        for line in sys.stdin:
            command = line.strip()

            if not command:
                continue

            self._process_command(command)

    def _process_command(self, command: str) -> None:
        """React to one stripped inbound UCI command."""
        if command == "uci":
            self._send(f"id name {self.engine_name}")
            self._send("id author KSI UWr")
            self._send("uciok")
        elif command == "isready":
            self._ensure_provider()
            self._send("readyok")
        elif command == "ucinewgame":
            self.current_moves = ""
            self._ensure_provider()
            if self.provider is not None and self._startup_error is None:
                self.provider.reset_session()
        elif command.startswith("position"):
            self.current_moves = moves_from_position_command(command)
        elif command.startswith("go"):
            rest = command[2:].strip()
            self._handle_cmd_go(rest)
        elif command == "quit":
            sys.exit(0)

    def _send(self, msg: str) -> None:
        print(msg)
        sys.stdout.flush()
