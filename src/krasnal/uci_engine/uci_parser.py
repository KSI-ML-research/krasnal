import os
import re
import sys
from dataclasses import dataclass

from loguru import logger

from krasnal.inference.exceptions import NoLegalMovesError
from krasnal.tokens import BLACK_WON_ID, WHITE_WON_ID
from krasnal.uci_engine.go_params import GoParams, parse_go_rest
from krasnal.uci_engine.provider import ChessModelProvider, ModelProviderError


@dataclass(frozen=True, slots=True)
class CmdUci:
    """Handshake: client sent ``uci``."""


@dataclass(frozen=True, slots=True)
class CmdIsReady:
    """Client sent ``isready``."""


@dataclass(frozen=True, slots=True)
class CmdUciNewGame:
    """Client sent ``ucinewgame``."""


@dataclass(frozen=True, slots=True)
class CmdPosition:
    """``position …`` — move history extracted from the first ``moves`` token onward."""

    moves_uci: str


@dataclass(frozen=True, slots=True)
class CmdGo:
    """Client asked for a move (``go`` …); clock fields parsed when present."""

    params: GoParams


@dataclass(frozen=True, slots=True)
class CmdSetOption:
    """``setoption name … value …`` (Krasnal-specific options use ``Krasnal*`` names)."""

    name: str
    value: str


@dataclass(frozen=True, slots=True)
class CmdQuit:
    """Client sent ``quit``."""


@dataclass(frozen=True, slots=True)
class Unrecognized:
    """Line does not match a supported inbound command."""

    line: str


UciInbound = (
    CmdUci
    | CmdIsReady
    | CmdUciNewGame
    | CmdPosition
    | CmdGo
    | CmdSetOption
    | CmdQuit
    | Unrecognized
)

_SETOPTION_RE = re.compile(
    r"^setoption\s+name\s+(\S+)\s+value\s*(.*)$",
    re.IGNORECASE,
)


def parse_setoption_line(stripped_line: str) -> CmdSetOption | None:
    """Parse ``setoption name <id> value <x>`` (value may be empty)."""
    m = _SETOPTION_RE.match(stripped_line.strip())
    if m is None:
        return None
    return CmdSetOption(name=m.group(1), value=m.group(2).strip())


def parse_uci_line(stripped_line: str) -> UciInbound:
    """Parse one stripped, non-empty line from stdin into an inbound message."""
    if stripped_line == "uci":
        return CmdUci()
    if stripped_line == "isready":
        return CmdIsReady()
    if stripped_line == "ucinewgame":
        return CmdUciNewGame()
    if stripped_line == "quit":
        return CmdQuit()
    if stripped_line.startswith("position"):
        parts = stripped_line.split("moves", 1)
        moves = parts[1].strip() if len(parts) > 1 else ""
        return CmdPosition(moves_uci=moves)
    if stripped_line.startswith("go"):
        rest = stripped_line[2:].strip()
        return CmdGo(params=parse_go_rest(rest))
    so = parse_setoption_line(stripped_line)
    if so is not None:
        return so
    return Unrecognized(line=stripped_line)


class UCIParser:
    """
    Loop listening for input messages and sending
    the game state to the Provider to obtain a move.

    With ``lazy_start=True`` (default for ``python -m krasnal.uci_engine.run``), the heavy
    model load runs on the first ``uci`` line *after* replying with ``id`` / ``option`` /
    ``uciok``, so the subprocess stays alive for python-chess during long imports.

    Clocks: lichess-bot sends ``go wtime … btime …`` in milliseconds; these are
    forwarded to the model when ``use_time_conditioning`` is enabled.

    Optional ``setoption`` names (declare them in the ``uci`` handshake):

    - ``KrasnalWhiteElo`` / ``KrasnalBlackElo`` — integer rating (use ``-1`` for unknown).
    - ``KrasnalInitialSeconds`` / ``KrasnalIncrementSeconds`` — base clock and increment
      in seconds for the time-control bucket token (use ``-1`` for unknown).
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
        self.engine_side: str = "both"

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

    def _get_outcome_token(self) -> int:
        """Derive outcome token based on engine side and move count."""
        move_count = len(self.current_moves.split()) if self.current_moves else 0

        if self.engine_side == "white":
            return WHITE_WON_ID
        elif self.engine_side == "black":
            return BLACK_WON_ID
        else:
            if move_count % 2 == 0:
                return WHITE_WON_ID
            else:
                return BLACK_WON_ID

    def _send_engine_options(self) -> None:
        """Advertise optional metadata knobs (lichess-bot: ``engine: uci_options``)."""
        self._send("option name KrasnalWhiteElo type spin default -1 min -1 max 4000")
        self._send("option name KrasnalBlackElo type spin default -1 min -1 max 4000")
        self._send("option name KrasnalInitialSeconds type spin default -1 min -1 max 10800")
        self._send("option name KrasnalIncrementSeconds type spin default -1 min -1 max 600")

    def _send_info_chunks(self, message: str, *, chunk: int = 220) -> None:
        """Emit UCI ``info string`` lines, chunked (some GUIs truncate one long line)."""
        collapsed = message.replace("\r", " ").replace("\n", " | ")
        for i in range(0, len(collapsed), chunk):
            self._send(f"info string {collapsed[i : i + chunk]}")

    def _emit_bestmove_none(self, info_body: str) -> None:
        self._send_info_chunks(info_body)
        self._send("bestmove (none)")

    def _handle_cmd_go(self, params: GoParams) -> None:
        if self._startup_error is not None:
            logger.error("go failed (startup): {}", self._startup_error)
            self._emit_bestmove_none(f"krasnal-uci startup: {self._startup_error}")
            return
        if self.provider is None:
            logger.error("go failed: no provider")
            self._emit_bestmove_none("krasnal-uci internal error (no provider)")
            return

        self.provider.set_go_params(params)
        try:
            best = self.provider.get_best_move(self.current_moves)
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
        """React to one parsed inbound UCI message (``command`` must be stripped)."""
        msg = parse_uci_line(command)
        match msg:
            case CmdUci():
                self._ensure_provider()
                self._send(f"id name {self.engine_name}")
                self._send("id author KSI UWr")
                self._send_engine_options()
                self._send("uciok")
            case CmdIsReady():
                self._send("readyok")
            case CmdUciNewGame():
                self.current_moves = ""
                if self.provider is not None and self._startup_error is None:
                    self.provider.reset_session(self._get_outcome_token())
            case CmdPosition(moves_uci=moves_uci):
                self.current_moves = moves_uci
            case CmdSetOption(name=name, value=value):
                if self.provider is not None:
                    self.provider.apply_setoption(name, value)
            case CmdGo(params=params):
                self._handle_cmd_go(params)
            case CmdQuit():
                sys.exit(0)
            case Unrecognized():
                pass

    def _send(self, msg: str) -> None:
        print(msg)
        sys.stdout.flush()
