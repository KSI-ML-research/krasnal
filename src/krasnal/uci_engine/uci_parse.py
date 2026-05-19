"""Parse a single non-empty UCI line into a structured inbound message."""

from __future__ import annotations

import re
from dataclasses import dataclass

from krasnal.uci_engine.go_params import GoParams, parse_go_rest

__all__ = [
    "CmdGo",
    "CmdIsReady",
    "CmdPosition",
    "CmdQuit",
    "CmdSetOption",
    "CmdUci",
    "CmdUciNewGame",
    "UciInbound",
    "Unrecognized",
    "parse_uci_line",
]


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
