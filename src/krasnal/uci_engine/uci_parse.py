"""Parse a single non-empty UCI line into a structured inbound message."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "CmdGo",
    "CmdIsReady",
    "CmdPosition",
    "CmdQuit",
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
    """Client asked for a move (``go`` … options ignored by this engine)."""

    rest: str


@dataclass(frozen=True, slots=True)
class CmdQuit:
    """Client sent ``quit``."""


@dataclass(frozen=True, slots=True)
class Unrecognized:
    """Line does not match a supported inbound command."""

    line: str


UciInbound = CmdUci | CmdIsReady | CmdUciNewGame | CmdPosition | CmdGo | CmdQuit | Unrecognized


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
        return CmdGo(rest=stripped_line[2:].strip())
    return Unrecognized(line=stripped_line)
