"""UCI ``go`` line clock fields (lichess-bot sends ``wtime`` / ``btime`` in milliseconds)."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GoParams:
    wtime_ms: int | None = None
    btime_ms: int | None = None
    winc_ms: int | None = None
    binc_ms: int | None = None
    movetime_ms: int | None = None


_INT_GO_KEYS = frozenset(
    {
        "wtime",
        "btime",
        "winc",
        "binc",
        "movetime",
        "depth",
        "nodes",
        "movestogo",
    }
)


def parse_go_rest(rest: str) -> GoParams:
    """Parse tokens after ``go`` into structured clock fields (other flags ignored)."""
    parts = rest.split()
    kw: dict[str, int] = {}
    i = 0
    while i < len(parts):
        key = parts[i]
        if key in ("infinite", "ponder"):
            i += 1
            continue
        if key == "searchmoves":
            i += 1
            while (
                i < len(parts)
                and parts[i] not in _INT_GO_KEYS
                and parts[i]
                not in (
                    "infinite",
                    "ponder",
                    "searchmoves",
                )
            ):
                i += 1
            continue
        if key in _INT_GO_KEYS and i + 1 < len(parts):
            with suppress(ValueError):
                kw[key] = int(parts[i + 1])
            i += 2
            continue
        i += 1

    return GoParams(
        wtime_ms=kw.get("wtime"),
        btime_ms=kw.get("btime"),
        winc_ms=kw.get("winc"),
        binc_ms=kw.get("binc"),
        movetime_ms=kw.get("movetime"),
    )


def uci_ms_to_clock_seconds(ms: int) -> int:
    """Convert UCI clock milliseconds to whole seconds (clamped, avoids sentinel collision)."""
    s = max(0, ms) // 1000
    return min(s, 2**31 - 2)
