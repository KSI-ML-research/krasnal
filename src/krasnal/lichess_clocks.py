from __future__ import annotations

import io
import re
import time
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import chess.pgn

_CLK_RE = re.compile(r"\[%clk\s+([^\]]+)\]")


@dataclass(frozen=True)
class LichessClockExtraction:
    lichess_id: str
    move_uci: list[str] | None
    move_clocks_seconds: list[float | None] | None
    moves_match: bool
    error: str | None = None


def _clock_to_seconds(clock_text: str) -> float:
    parts = clock_text.strip().split(":")
    if len(parts) == 2:
        minutes_text, seconds_text = parts
        hours = 0
    elif len(parts) == 3:
        hours_text, minutes_text, seconds_text = parts
        hours = int(hours_text)
    else:
        raise ValueError(f"Unsupported clock format: {clock_text!r}")

    minutes = int(minutes_text)
    seconds = float(seconds_text)
    return hours * 3600 + minutes * 60 + seconds


def _clock_seconds_from_comment(comment: str) -> float | None:
    match = _CLK_RE.search(comment)
    if match is None:
        return None
    return _clock_to_seconds(match.group(1))


def fetch_lichess_pgn(lichess_id: str, *, timeout: float = 30.0, max_retries: int = 3) -> str:
    params = urlencode(
        {
            "clocks": "true",
            "evals": "false",
            "opening": "false",
            "moves": "true",
            "tags": "true",
            "literate": "false",
        }
    )
    url = f"https://lichess.org/game/export/{lichess_id}?{params}"
    request = Request(url, headers={"Accept": "application/x-chess-pgn", "User-Agent": "krasnal"})

    for attempt in range(max_retries):
        try:
            with urlopen(request, timeout=timeout) as response:
                pgn_text = response.read().decode("utf-8")
            if not pgn_text.strip():
                raise ValueError(f"Empty PGN response for {lichess_id}")
            return pgn_text
        except HTTPError as exc:
            transient = exc.code in {429, 500, 502, 503, 504}
            if not transient or attempt + 1 >= max_retries:
                raise
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            sleep_seconds = float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
            time.sleep(sleep_seconds)
        except URLError:
            if attempt + 1 >= max_retries:
                raise
            time.sleep(2**attempt)

    raise RuntimeError(f"Failed to fetch PGN for {lichess_id} after {max_retries} retries")


def extract_uci_moves_and_clocks(pgn_text: str) -> tuple[list[str], list[float | None]]:
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        raise ValueError("Could not parse PGN")

    uci_moves: list[str] = []
    move_clocks_seconds: list[float | None] = []

    for node in game.mainline():
        move = node.move
        if move is None:
            continue
        uci_moves.append(move.uci())
        move_clocks_seconds.append(_clock_seconds_from_comment(node.comment or ""))

    if not uci_moves:
        raise ValueError("Parsed PGN has no moves")

    return uci_moves, move_clocks_seconds


def fetch_and_extract_clocks(
    lichess_id: str,
    expected_uci_moves: str | None,
    *,
    timeout: float = 30.0,
    max_retries: int = 3,
) -> LichessClockExtraction:
    try:
        pgn_text = fetch_lichess_pgn(lichess_id, timeout=timeout, max_retries=max_retries)
        move_uci, move_clocks_seconds = extract_uci_moves_and_clocks(pgn_text)
        moves_match = expected_uci_moves is None or move_uci == expected_uci_moves.split()
        return LichessClockExtraction(
            lichess_id=lichess_id,
            move_uci=move_uci,
            move_clocks_seconds=move_clocks_seconds if moves_match else None,
            moves_match=moves_match,
            error=None if moves_match else "PGN move sequence does not match Aix UCI moves",
        )
    except Exception as exc:  # noqa: BLE001
        return LichessClockExtraction(
            lichess_id=lichess_id,
            move_uci=None,
            move_clocks_seconds=None,
            moves_match=False,
            error=str(exc),
        )