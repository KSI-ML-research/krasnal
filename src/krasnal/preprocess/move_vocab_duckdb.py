"""Build move vocabulary from filtered parquet via DuckDB (no Python corpus scan)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from krasnal.tokens import save_move_vocab

if TYPE_CHECKING:
    import duckdb

_AIX_ROLE_CANONICAL = """
CASE lower(nullif(trim(list_extract(piece_moved, i)), ''))
    WHEN 'pawn' THEN 'pawn' WHEN 'p' THEN 'pawn'
    WHEN 'knight' THEN 'knight' WHEN 'n' THEN 'knight'
    WHEN 'bishop' THEN 'bishop' WHEN 'b' THEN 'bishop'
    WHEN 'rook' THEN 'rook' WHEN 'r' THEN 'rook'
    WHEN 'queen' THEN 'queen' WHEN 'q' THEN 'queen'
    WHEN 'king' THEN 'king' WHEN 'k' THEN 'king'
    ELSE lower(nullif(trim(list_extract(piece_moved, i)), ''))
END
"""


def move_key_list_sql(*, piece_aware_moves: bool, side_prefixed_moves: bool) -> str:
    """SQL list expression: move keys for one game row (``uci_moves``, ``piece_moved``)."""
    uci = "list_extract(string_split(uci_moves, ' '), i)"
    key = uci
    if piece_aware_moves:
        key = f"({_AIX_ROLE_CANONICAL}) || ':' || {uci}"
    if side_prefixed_moves:
        key = f"CASE WHEN (i - 1) % 2 = 0 THEN 'w:' ELSE 'b:' END || {key}"
    return f"list_transform(range(1, len(piece_moved) + 1), i -> {key})"


def build_move_vocab_from_filtered_parquet(
    con: duckdb.DuckDBPyConnection,
    filtered_glob: str,
    output_path: Path,
    *,
    piece_aware_moves: bool,
    side_prefixed_moves: bool,
) -> dict:
    """Collect distinct move keys over filtered shards and write ``move_vocab.json``."""
    move_keys = move_key_list_sql(
        piece_aware_moves=piece_aware_moves,
        side_prefixed_moves=side_prefixed_moves,
    )
    rows = con.execute(
        f"""
        SELECT DISTINCT move_key
        FROM (
            SELECT unnest({move_keys}) AS move_key
            FROM read_parquet('{filtered_glob}')
            WHERE uci_moves IS NOT NULL
              AND len(piece_moved) > 0
        )
        WHERE move_key IS NOT NULL AND move_key != ''
        ORDER BY move_key
        """
    ).fetchall()
    move_key_set = {row[0] for row in rows}
    artifact = save_move_vocab(
        output_path,
        move_key_set,
        piece_aware_moves=piece_aware_moves,
        side_prefixed_moves=side_prefixed_moves,
    )
    return artifact
