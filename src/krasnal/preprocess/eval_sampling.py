"""Maia-style eval holdout game sampling (same Elo bin and rating range)."""

from __future__ import annotations

EVAL_MONTH = "2019-12"
EVAL_GAMES_PER_BIN = 10_000


def maia_eval_sample_sql(
    inner_query: str,
    *,
    seed: int,
    games_per_bin: int = EVAL_GAMES_PER_BIN,
    min_elo: int = 1100,
    max_elo: int = 1999,
) -> str:
    """Wrap a filtered-games SELECT; keep at most ``games_per_bin`` per 100-point Elo bin."""
    return f"""
WITH games AS (
{inner_query}
),
eligible AS (
    SELECT
        games.*,
        (white_rating // 100) * 100 AS elo_bin
    FROM games
    WHERE len(clocks_white) > 0
      AND len(clocks_black) > 0
      AND (white_rating // 100) * 100 = (black_rating // 100) * 100
      AND (white_rating // 100) * 100 >= {min_elo}
      AND (white_rating // 100) * 100 <= {(max_elo // 100) * 100}
),
ranked AS (
    SELECT
        *,
        row_number() OVER (
            PARTITION BY elo_bin
            ORDER BY hash(lichess_id, {seed})
        ) AS _rn
    FROM eligible
)
SELECT * EXCLUDE (elo_bin, _rn)
FROM ranked
WHERE _rn <= {games_per_bin}
"""
