"""Maia-style eval holdout game sampling (same Elo bin and rating range)."""

from __future__ import annotations

EVAL_MONTH = "2019-12"
EVAL_GAMES_PER_BIN = 10_000


def _elo_bin_sql(rating_col: str) -> str:
    """Match ``get_elo_bucket`` bins: 100-point bands below 2200, one bin for 2200+."""
    return f"CASE WHEN {rating_col} >= 2200 THEN 2200 ELSE ({rating_col} // 100) * 100 END"


def maia_eval_sample_sql(
    inner_query: str,
    *,
    seed: int,
    games_per_bin: int = EVAL_GAMES_PER_BIN,
    min_elo: int = 1100,
) -> str:
    """Wrap a filtered-games SELECT; keep at most ``games_per_bin`` per Elo bin."""
    white_bin = _elo_bin_sql("white_rating")
    return f"""
WITH games AS (
{inner_query}
),
eligible AS (
    SELECT
        games.*,
        {white_bin} AS elo_bin
    FROM games
    WHERE len(clocks_white) > 0
      AND len(clocks_black) > 0
      AND {white_bin} = {_elo_bin_sql("black_rating")}
      AND {white_bin} >= {(min_elo // 100) * 100}
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
