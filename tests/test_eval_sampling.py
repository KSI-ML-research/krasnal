import duckdb
import polars as pl

from krasnal.preprocess.eval_sampling import (
    EVAL_GAMES_PER_BIN,
    EVAL_MIN_CLOCK,
    maia_eval_sample_sql,
)


def _synthetic_games() -> pl.DataFrame:
    rows = []
    for bin_base in (1500, 1600):
        for game_idx in range(12_000):
            low_clock = game_idx < 100
            rows.append(
                {
                    "lichess_id": f"{bin_base}_{game_idx}",
                    "uci_moves": "e2e4",
                    "clocks_white": [20 if low_clock else 40],
                    "clocks_black": [40],
                    "white_rating": bin_base + 25,
                    "black_rating": bin_base + 75,
                }
            )
    rows.append(
        {
            "lichess_id": "mismatch",
            "uci_moves": "e2e4",
            "clocks_white": [40],
            "clocks_black": [40],
            "white_rating": 1500,
            "black_rating": 1700,
        }
    )
    return pl.DataFrame(rows)


def test_maia_eval_sample_sql_filters_and_caps_per_bin(tmp_path):
    seed = 7
    games_per_bin = 100
    path = tmp_path / "games.parquet"
    _synthetic_games().write_parquet(path)

    con = duckdb.connect()
    sql = maia_eval_sample_sql(
        f"SELECT * FROM '{path}'",
        seed=seed,
        games_per_bin=games_per_bin,
        min_clock=EVAL_MIN_CLOCK,
    )
    sampled = pl.from_pandas(con.execute(sql).df())

    assert "mismatch" not in set(sampled["lichess_id"])
    assert len(sampled) == 2 * games_per_bin
    bin_counts = (
        sampled.with_columns((pl.col("white_rating") // 100 * 100).alias("bin"))
        .group_by("bin")
        .len()
        .sort("bin")
    )
    assert bin_counts["len"].to_list() == [games_per_bin, games_per_bin]


def test_maia_eval_sample_sql_caps_per_bin(tmp_path):
    n = EVAL_GAMES_PER_BIN + 50
    path = tmp_path / "one_bin.parquet"
    pl.DataFrame(
        {
            "lichess_id": [f"g{i}" for i in range(n)],
            "clocks_white": [[40]] * n,
            "clocks_black": [[40]] * n,
            "white_rating": [1550] * n,
            "black_rating": [1550] * n,
        }
    ).write_parquet(path)
    con = duckdb.connect()
    inner = f"SELECT * FROM '{path}'"
    count = con.execute(f"SELECT count(*) FROM ({maia_eval_sample_sql(inner, seed=1)})").fetchone()[
        0
    ]
    assert count == EVAL_GAMES_PER_BIN
