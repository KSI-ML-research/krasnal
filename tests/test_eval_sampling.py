import duckdb
import polars as pl

from krasnal.preprocess.eval_sampling import EVAL_GAMES_PER_BIN, maia_eval_sample_sql


def _synthetic_games() -> pl.DataFrame:
    rows = []
    for bin_base in (1500, 1600):
        for game_idx in range(12_000):
            rows.append(
                {
                    "lichess_id": f"{bin_base}_{game_idx}",
                    "uci_moves": "e2e4",
                    "clocks_white": [20 if game_idx < 100 else 40],
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


def test_maia_eval_sample_sql_does_not_filter_whole_games_by_clock(tmp_path):
    path = tmp_path / "low_clock.parquet"
    pl.DataFrame(
        {
            "lichess_id": ["low_clock"],
            "clocks_white": [[10]],
            "clocks_black": [[40]],
            "white_rating": [1550],
            "black_rating": [1550],
        }
    ).write_parquet(path)

    con = duckdb.connect()
    inner = f"SELECT * FROM '{path}'"
    sampled = pl.from_pandas(
        con.execute(f"SELECT * FROM ({maia_eval_sample_sql(inner, seed=1)})").df()
    )

    assert sampled["lichess_id"].to_list() == ["low_clock"]


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


def test_maia_eval_sample_sql_respects_min_elo(tmp_path):
    path = tmp_path / "rating_range.parquet"
    pl.DataFrame(
        {
            "lichess_id": ["too_low", "eligible_low", "eligible_high", "high_elo"],
            "clocks_white": [[40]] * 4,
            "clocks_black": [[40]] * 4,
            "white_rating": [1050, 1150, 1950, 2250],
            "black_rating": [1050, 1150, 1950, 2250],
        }
    ).write_parquet(path)

    con = duckdb.connect()
    inner = f"SELECT * FROM '{path}'"
    sampled = pl.from_pandas(
        con.execute(f"SELECT * FROM ({maia_eval_sample_sql(inner, seed=1)})").df()
    )

    assert sampled["lichess_id"].to_list() == ["eligible_low", "eligible_high", "high_elo"]


def test_maia_eval_sample_sql_collapses_2200_plus_into_one_bin(tmp_path):
    path = tmp_path / "high_elo.parquet"
    pl.DataFrame(
        {
            "lichess_id": ["bin_2200", "bin_2400", "mismatch"],
            "clocks_white": [[40]] * 3,
            "clocks_black": [[40]] * 3,
            "white_rating": [2250, 2450, 2250],
            "black_rating": [2280, 2480, 2100],
        }
    ).write_parquet(path)

    con = duckdb.connect()
    inner = f"SELECT * FROM '{path}'"
    sampled = pl.from_pandas(
        con.execute(f"SELECT * FROM ({maia_eval_sample_sql(inner, seed=1, games_per_bin=10)})").df()
    )

    assert sampled["lichess_id"].to_list() == ["bin_2200", "bin_2400"]
