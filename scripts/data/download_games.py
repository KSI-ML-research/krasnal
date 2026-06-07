#!/usr/bin/env python3
"""
Download Aix-compatible Lichess database files and filter them with DuckDB + Aix.

Filters:
    - Both players within min_elo/max_elo (configured in download.yaml)
    - Lichess estimated duration >= min_estimated_duration seconds
    - Normal termination only
    - Month range: ``start_month`` / ``end_month`` in ``download.yaml`` (``%clk`` clocks)

Output: Parquet files in data/1_filtered/

Usage:
    just download-games
    just download-games target_games=10000000
    just download-games min_elo=1800
    just download-games min_estimated_duration=480
"""

import logging
import os
import sys
import time
from pathlib import Path

import duckdb
import hydra
from huggingface_hub import hf_hub_download
from loguru import logger
from omegaconf import DictConfig

from krasnal import configure_logging
from krasnal.config import MOVE_VOCAB_PATH, RAW_UCI_DIR
from krasnal.preprocess.eval_sampling import (
    EVAL_GAMES_PER_BIN,
    EVAL_MONTH,
    maia_eval_sample_sql,
)
from krasnal.preprocess.move_vocab_duckdb import build_move_vocab_from_filtered_parquet

configure_logging()
logging.getLogger("httpx").setLevel(logging.WARNING)

# ─── Configuration ───────────────────────────────────────────────────────────

HF_REPO = "thomasd1/aix-lichess-database"

# If ``end_month`` is null, months run through this (HF catalog / default cap).
DEFAULT_END_MONTH = "2026-03"

SKIP_MONTHS = {"2020-12", "2021-01", "2020-07", "2016-12", EVAL_MONTH}


def _parse_month(label: str) -> tuple[int, int]:
    year_s, mon_s = label.split("-", 1)
    return int(year_s), int(mon_s)


def _expand_month_range(start: str, end: str) -> list[str]:
    """Inclusive ``start`` … ``end`` as YYYY-MM strings."""
    y, mo = _parse_month(start)
    ey, emo = _parse_month(end)
    if (y, mo) > (ey, emo):
        return []
    out: list[str] = []
    while (y, mo) <= (ey, emo):
        out.append(f"{y}-{mo:02d}")
        mo += 1
        if mo > 12:
            mo = 1
            y += 1
    return out


def _resolve_months(cfg: DictConfig) -> list[str]:
    start = str(cfg.start_month)
    end_raw = cfg.get("end_month")
    end = str(end_raw) if end_raw is not None else DEFAULT_END_MONTH
    return _expand_month_range(start, end)


# ─── DuckDB + Aix Query ─────────────────────────────────────────────────────
# ``move_details`` (Aix) exposes per-move ``capture`` / ``promotion``; opponent
# material after each ply is accumulated in SQL (see aix docs/functions.md).

_STARTING_SIDE_MATERIAL = 39

_AIX_CAPTURE_MATERIAL = """CASE lower(nullif(trim(u.m.capture), ''))
    WHEN 'pawn' THEN 1 WHEN 'p' THEN 1
    WHEN 'knight' THEN 3 WHEN 'n' THEN 3
    WHEN 'bishop' THEN 3 WHEN 'b' THEN 3
    WHEN 'rook' THEN 5 WHEN 'r' THEN 5
    WHEN 'queen' THEN 9 WHEN 'q' THEN 9
    WHEN 'king' THEN 0 WHEN 'k' THEN 0
    ELSE 0
END"""

_AIX_PROMOTION_MATERIAL = """CASE lower(nullif(trim(u.m.promotion), ''))
    WHEN 'pawn' THEN 1 WHEN 'p' THEN 1
    WHEN 'knight' THEN 3 WHEN 'n' THEN 3
    WHEN 'bishop' THEN 3 WHEN 'b' THEN 3
    WHEN 'rook' THEN 5 WHEN 'r' THEN 5
    WHEN 'queen' THEN 9 WHEN 'q' THEN 9
    WHEN 'king' THEN 0 WHEN 'k' THEN 0
    ELSE 0
END"""

FILTER_QUERY = f"""
WITH eligible AS (
    SELECT *
    FROM '{{parquet_path}}'
    WHERE white_rating >= {{min_elo}}
      AND black_rating >= {{min_elo}}
      {{max_elo_condition}}
      AND time_initial::INTEGER + 40 * time_increment::INTEGER >= {{min_estimated_duration}}
      AND termination = 'Normal'
      AND result IN ('1-0', '0-1', '1/2-1/2')
      AND utc_timestamp >= '{{date_start}}'
      AND utc_timestamp < '{{date_end}}'
),
filtered AS (
{{filtered_query}}
),
decoded AS (
    SELECT filtered.*, move_details(movedata) AS md
    FROM filtered
),
per_ply AS (
    SELECT
        decoded.lichess_id,
        (u.ply_idx - 1)::INTEGER AS ply,
        CASE
            WHEN (u.ply_idx - 1) % 2 = 1 THEN -({_AIX_CAPTURE_MATERIAL})
            WHEN nullif(trim(u.m.promotion), '') IS NOT NULL
                THEN ({_AIX_PROMOTION_MATERIAL}) - 1
            ELSE 0
        END AS delta_white,
        CASE
            WHEN (u.ply_idx - 1) % 2 = 0 THEN -({_AIX_CAPTURE_MATERIAL})
            WHEN nullif(trim(u.m.promotion), '') IS NOT NULL
                THEN ({_AIX_PROMOTION_MATERIAL}) - 1
            ELSE 0
        END AS delta_black
    FROM decoded,
    UNNEST(decoded.md) WITH ORDINALITY AS u(m, ply_idx)
),
running AS (
    SELECT
        lichess_id,
        ply,
        {_STARTING_SIDE_MATERIAL}
            + sum(delta_white) OVER (
                PARTITION BY lichess_id ORDER BY ply ROWS UNBOUNDED PRECEDING
            ) AS white_mat,
        {_STARTING_SIDE_MATERIAL}
            + sum(delta_black) OVER (
                PARTITION BY lichess_id ORDER BY ply ROWS UNBOUNDED PRECEDING
            ) AS black_mat
    FROM per_ply
),
opponent_material_by_game AS (
    SELECT
        lichess_id,
        list(
            CASE
                WHEN ply % 2 = 0 THEN black_mat::UTINYINT
                ELSE white_mat::UTINYINT
            END
            ORDER BY ply
        ) AS opponent_material
    FROM running
    GROUP BY lichess_id
)
SELECT
    decoded.lichess_id,
    to_uci(decoded.movedata) AS uci_moves,
    decoded.clocks_white,
    decoded.clocks_black,
    decoded.md.apply(m -> m.is_check) AS is_check,
    decoded.md.apply(m -> m.role) AS piece_moved,
    om.opponent_material,
    decoded.white_rating,
    decoded.black_rating,
    decoded.result,
    CASE
        WHEN move_details_ext_at(decoded.movedata, -1).is_checkmate THEN 'mate'
        WHEN move_details_ext_at(decoded.movedata, -1).is_stalemate THEN 'stalemate'
        WHEN decoded.result = '1-0' OR decoded.result = '0-1' THEN 'resignation'
        WHEN decoded.result = '1/2-1/2' THEN 'agreement'
        ELSE 'unknown'
    END AS game_end_reason,
    decoded.time_initial,
    decoded.time_increment,
    decoded.utc_timestamp,
    decoded.opening,
    decoded.eco,
    decoded.ply_count
FROM decoded
INNER JOIN opponent_material_by_game om USING (lichess_id)
"""


def download_aix_file(month: str, compression: str) -> Path:
    """
    Download Aix parquet file from HuggingFace.

    Files are cached in HF's default cache (~/.cache/huggingface/).
    """
    filename = f"high_compression/aix_lichess_{month}_{compression}.parquet"

    logger.info(f"Downloading {filename}...")
    start = time.time()
    hf_token = os.environ.get("HF_TOKEN")
    downloaded_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=filename,
        repo_type="dataset",
        token=hf_token,
    )
    elapsed = time.time() - start
    size_mb = Path(downloaded_path).stat().st_size / (1024 * 1024)
    speed = size_mb / elapsed if elapsed > 0 else 0
    logger.info(f"Downloaded {size_mb:.0f} MB in {elapsed:.0f}s ({speed:.1f} MB/s)")
    return Path(downloaded_path)


def configure_duckdb_connection(con: duckdb.DuckDBPyConnection) -> None:
    memory_limit = os.environ.get("DUCKDB_MEMORY_LIMIT", "16GB")
    con.execute(f"SET memory_limit = '{memory_limit}'")
    con.execute("SET preserve_insertion_order = false")
    logger.info("DuckDB memory_limit={}, preserve_insertion_order=false", memory_limit)


def copy_query_to_parquet(con: duckdb.DuckDBPyConnection, query: str, output_path: Path) -> int:
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    tmp_path.unlink(missing_ok=True)
    con.execute(f"""
        COPY (
            {query}
        ) TO '{tmp_path}' (FORMAT PARQUET)
    """)
    count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{tmp_path}')").fetchone()[0]
    if count == 0:
        tmp_path.unlink(missing_ok=True)
        return 0
    tmp_path.replace(output_path)
    return count


def filtered_month_query(
    parquet_path: Path,
    date_start: str,
    date_end: str,
    min_elo: int,
    max_elo: int | None,
    min_estimated_duration: int,
    filtered_query: str,
) -> str:
    max_elo_condition = (
        f"AND white_rating <= {max_elo}\n      AND black_rating <= {max_elo}"
        if max_elo is not None
        else ""
    )
    return FILTER_QUERY.format(
        parquet_path=parquet_path,
        date_start=date_start,
        date_end=date_end,
        min_elo=min_elo,
        max_elo_condition=max_elo_condition,
        min_estimated_duration=min_estimated_duration,
        filtered_query=filtered_query,
    )


def filter_month(
    parquet_path: Path,
    month: str,
    output_dir: Path,
    con: duckdb.DuckDBPyConnection,
    min_elo: int,
    max_elo: int | None,
    min_estimated_duration: int,
    *,
    max_games: int,
    chunk_games: int,
    eval_sample: bool = False,
    eval_seed: int = 0,
    eval_games_per_bin: int = EVAL_GAMES_PER_BIN,
) -> int:
    """
    Filter one month of Aix games and save to Parquet.

    Returns the number of games that passed filters.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    year, mon = month.split("-")
    date_start = f"{year}-{mon}-01"
    next_mon_int = int(mon) + 1
    if next_mon_int > 12:
        next_year = int(year) + 1
        next_mon = "01"
    else:
        next_year = int(year)
        next_mon = f"{next_mon_int:02d}"
    date_end = f"{next_year}-{next_mon}-01"

    if eval_sample:
        output_path = output_dir / f"filtered_{month}.parquet"
        if output_path.exists():
            logger.info(f"Output already exists: {output_path}, skipping")
            return con.execute(f"SELECT COUNT(*) FROM '{output_path}'").fetchone()[0]
        query = filtered_month_query(
            parquet_path,
            date_start,
            date_end,
            min_elo,
            max_elo,
            min_estimated_duration,
            maia_eval_sample_sql(
                "SELECT * FROM eligible",
                seed=eval_seed,
                games_per_bin=eval_games_per_bin,
                min_elo=min_elo,
                max_elo=max_elo,
            ),
        )
        logger.info(
            "Filtering {} with Maia eval sampling (seed={}, up to {} games/bin)",
            month,
            eval_seed,
            eval_games_per_bin,
        )
        start = time.time()
        count = copy_query_to_parquet(con, query, output_path)
        logger.info(
            "  {:,} games after eval sampling in {:.0f}s -> {}",
            count,
            time.time() - start,
            output_path.name,
        )
        return count

    total = 0
    chunk_idx = 0
    logger.info(f"Filtering {month}...")
    while total < max_games:
        chunk_start = chunk_idx * chunk_games
        chunk_end = min(chunk_start + chunk_games, max_games)
        output_path = output_dir / f"filtered_{month}_{chunk_idx:04d}.parquet"
        if output_path.exists():
            count = con.execute(f"SELECT COUNT(*) FROM '{output_path}'").fetchone()[0]
            logger.info("  {} exists, skipping ({:,} games)", output_path.name, count)
        else:
            filtered_query = f"""
    SELECT * EXCLUDE (_rn)
    FROM (
        SELECT *, row_number() OVER (ORDER BY hash(lichess_id, {eval_seed})) AS _rn
        FROM eligible
    )
    WHERE _rn > {chunk_start} AND _rn <= {chunk_end}
"""
            query = filtered_month_query(
                parquet_path,
                date_start,
                date_end,
                min_elo,
                max_elo,
                min_estimated_duration,
                filtered_query,
            )
            start = time.time()
            count = copy_query_to_parquet(con, query, output_path)
            logger.info(
                "  {:,} games in {:.0f}s -> {}",
                count,
                time.time() - start,
                output_path.name,
            )
        total += count
        if count == 0:
            break
        chunk_idx += 1
    return total


def _eval_sampling_params(cfg: DictConfig) -> tuple[str, int, int, int, int]:
    eval_cfg = cfg.get("eval", {})
    month = str(eval_cfg.get("month", EVAL_MONTH))
    games_per_bin = int(eval_cfg.get("games_per_bin", EVAL_GAMES_PER_BIN))
    min_elo = int(eval_cfg.get("min_elo", 1100))
    max_elo = int(eval_cfg.get("max_elo", 1999))
    seed = int(cfg.get("seed", 0))
    return month, seed, games_per_bin, min_elo, max_elo


@hydra.main(version_base=None, config_path="../../config", config_name="download")
def main(cfg: DictConfig) -> None:
    logger.info("=" * 60)
    logger.info("Aix Lichess Database Filter")
    logger.info("=" * 60)

    min_elo = cfg.min_elo
    max_elo = cfg.get("max_elo")
    min_estimated_duration = cfg.min_estimated_duration
    target_games = cfg.target_games
    chunk_games = cfg.filter_chunk_games
    compression = cfg.compression

    months = _resolve_months(cfg)
    if not months:
        logger.error("No months to process (check ``start_month`` / ``end_month`` in config).")
        sys.exit(1)

    logger.info(f"Target games: {target_games:,}")
    logger.info(f"Months to process: {len(months)} (first={months[0]}, last={months[-1]})")
    logger.info(f"Compression: {compression}")
    elo_range = f"{min_elo}+" if max_elo is None else f"{min_elo}-{max_elo}"
    logger.info(f"Training Elo range: {elo_range}")
    logger.info(f"Min estimated duration: {min_estimated_duration}s")
    logger.info(f"Filter chunk games: {chunk_games:,}")
    eval_month, eval_seed, eval_games_per_bin, eval_min_elo, eval_max_elo = _eval_sampling_params(
        cfg
    )
    logger.info(
        "Eval holdout: month={}, up to {} games/bin, seed={}, Elo={}-{}",
        eval_month,
        eval_games_per_bin,
        eval_seed,
        eval_min_elo,
        eval_max_elo,
    )

    hf_transfer_status = (
        "enabled" if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1" else "disabled"
    )
    logger.info(f"hf_transfer: {hf_transfer_status}")
    logger.info("")

    con = duckdb.connect()
    configure_duckdb_connection(con)
    logger.info("Installing Aix DuckDB extension...")
    con.execute("INSTALL aixchess FROM community")
    con.execute("LOAD aixchess")
    logger.info("Aix extension loaded successfully")

    total_games = 0

    for month in months:
        if month in SKIP_MONTHS:
            logger.warning(f"Skipping {month} (known eval issues)")
            continue

        if total_games >= target_games:
            logger.info(f"Target reached ({total_games:,} games). Stopping.")
            break

        try:
            parquet_path = download_aix_file(month, compression)
        except Exception as e:
            logger.error(f"Failed to download {month}: {e}")
            continue

        try:
            month_min_elo = eval_min_elo if month == eval_month else min_elo
            month_max_elo = eval_max_elo if month == eval_month else max_elo
            count = filter_month(
                parquet_path,
                month,
                RAW_UCI_DIR,
                con,
                month_min_elo,
                month_max_elo,
                min_estimated_duration,
                max_games=target_games - total_games,
                chunk_games=chunk_games,
                eval_sample=month == eval_month,
                eval_seed=eval_seed,
                eval_games_per_bin=eval_games_per_bin,
            )
            total_games += count
        except Exception as e:
            logger.error(f"Failed to filter {month}: {e}")
            continue

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"FINAL: {total_games:,} games collected")
    logger.info(f"Output directory: {RAW_UCI_DIR}")
    if RAW_UCI_DIR.exists():
        total_size = sum(f.stat().st_size for f in RAW_UCI_DIR.glob("*.parquet")) / (
            1024 * 1024 * 1024
        )
        logger.info(f"Total output size: {total_size:.2f} GB")
    logger.info("=" * 60)

    con.close()

    # Always ensure eval holdout month is available (Maia-style subsample at filter time).
    eval_output = RAW_UCI_DIR / f"filtered_{eval_month}.parquet"
    if not eval_output.exists():
        logger.info("Downloading eval month {}...", eval_month)
        con2 = duckdb.connect()
        configure_duckdb_connection(con2)
        con2.execute("INSTALL aixchess FROM community")
        con2.execute("LOAD aixchess")
        try:
            eval_parquet = download_aix_file(eval_month, compression)
            filter_month(
                eval_parquet,
                eval_month,
                RAW_UCI_DIR,
                con2,
                eval_min_elo,
                eval_max_elo,
                min_estimated_duration,
                max_games=eval_games_per_bin,
                chunk_games=chunk_games,
                eval_sample=True,
                eval_seed=eval_seed,
                eval_games_per_bin=eval_games_per_bin,
            )
        except Exception as e:
            logger.error("Failed to download/filter eval month {}: {}", eval_month, e)
        con2.close()
    else:
        logger.info("Eval month {} already present: {}", eval_month, eval_output)

    _build_move_vocab_from_filtered(cfg)


def _build_move_vocab_from_filtered(cfg: DictConfig) -> None:
    piece_aware_moves = bool(cfg.get("piece_aware_moves", False))
    side_prefixed_moves = bool(cfg.get("side_prefixed_moves", True))
    filtered_files = sorted(RAW_UCI_DIR.glob("filtered_*.parquet"))
    if not filtered_files:
        logger.warning("No filtered parquet files found; skipping move vocabulary build")
        return

    filtered_glob = str(RAW_UCI_DIR / "filtered_*.parquet")
    logger.info(
        "Building move vocabulary from {} filtered shards (piece_aware={}, side_prefixed={})",
        len(filtered_files),
        piece_aware_moves,
        side_prefixed_moves,
    )
    con = duckdb.connect()
    try:
        artifact = build_move_vocab_from_filtered_parquet(
            con,
            filtered_glob,
            MOVE_VOCAB_PATH,
            piece_aware_moves=piece_aware_moves,
            side_prefixed_moves=side_prefixed_moves,
        )
    finally:
        con.close()
    logger.info(
        "Wrote {} move keys to {}",
        artifact["manifest"]["vocab_size"],
        MOVE_VOCAB_PATH,
    )


if __name__ == "__main__":
    main()
