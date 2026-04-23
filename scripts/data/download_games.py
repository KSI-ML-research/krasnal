#!/usr/bin/env python3
"""
Download Aix-compatible Lichess database files and filter them with DuckDB + Aix.

Filters:
    - Both players >= min_elo (configured in download.yaml)
    - Time control >= min_time seconds base (5+0 and above)
    - Games must have Stockfish evals
    - Normal termination only
    - Date range: 2013-01 to 2026-03 (configurable)

Output: Parquet files in data/1_filtered/

Usage:
    just download-games
    just download-games target_games=10000000
    just download-games min_elo=1800
"""

import os
import time
from pathlib import Path

import duckdb
import hydra
from huggingface_hub import hf_hub_download
from loguru import logger
from omegaconf import DictConfig

# ─── Configuration ───────────────────────────────────────────────────────────

HF_REPO = "thomasd1/aix-lichess-database"
COMPRESSION = "high"

CACHE_DIR = Path("data/0_aix_downloads")
OUTPUT_DIR = Path("data/1_filtered")

DEFAULT_MONTHS = [
    f"{y}-{m:02d}"
    for y in range(2013, 2027)
    for m in range(1, 13)
    if (y, m) >= (2013, 1) and (y, m) <= (2026, 3)
]

SKIP_MONTHS = {"2020-12", "2021-01", "2020-07", "2016-12"}

# ─── DuckDB + Aix Query ─────────────────────────────────────────────────────

FILTER_QUERY = """
SELECT
    lichess_id,
    to_uci(movedata) AS uci_moves,
    list_eval_to_centipawns(evals) AS evals_cp,
    evals AS evals_raw,
    move_details(movedata).apply(m -> m.is_check) AS is_check,
    move_details(movedata).apply(m -> m.capture IS NOT NULL AND m.capture != '') AS is_capture,
    move_details(movedata).apply(m -> m.role) AS piece_moved,
    move_details(movedata).apply(m -> m.promotion) AS promotion,
    move_details(movedata).apply(m -> m.is_en_passant) AS is_en_passant,
    white_rating,
    black_rating,
    result,
    CASE
        WHEN move_details_ext_at(movedata, -1).is_checkmate THEN 'mate'
        WHEN move_details_ext_at(movedata, -1).is_stalemate THEN 'stalemate'
        WHEN result = '1-0' OR result = '0-1' THEN 'resignation'
        WHEN result = '1/2-1/2' THEN 'agreement'
        ELSE 'unknown'
    END AS game_end_reason,
    time_initial,
    time_increment,
    utc_timestamp,
    opening,
    eco,
    ply_count,
    fen_at_position(movedata, 0) AS fen
FROM '{parquet_path}'
WHERE white_rating >= {min_elo}
  AND black_rating >= {min_elo}
  AND time_initial >= {min_time}
  AND termination = 'Normal'
  AND evals IS NOT NULL
  AND result IN ('1-0', '0-1', '1/2-1/2')
  AND utc_timestamp >= '{date_start}'
  AND utc_timestamp < '{date_end}'
"""


def download_aix_file(month: str, compression: str, cache_dir: Path) -> Path:
    """Download Aix parquet file from HuggingFace."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = f"aix_lichess_{month}_{compression}.parquet"
    downloaded_path = cache_dir / f"{compression}_compression" / filename
    if downloaded_path.exists():
        return downloaded_path

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


def filter_month(
    parquet_path: Path,
    month: str,
    output_dir: Path,
    con: duckdb.DuckDBPyConnection,
    min_elo: int,
    min_time: int,
) -> int:
    """
    Filter one month of Aix games and save to Parquet.

    Returns the number of games that passed filters.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"filtered_{month}.parquet"

    if output_path.exists():
        logger.info(f"Output already exists: {output_path}, skipping")
        result = con.execute(f"SELECT COUNT(*) FROM '{output_path}'").fetchone()
        return result[0]

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

    query = FILTER_QUERY.format(
        parquet_path=parquet_path,
        date_start=date_start,
        date_end=date_end,
        min_elo=min_elo,
        min_time=min_time,
    )

    logger.info(f"Filtering {month}...")
    start = time.time()

    con.execute(f"""
        COPY (
            {query}
        ) TO '{output_path}' (FORMAT PARQUET)
    """)

    elapsed = time.time() - start
    result = con.execute(f"SELECT COUNT(*) FROM '{output_path}'").fetchone()
    count = result[0]

    logger.info(f"  {count:,} games in {elapsed:.0f}s -> {output_path.name}")
    return count


@hydra.main(version_base=None, config_path="../../config", config_name="download")
def main(cfg: DictConfig) -> None:
    logger.info("=" * 60)
    logger.info("Aix Lichess Database Filter")
    logger.info("=" * 60)

    min_elo = cfg.min_elo
    min_time = cfg.min_time
    target_games = cfg.target_games
    compression = cfg.compression

    months = cfg.months if cfg.months else DEFAULT_MONTHS

    logger.info(f"Target games: {target_games:,}")
    logger.info(f"Months to process: {len(months)}")
    logger.info(f"Compression: {compression}")
    logger.info(f"Min Elo: {min_elo}")
    logger.info(f"Min time control: {min_time}s")

    hf_transfer_status = (
        "enabled" if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1" else "disabled"
    )
    logger.info(f"hf_transfer: {hf_transfer_status}")
    logger.info("")

    con = duckdb.connect()
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

        filename = f"aix_lichess_{month}_{compression}.parquet"
        parquet_path = CACHE_DIR / f"{compression}_compression" / filename

        if parquet_path.exists():
            logger.info(f"Using cached file: {parquet_path.name}")
        else:
            try:
                parquet_path = download_aix_file(month, compression, CACHE_DIR)
            except Exception as e:
                logger.error(f"Failed to download {month}: {e}")
                continue

        try:
            count = filter_month(parquet_path, month, OUTPUT_DIR, con, min_elo, min_time)
            total_games += count
        except Exception as e:
            logger.error(f"Failed to filter {month}: {e}")
            continue

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"FINAL: {total_games:,} games collected")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    if OUTPUT_DIR.exists():
        total_size = sum(f.stat().st_size for f in OUTPUT_DIR.glob("*.parquet")) / (
            1024 * 1024 * 1024
        )
        logger.info(f"Total output size: {total_size:.2f} GB")
    logger.info("=" * 60)

    con.close()


if __name__ == "__main__":
    main()
