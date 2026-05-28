#!/usr/bin/env python3
"""
Download Aix-compatible Lichess database files and filter them with DuckDB + Aix.

Filters:
    - Both players >= min_elo (configured in download.yaml)
    - Time control >= min_time seconds base (5+0 and above)
    - Require evals only when require_evals is enabled
    - Normal termination only
    - Month range: ``start_month`` / ``end_month`` in ``download.yaml`` (``%clk`` clocks)

Output: Parquet files in data/1_filtered/

Usage:
    just download-games
    just download-games target_games=10000000
    just download-games min_elo=1800
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

configure_logging()
logging.getLogger("httpx").setLevel(logging.WARNING)

# ─── Configuration ───────────────────────────────────────────────────────────

HF_REPO = "thomasd1/aix-lichess-database"

OUTPUT_DIR = Path("data/1_filtered")

# If ``end_month`` is null, months run through this (HF catalog / default cap).
DEFAULT_END_MONTH = "2026-03"

SKIP_MONTHS = {"2020-12", "2021-01", "2020-07", "2016-12", "2019-12"}
EVAL_MONTH = "2019-12"


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

FILTER_QUERY = """
SELECT
    lichess_id,
    to_uci(movedata) AS uci_moves,
    clocks_white,
    clocks_black,
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
{evals_condition}
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


def filter_month(
    parquet_path: Path,
    month: str,
    output_dir: Path,
    con: duckdb.DuckDBPyConnection,
    min_elo: int,
    min_time: int,
    require_evals: bool,
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

    evals_condition = "AND evals IS NOT NULL" if require_evals else ""

    query = FILTER_QUERY.format(
        parquet_path=parquet_path,
        date_start=date_start,
        date_end=date_end,
        min_elo=min_elo,
        min_time=min_time,
        evals_condition=evals_condition,
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
    require_evals = cfg.get("require_evals", False)

    months = _resolve_months(cfg)
    if not months:
        logger.error("No months to process (check ``start_month`` / ``end_month`` in config).")
        sys.exit(1)

    logger.info(f"Target games: {target_games:,}")
    logger.info(f"Months to process: {len(months)} (first={months[0]}, last={months[-1]})")
    logger.info(f"Compression: {compression}")
    logger.info(f"Min Elo: {min_elo}")
    logger.info(f"Min time control: {min_time}s")
    logger.info(f"Require evals: {require_evals}")

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

        try:
            parquet_path = download_aix_file(month, compression)
        except Exception as e:
            logger.error(f"Failed to download {month}: {e}")
            continue

        try:
            count = filter_month(
                parquet_path, month, OUTPUT_DIR, con, min_elo, min_time, require_evals
            )
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

    # Always ensure eval month is available
    eval_output = OUTPUT_DIR / f"filtered_{EVAL_MONTH}.parquet"
    if not eval_output.exists():
        logger.info("Downloading eval month {}...", EVAL_MONTH)
        con2 = duckdb.connect()
        con2.execute("INSTALL aixchess FROM community")
        con2.execute("LOAD aixchess")
        try:
            eval_parquet = download_aix_file(EVAL_MONTH, compression)
            filter_month(
                eval_parquet,
                EVAL_MONTH,
                OUTPUT_DIR,
                con2,
                min_elo,
                min_time,
                require_evals,
            )
        except Exception as e:
            logger.error("Failed to download/filter eval month {}: {}", EVAL_MONTH, e)
        con2.close()
    else:
        logger.info("Eval month {} already present: {}", EVAL_MONTH, eval_output)


if __name__ == "__main__":
    main()
