"""Augment Aix-filtered parquet files with PGN clock annotations from Lichess."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import hydra
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
from omegaconf import DictConfig
from tqdm.auto import tqdm

from krasnal.lichess_clocks import fetch_and_extract_clocks


def _augment_parquet_file(
    input_path: Path,
    output_path: Path,
    *,
    workers: int,
    batch_size: int,
    timeout: float,
    max_retries: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    parquet_file = pq.ParquetFile(input_path)
    total_rows = parquet_file.metadata.num_rows if parquet_file.metadata is not None else None
    writer: pq.ParquetWriter | None = None

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        progress = tqdm(total=total_rows, desc=input_path.name, unit="row")
        try:
            for batch in parquet_file.iter_batches(batch_size=batch_size):
                table = pa.Table.from_batches([batch])
                if "lichess_id" not in table.column_names or "uci_moves" not in table.column_names:
                    raise ValueError(f"{input_path} is missing lichess_id or uci_moves columns")

                lichess_ids = table.column("lichess_id").to_pylist()
                uci_moves = table.column("uci_moves").to_pylist()
                if workers <= 1:
                    results = [
                        fetch_and_extract_clocks(
                            str(lichess_id),
                            str(moves) if moves is not None else None,
                            timeout=timeout,
                            max_retries=max_retries,
                        )
                        for lichess_id, moves in zip(lichess_ids, uci_moves, strict=True)
                    ]
                else:
                    def _fetch_one(lichess_id: str, moves: str | None):
                        return fetch_and_extract_clocks(
                            lichess_id,
                            moves,
                            timeout=timeout,
                            max_retries=max_retries,
                        )

                    results = list(
                        executor.map(
                            _fetch_one,
                            (str(lichess_id) for lichess_id in lichess_ids),
                            (str(moves) if moves is not None else None for moves in uci_moves),
                        )
                    )

                augmented = table.append_column(
                    "move_clocks_seconds",
                    pa.array(
                        [result.move_clocks_seconds for result in results],
                        type=pa.list_(pa.float64()),
                    ),
                )
                augmented = augmented.append_column(
                    "clock_moves_match",
                    pa.array([result.moves_match for result in results], type=pa.bool_()),
                )
                augmented = augmented.append_column(
                    "clock_fetch_error",
                    pa.array([result.error for result in results], type=pa.string()),
                )

                if writer is None:
                    writer = pq.ParquetWriter(output_path, augmented.schema, compression="zstd")
                writer.write_table(augmented)
                progress.update(len(results))
        finally:
            progress.close()
            if writer is not None:
                writer.close()


@hydra.main(version_base=None, config_path="../../config", config_name="augment_clocks")
def main(cfg: DictConfig) -> None:
    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)
    workers = int(cfg.workers)
    batch_size = int(cfg.batch_size)
    timeout = float(cfg.request_timeout)
    max_retries = int(cfg.max_retries)

    parquet_files = sorted(input_dir.glob("*.parquet"))
    if not parquet_files:
        logger.error(f"No parquet files found in {input_dir}")
        return

    logger.info(f"Input dir: {input_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Workers: {workers}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Timeout: {timeout}s")
    logger.info(f"Max retries: {max_retries}")

    for input_path in parquet_files:
        output_path = output_dir / input_path.name
        if output_path.exists() and bool(cfg.skip_existing):
            logger.info(f"Skipping existing {output_path.name}")
            continue

        logger.info(f"Augmenting {input_path.name} -> {output_path.name}")
        _augment_parquet_file(
            input_path,
            output_path,
            workers=workers,
            batch_size=batch_size,
            timeout=timeout,
            max_retries=max_retries,
        )


if __name__ == "__main__":
    main()