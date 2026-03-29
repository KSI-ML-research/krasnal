from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl


class CotShardWriter:
    """Persist consumed CoT rows as parquet shards."""

    def __init__(
        self,
        *,
        output_dir: Path,
        shard_size: int,
        filename_prefix: str = "",
    ) -> None:
        if shard_size <= 0:
            raise ValueError("shard_size must be > 0")
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.filename_prefix = filename_prefix
        self.buffer: list[dict[str, int | str | list[int] | None]] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def add_rows(self, rows: list[dict[str, int | str | list[int] | None]]) -> int:
        """Buffer rows and flush automatically when enough accumulate."""
        self.buffer.extend(rows)
        if len(self.buffer) >= self.shard_size:
            return self.flush()
        return 0

    def flush(self) -> int:
        """Write the buffered rows as one parquet shard."""
        if not self.buffer:
            return 0
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shard_path = self.output_dir / f"{self.filename_prefix}shard_{timestamp}.parquet"
        pl.DataFrame(self.buffer).write_parquet(shard_path)
        row_count = len(self.buffer)
        self.buffer.clear()
        return row_count
