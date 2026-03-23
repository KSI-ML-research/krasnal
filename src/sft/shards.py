from __future__ import annotations

import json
from pathlib import Path

import polars as pl


class CotShardWriter:
    """Persist consumed CoT rows as parquet shards and an optional manifest."""

    def __init__(
        self,
        *,
        output_dir: Path,
        manifest_path: Path | None,
        shard_size: int,
        metadata: dict[str, object],
        filename_prefix: str = "",
    ) -> None:
        if shard_size <= 0:
            raise ValueError("shard_size must be > 0")
        self.output_dir = output_dir
        self.manifest_path = manifest_path
        self.shard_size = shard_size
        self.metadata = metadata
        self.filename_prefix = filename_prefix
        self.buffer: list[dict[str, int | str | list[int] | None]] = []
        self.shard_count = 0
        self.total_rows = 0
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.write_manifest()

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
        self.shard_count += 1
        shard_path = self.output_dir / f"{self.filename_prefix}shard_{self.shard_count:06d}.parquet"
        pl.DataFrame(self.buffer).write_parquet(shard_path)
        row_count = len(self.buffer)
        self.total_rows += row_count
        self.buffer.clear()
        self.write_manifest()
        return row_count

    def buffered_rows(self) -> int:
        """Return the number of rows still buffered in memory."""
        return len(self.buffer)

    def write_manifest(self) -> None:
        """Write shard metadata for replay and inspection."""
        if self.manifest_path is None:
            return
        payload = {
            **self.metadata,
            "shard_count": self.shard_count,
            "total_rows": self.total_rows,
            "buffered_rows": self.buffered_rows(),
        }
        with open(self.manifest_path, "w") as f:
            json.dump(payload, f, indent=2)
