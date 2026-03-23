import json
from pathlib import Path

import polars as pl

from src.sft import (
    CotReplaySource,
    CotShardWriter,
    build_cot_row,
    compute_batch_sizes,
    derive_worker_seed,
    serialize_pv_tokens,
)
from src.tokenizer import Tokenizer


def test_serialize_pv_tokens_uses_branch_separator():
    tokenizer = Tokenizer(Path("src/uci_moves.txt"))
    token_ids = serialize_pv_tokens(tokenizer, [["e2e4", "e7e5"], ["d2d4"]])
    assert token_ids == [
        tokenizer.move_to_id["e2e4"],
        tokenizer.move_to_id["e7e5"],
        tokenizer.step_back_id,
        tokenizer.move_to_id["d2d4"],
    ]


def test_build_cot_row_places_actual_move_after_think_block_and_sets_metadata():
    tokenizer = Tokenizer(Path("src/uci_moves.txt"))
    row = build_cot_row(
        tokenizer=tokenizer,
        result=1,
        prefix_moves=["e2e4"],
        pv_lines=[["c7c5", "g1f3"], ["e7e5"]],
        actual_move="c7c5",
        depth=10,
        movetime_ms=None,
        stockfish_score_cp=27,
        source_game_index=123,
    )
    think_end_index = row["token_ids"].index(tokenizer.think_end_id)
    assert row["token_ids"][think_end_index + 1] == tokenizer.move_to_id["c7c5"]
    assert row["pv_count"] == 2
    assert row["depth"] == 10
    assert row["stockfish_score_cp"] == 27
    assert row["source_game_index"] == 123


def test_shard_writer_flushes_rows_and_updates_manifest(tmp_path):
    shard_dir = tmp_path / "cot_shards"
    manifest_path = tmp_path / "cot_manifest.json"
    writer = CotShardWriter(
        output_dir=shard_dir,
        manifest_path=manifest_path,
        shard_size=2,
        metadata={"mode": "online"},
    )

    rows = [
        {"token_ids": [1, 2, 3], "pv_count": 1},
        {"token_ids": [4, 5, 6], "pv_count": 2},
    ]
    flushed = writer.add_rows(rows)

    assert flushed == 2
    shard_paths = sorted(shard_dir.glob("*.parquet"))
    assert len(shard_paths) == 1

    payload = json.loads(manifest_path.read_text())
    assert payload["mode"] == "online"
    assert payload["shard_count"] == 1
    assert payload["total_rows"] == 2


def test_shard_writer_supports_prefixed_files_without_manifest(tmp_path):
    shard_dir = tmp_path / "global_shards"
    writer = CotShardWriter(
        output_dir=shard_dir,
        manifest_path=None,
        shard_size=1,
        metadata={"mode": "online"},
        filename_prefix="run_a_",
    )

    flushed = writer.add_rows([{"token_ids": [1, 2, 3], "pv_count": 1}])

    assert flushed == 1
    shard_paths = sorted(shard_dir.glob("*.parquet"))
    assert len(shard_paths) == 1
    assert shard_paths[0].name == "run_a_shard_000001.parquet"


def test_replay_source_loads_saved_shards(tmp_path):
    shard_dir = tmp_path / "cot_shards"
    shard_dir.mkdir()
    pl.DataFrame({"token_ids": [[1, 2, 3], [4, 5, 6]]}).write_parquet(
        shard_dir / "shard_000001.parquet"
    )
    pl.DataFrame({"token_ids": [[7, 8, 9]]}).write_parquet(shard_dir / "shard_000002.parquet")

    source = CotReplaySource(shard_dir, seed=42)

    assert len(source) == 3
    assert len(source.sample_sequences(2)) == 2


def test_replay_source_can_consume_rows_once(tmp_path):
    shard_dir = tmp_path / "cot_shards"
    shard_dir.mkdir()
    pl.DataFrame({"token_ids": [[1, 2], [3, 4], [5, 6]]}).write_parquet(
        shard_dir / "shard_000001.parquet"
    )

    source = CotReplaySource(shard_dir, seed=42)

    assert source.total_rows == 3
    assert len(source.take_sequences(2)) == 2
    assert source.remaining_rows() == 1
    assert len(source.take_sequences(2)) == 1
    assert source.remaining_rows() == 0


def test_compute_batch_sizes_preserves_ratio_shape():
    cot_batch_size, normal_batch_size = compute_batch_sizes(32, 0.7)
    assert cot_batch_size == 22
    assert normal_batch_size == 10


def test_derive_worker_seed_is_deterministic_and_distinct():
    assert derive_worker_seed(42, 0) == 10042
    assert derive_worker_seed(42, 1) == 20042
    assert derive_worker_seed(42, 0) != derive_worker_seed(42, 1)
