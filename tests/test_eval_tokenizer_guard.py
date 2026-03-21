import json
from pathlib import Path

import pytest

from src.config import MOVES_FILE
from src.evals.run import validate_eval_tokenizer_compatibility
from src.tokenizer import Tokenizer


def _write_meta(dataset_path: Path, tokenizer: Tokenizer, *, hash_override: str | None = None):
    payload = {
        "metadata_format": 1,
        "dataset_path": str(dataset_path),
        "tokenizer_hash": hash_override or tokenizer.mapping_hash(),
        "vocab_size": tokenizer.get_vocab_size(),
        "seed": 42,
    }
    with Path(f"{dataset_path}.meta.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f)


def test_eval_guard_allows_matching_metadata(tmp_path):
    tokenizer = Tokenizer(MOVES_FILE)
    dataset_path = tmp_path / "eval.parquet"
    dataset_path.write_bytes(b"x")
    _write_meta(dataset_path, tokenizer)

    validate_eval_tokenizer_compatibility(dataset_path, tokenizer, allow_legacy=False)


def test_eval_guard_rejects_hash_mismatch(tmp_path):
    tokenizer = Tokenizer(MOVES_FILE)
    dataset_path = tmp_path / "eval.parquet"
    dataset_path.write_bytes(b"x")
    _write_meta(dataset_path, tokenizer, hash_override="deadbeef")

    with pytest.raises(SystemExit):
        validate_eval_tokenizer_compatibility(dataset_path, tokenizer, allow_legacy=False)


def test_eval_guard_requires_metadata_by_default(tmp_path):
    tokenizer = Tokenizer(MOVES_FILE)
    dataset_path = tmp_path / "eval.parquet"
    dataset_path.write_bytes(b"x")

    with pytest.raises(SystemExit):
        validate_eval_tokenizer_compatibility(dataset_path, tokenizer, allow_legacy=False)


def test_eval_guard_legacy_override_without_metadata(tmp_path):
    tokenizer = Tokenizer(MOVES_FILE)
    dataset_path = tmp_path / "eval.parquet"
    dataset_path.write_bytes(b"x")

    validate_eval_tokenizer_compatibility(dataset_path, tokenizer, allow_legacy=True)
