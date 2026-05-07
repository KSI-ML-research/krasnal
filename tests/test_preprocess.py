import json
from importlib import util
from pathlib import Path

import polars as pl
import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "data" / "preprocess.py"
_SPEC = util.spec_from_file_location("preprocess_module", _MODULE_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

_compute_check_qa_probs = _MODULE._compute_check_qa_probs
build_move_vocab_from_corpus = _MODULE.build_move_vocab_from_corpus
_resolve_preprocess_workers = _MODULE._resolve_preprocess_workers


def test_compute_check_qa_probs_balances_yes_no_average():
    p_yes, p_no = _compute_check_qa_probs(check_count=30, no_check_count=70, check_qa_prob=0.5)

    assert p_yes == 0.5
    assert p_no == 0.21428571428571427


def test_compute_check_qa_probs_handles_no_non_check_positions():
    p_yes, p_no = _compute_check_qa_probs(check_count=10, no_check_count=0, check_qa_prob=0.5)

    assert p_yes == 0.5
    assert p_no == 0.0


def test_build_move_vocab_from_corpus_writes_sorted_manifest_vocab(tmp_path):
    corpus_path = tmp_path / "games.parquet"
    pl.DataFrame(
        {
            "uci_moves": ["g1f3 e7e5", "e2e4"],
            "piece_moved": [["n", "p"], ["p"]],
        }
    ).write_parquet(corpus_path)
    output_path = tmp_path / "move_vocab.json"

    build_move_vocab_from_corpus(
        [corpus_path],
        piece_aware_moves=True,
        side_prefixed_moves=True,
        output_path=output_path,
    )

    payload = json.loads(output_path.read_text())
    move_vocab = {
        token: token_id for token, token_id in payload["vocab"].items() if not token.startswith("<")
    }

    assert payload["manifest"]["piece_aware_moves"] is True
    assert payload["manifest"]["side_prefixed_moves"] is True
    assert payload["manifest"]["vocab_size"] == len(payload["vocab"])
    assert list(move_vocab) == ["b:pawn:e7e5", "w:knight:g1f3", "w:pawn:e2e4"]


def test_build_move_vocab_from_corpus_fails_on_malformed_piece_moved(tmp_path):
    corpus_path = tmp_path / "games.parquet"
    pl.DataFrame(
        {
            "uci_moves": ["e2e4 e7e5"],
            "piece_moved": [["p"]],
        }
    ).write_parquet(corpus_path)

    with pytest.raises(ValueError, match="piece_moved length"):
        build_move_vocab_from_corpus(
            [corpus_path],
            piece_aware_moves=False,
            side_prefixed_moves=True,
            output_path=tmp_path / "move_vocab.json",
        )


def test_resolve_preprocess_workers_allows_explicit_worker_count():
    assert _resolve_preprocess_workers({"preprocess_workers": 24}, shard_count=3) == 24


def test_resolve_preprocess_workers_caps_auto_mode_at_eight(monkeypatch):
    monkeypatch.setattr(_MODULE.os, "cpu_count", lambda: 64)

    assert _resolve_preprocess_workers({"preprocess_workers": 0}, shard_count=32) == 8
