import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import MOVES_FILE
from src.tokenizer import (
    Tokenizer,
    load_tokenizer_from_sidecar,
    save_tokenizer_for_artifact,
    tokenizer_sidecar_path_for_artifact,
)


def test_tokenizer_initialization():
    tok = Tokenizer(MOVES_FILE)

    # Check if special tokens are mapped
    assert "<SOS>" in tok.move_to_id
    assert "<EOS>" in tok.move_to_id
    assert "<PAD>" in tok.move_to_id
    assert "<think>" in tok.move_to_id
    assert "</think>" in tok.move_to_id
    assert "<back>" in tok.move_to_id

    # Check regular moves
    assert "e2e4" in tok.move_to_id
    assert "a7a8q" in tok.move_to_id
    assert "a7a8r" in tok.move_to_id


def test_tokenizer_encode_decode():
    tok = Tokenizer(MOVES_FILE)

    # Regular encode
    moves_str = "e2e4 e7e5"
    encoded = tok.encode(moves_str)
    assert len(encoded) == 2
    assert tok.decode(encoded) == moves_str


def test_tokenizer_cot_encode_decode():
    tok = Tokenizer(MOVES_FILE)

    # CoT encode
    cot_str = "<think> e2e4 e7e5 <back> g1f3 </think> b8c6"
    encoded = tok.encode(cot_str)

    assert len(encoded) == 7

    # Make sure thought starts decoding perfectly
    assert tok.decode(encoded) == cot_str


def test_tokenizer_handles_unknown_moves_with_pad():
    tok = Tokenizer(MOVES_FILE)

    # "unknown" should be fallback to <PAD> token id
    encoded = tok.encode("e2e4 unknown_move e7e5")
    assert encoded[1] == tok.pad_id


def test_tokenizer_sidecar_round_trip(tmp_path):
    tok = Tokenizer(MOVES_FILE)
    artifact_path = tmp_path / "model.pt"
    sidecar_path = save_tokenizer_for_artifact(tok, artifact_path)

    assert sidecar_path == tokenizer_sidecar_path_for_artifact(artifact_path)
    assert sidecar_path.exists()

    restored = load_tokenizer_from_sidecar(sidecar_path)
    assert restored.move_to_id == tok.move_to_id
    assert restored.id_to_move == tok.id_to_move
    assert restored.mapping_hash() == tok.mapping_hash()
