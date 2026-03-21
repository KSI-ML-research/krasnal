import json
import tempfile
from pathlib import Path

import pytest

from config import SOS_ID
from tokenizer import Tokenizer, save_tokenizer_for_artifact


@pytest.fixture
def uci_moves_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("e2e4\n")
        f.write("e7e5\n")
        f.write("g1f3\n")
        f.write(Path(f.name).name + "\n")
    path = Path(f.name)
    yield path
    path.unlink(missing_ok=True)


def test_tokenizer_encode_decode_roundtrip(uci_moves_file):
    tok = Tokenizer(uci_moves_file)
    original = "e2e4 e7e5"
    encoded = tok.encode(original)
    decoded = tok.decode(encoded)
    assert decoded.strip() == original


def test_tokenizer_encode_empty_string(uci_moves_file):
    tok = Tokenizer(uci_moves_file)
    assert tok.encode("") == []
    assert tok.encode("e2e4") != []


def test_tokenizer_vocab_size(uci_moves_file):
    tok = Tokenizer(uci_moves_file)
    special_count = 6
    assert tok.get_vocab_size() > special_count


def test_tokenizer_mapping_hash(uci_moves_file):
    tok1 = Tokenizer(uci_moves_file)
    tok2 = Tokenizer(uci_moves_file)
    assert tok1.mapping_hash() == tok2.mapping_hash()
    assert len(tok1.mapping_hash()) == 64


def test_tokenizer_from_mapping_roundtrip(uci_moves_file):
    tok1 = Tokenizer(uci_moves_file)
    restored = Tokenizer.from_mapping(tok1.move_to_id)
    assert restored.get_vocab_size() == tok1.get_vocab_size()
    assert restored.encode("e2e4") == tok1.encode("e2e4")


def test_save_tokenizer_for_artifact(uci_moves_file):
    tok = Tokenizer(uci_moves_file)
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_path = Path(tmpdir) / "model.pt"
        sidecar = save_tokenizer_for_artifact(tok, artifact_path)
        assert sidecar.exists()
        with sidecar.open() as f:
            data = json.load(f)
        assert data["tokenizer_format"] == 1
        assert data["vocab_size"] == tok.get_vocab_size()
        assert data["move_to_id"]["<SOS>"] == SOS_ID
