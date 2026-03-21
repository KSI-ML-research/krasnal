import hashlib
import json
from pathlib import Path

from config import (
    DRAW_ID,
    EOS_ID,
    PAD_ID,
    SOS_ID,
    WIN_BLACK_ID,
    WIN_WHITE_ID,
)


class Tokenizer:
    def __init__(self, uci_moves_path: Path):
        self.move_to_id = {}
        self.id_to_move = {}

        self.sos_id = SOS_ID
        self.eos_id = EOS_ID
        self.pad_id = PAD_ID
        self.win_white_id = WIN_WHITE_ID
        self.win_black_id = WIN_BLACK_ID
        self.draw_id = DRAW_ID

        with open(uci_moves_path) as f:
            all_uci_moves = [line.strip() for line in f if line.strip()]

        self.move_to_id["<SOS>"] = self.sos_id
        self.move_to_id["<EOS>"] = self.eos_id
        self.move_to_id["<PAD>"] = self.pad_id
        self.move_to_id["<WW>"] = self.win_white_id
        self.move_to_id["<BW>"] = self.win_black_id
        self.move_to_id["<DW>"] = self.draw_id

        for idx, move in enumerate(
            all_uci_moves,
            start=max(SOS_ID, EOS_ID, PAD_ID, WIN_WHITE_ID, WIN_BLACK_ID, DRAW_ID) + 1,
        ):
            self.move_to_id[move] = idx

        self.id_to_move = {v: k for k, v in self.move_to_id.items()}

    def get_vocab_size(self):
        return len(self.move_to_id)

    def encode(self, moves_str: str) -> list[int]:
        if not moves_str:
            return []
        tokens = moves_str.split(" ")
        return [self.move_to_id.get(t, self.pad_id) for t in tokens]

    def decode(self, ids: list[int]) -> str:
        return " ".join([self.id_to_move.get(i, "") for i in ids])

    def save_to_json(self, path: Path):
        with open(path, "w") as f:
            json.dump(self.move_to_id, f)

    def mapping_hash(self) -> str:
        payload = json.dumps(self.move_to_id, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_mapping(cls, move_to_id: dict[str, int]) -> Tokenizer:
        tok = cls.__new__(cls)
        tok.move_to_id = {k: int(v) for k, v in move_to_id.items()}
        tok.id_to_move = {int(v): k for k, v in tok.move_to_id.items()}
        tok.sos_id = SOS_ID
        tok.eos_id = EOS_ID
        tok.pad_id = PAD_ID
        tok.win_white_id = WIN_WHITE_ID
        tok.win_black_id = WIN_BLACK_ID
        tok.draw_id = DRAW_ID
        return tok


def tokenizer_sidecar_path_for_artifact(artifact_path: Path) -> Path:
    return artifact_path.parent / "tokenizer.json"


def save_tokenizer_for_artifact(tokenizer: Tokenizer, artifact_path: Path) -> Path:
    sidecar_path = tokenizer_sidecar_path_for_artifact(artifact_path)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tokenizer_format": 1,
        "mapping_hash": tokenizer.mapping_hash(),
        "vocab_size": tokenizer.get_vocab_size(),
        "move_to_id": tokenizer.move_to_id,
    }
    with sidecar_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True)
    return sidecar_path
