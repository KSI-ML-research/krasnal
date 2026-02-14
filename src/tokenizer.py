import json
from pathlib import Path


class Tokenizer:
    def __init__(self, uci_moves_path: Path):
        self.move_to_id = {}
        self.id_to_move = {}

        with open(uci_moves_path, "r") as f:
            all_uci_moves = [line.strip() for line in f if line.strip()]

        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"
        self.pad_token = "<PAD>"

        self.sos_id = 0
        self.eos_id = 1
        self.pad_id = 2

        self.move_to_id[self.sos_token] = self.sos_id
        self.move_to_id[self.eos_token] = self.eos_id
        self.move_to_id[self.pad_token] = self.pad_id

        for idx, move in enumerate(all_uci_moves, start=3):
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
