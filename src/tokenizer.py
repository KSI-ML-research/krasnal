import json
from pathlib import Path

tokens = {
    "SOS_ID": 0,
    "EOS_ID": 1,
    "PAD_ID" : 2,
    "ELO_BELOW_1000_ID" : 3,
    "ELO_1000_1499_ID" : 4,
    "ELO_1500_1999_ID" : 5,
    "ELO_2000_2499_ID" : 6,
    "ELO_2500_2999_ID" : 7,
    "ELO_ABOVE_2999_ID" : 8,
    "WIN_WHITE_ID" : 9,
    "WIN_BLACK_ID" : 10,
    "DRAW_ID" : 11,
}

class Tokenizer:
    def __init__(self, uci_moves_path: Path):
        self.move_to_id = {}
        self.id_to_move = {} 
        
        for key, value in tokens.items():
            setattr(self, key.lower(), value)
       
        with open(uci_moves_path) as f:
            all_uci_moves = [line.strip() for line in f if line.strip()]

        for key, value in tokens.items():
            token_name = f"<{key.replace('_ID', '')}>"
            self.move_to_id[token_name] = value

        for idx, move in enumerate(
            all_uci_moves,
            start=max( tokens.values()) + 1,
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
