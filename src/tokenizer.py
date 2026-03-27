import json
from pathlib import Path

# special tokens
SOS_ID = 0
EOS_ID = 1
PAD_ID = 2

# elo tokens
ELO_BELLOW_1000_ID = 3
ELO_1000_1499_ID = 4
ELO_1500_1999_ID = 5
ELO_2000_2499_ID = 6
ELO_2500_2999_ID = 7
ELO_ABOVE_2999_ID = 8

# outcome tokens
WIN_WHITE_ID = 9
WIN_BLACK_ID = 10
DRAW_ID = 11


class Tokenizer:
    def __init__(self, uci_moves_path: Path):
        self.move_to_id = {}
        self.id_to_move = {}

        self.sos_id = SOS_ID
        self.eos_id = EOS_ID
        self.pad_id = PAD_ID
        self.elo_bellow_1000_id = ELO_BELLOW_1000_ID
        self.elo_1000_1499_id = ELO_1000_1499_ID
        self.elo_1500_1999_id = ELO_1500_1999_ID
        self.elo_2000_2499_id = ELO_2000_2499_ID
        self.elo_2500_2999_id = ELO_2500_2999_ID
        self.elo_above_2999_id = ELO_ABOVE_2999_ID
        self.win_white_id = WIN_WHITE_ID
        self.win_black_id = WIN_BLACK_ID
        self.draw_id = DRAW_ID

        with open(uci_moves_path) as f:
            all_uci_moves = [line.strip() for line in f if line.strip()]

        self.move_to_id["<SOS>"] = self.sos_id
        self.move_to_id["<EOS>"] = self.eos_id
        self.move_to_id["<PAD>"] = self.pad_id
        self.move_to_id["<ELO_BELLOW_1000>"] = self.elo_bellow_1000_id
        self.move_to_id["<ELO_1000_1499>"] = self.elo_1000_1499_id
        self.move_to_id["<ELO_1500_1999>"] = self.elo_1500_1999_id
        self.move_to_id["<ELO_2000_2499>"] = self.elo_2000_2499_id
        self.move_to_id["<ELO_2500_2999>"] = self.elo_2500_2999_id
        self.move_to_id["<ELO_ABOVE_2999>"] = self.elo_above_2999_id
        self.move_to_id["<WHITE_WON>"] = self.win_white_id
        self.move_to_id["<BLACK_WON>"] = self.win_black_id
        self.move_to_id["<DRAW>"] = self.draw_id

        for idx, move in enumerate(
            all_uci_moves,
            start=max(
                SOS_ID,
                EOS_ID,
                PAD_ID,
                ELO_BELLOW_1000_ID,
                ELO_1000_1499_ID,
                ELO_1500_1999_ID,
                ELO_2000_2499_ID,
                ELO_2500_2999_ID,
                ELO_ABOVE_2999_ID,
                WIN_WHITE_ID,
                WIN_BLACK_ID,
                DRAW_ID,
            )
            + 1,
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
