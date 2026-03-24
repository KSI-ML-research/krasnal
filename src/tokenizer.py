import json
from pathlib import Path

from config import (
    DRAW_ID,
    ELO_1000_ID,
    ELO_1500_ID,
    ELO_2000_ID,
    ELO_2500_ID,
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
        self.elo_1000_id = ELO_1000_ID
        self.elo_1500_id = ELO_1500_ID
        self.elo_2000_id = ELO_2000_ID
        self.elo_2500_id = ELO_2500_ID
        self.win_white_id = WIN_WHITE_ID
        self.win_black_id = WIN_BLACK_ID
        self.draw_id = DRAW_ID

        with open(uci_moves_path) as f:
            all_uci_moves = [line.strip() for line in f if line.strip()]

        self.move_to_id["<SOS>"] = self.sos_id
        self.move_to_id["<EOS>"] = self.eos_id
        self.move_to_id["<PAD>"] = self.pad_id
        self.move_to_id["<E10>"] = self.elo_1000_id
        self.move_to_id["<E15>"] = self.elo_1500_id
        self.move_to_id["<E20>"] = self.elo_2000_id
        self.move_to_id["<E25>"] = self.elo_2500_id
        self.move_to_id["<WW>"] = self.win_white_id
        self.move_to_id["<BW>"] = self.win_black_id
        self.move_to_id["<DW>"] = self.draw_id

        for idx, move in enumerate(
            all_uci_moves,
            start=max(
                SOS_ID,
                EOS_ID,
                PAD_ID,
                ELO_1000_ID,
                ELO_1500_ID,
                ELO_2000_ID,
                ELO_2500_ID,
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
