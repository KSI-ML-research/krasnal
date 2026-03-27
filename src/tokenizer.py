import json
from pathlib import Path

SOS_ID = 0
EOS_ID = 1
PAD_ID = 2
WIN_WHITE_ID = 3
WIN_BLACK_ID = 4
DRAW_ID = 5
THINK_START_ID = 6
THINK_END_ID = 7
STEP_BACK_ID = 8

SPECIAL_TOKENS = [
    SOS_ID,
    EOS_ID,
    PAD_ID,
    WIN_WHITE_ID,
    WIN_BLACK_ID,
    DRAW_ID,
    THINK_START_ID,
    THINK_END_ID,
    STEP_BACK_ID,
]


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
        self.think_start_id = THINK_START_ID
        self.think_end_id = THINK_END_ID
        self.step_back_id = STEP_BACK_ID

        with open(uci_moves_path) as f:
            all_uci_moves = [line.strip() for line in f if line.strip()]

        self.move_to_id["<SOS>"] = self.sos_id
        self.move_to_id["<EOS>"] = self.eos_id
        self.move_to_id["<PAD>"] = self.pad_id
        self.move_to_id["<WW>"] = self.win_white_id
        self.move_to_id["<BW>"] = self.win_black_id
        self.move_to_id["<DW>"] = self.draw_id
        self.move_to_id["<think>"] = self.think_start_id
        self.move_to_id["</think>"] = self.think_end_id
        self.move_to_id["<branch>"] = self.step_back_id

        for idx, move in enumerate(
            all_uci_moves,
            start=max(SPECIAL_TOKENS) + 1,
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


def save_tokenizer_for_artifact(tokenizer: Tokenizer, model_path: Path) -> None:
    """Save tokenizer vocabulary alongside a model checkpoint."""
    tokenizer_path = model_path.parent / f"{model_path.stem}_tokenizer.json"
    tokenizer.save_to_json(tokenizer_path)
