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

SPECIAL_TOKENS = {
    "<sos>": 0,
    "<eos>": 1,
    "<pad>": 2,
    "<win_white>": 3,
    "<win_black>": 4,
    "<draw>": 5,
    "<think>": 6,
    "</think>": 7,
    "<branch>": 8,
}


class Tokenizer:
    def __init__(self, uci_moves_path: Path):
        self.move_to_id = {}
        self.id_to_move = {}

        with open(uci_moves_path) as f:
            all_uci_moves = [line.strip() for line in f if line.strip()]

        # add special tokens to the vocabulary
        for token, idx in SPECIAL_TOKENS.items():
            self.move_to_id[token] = idx

        # add UCI moves to the vocab, ensuring no overlap with special token IDs
        for idx, move in enumerate(all_uci_moves, start=max(SPECIAL_TOKENS.values()) + 1):
            self.move_to_id[move] = idx

        self.id_to_move = {v: k for k, v in self.move_to_id.items()}

    def get_vocab_size(self) -> int:
        return len(self.move_to_id)

    @property
    def sos_id(self) -> int:
        return SOS_ID

    @property
    def think_start_id(self) -> int:
        return THINK_START_ID

    @property
    def think_end_id(self) -> int:
        return THINK_END_ID

    @property
    def step_back_id(self) -> int:
        return STEP_BACK_ID

    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def win_white_id(self) -> int:
        return WIN_WHITE_ID

    @property
    def win_black_id(self) -> int:
        return WIN_BLACK_ID

    @property
    def draw_id(self) -> int:
        return DRAW_ID

    @property
    def eos_id(self) -> int:
        return EOS_ID

    def encode(self, moves_str: str) -> list[int]:
        if not moves_str:
            return []
        tokens = moves_str.split(" ")
        return [self.move_to_id.get(t, self.pad_id) for t in tokens]

    def decode(self, ids: list[int]) -> str:
        return " ".join([self.id_to_move.get(i, "") for i in ids])

    def save_to_json(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(self.move_to_id, f)


def save_tokenizer_for_artifact(tokenizer: Tokenizer, model_path: Path) -> None:
    """Save tokenizer vocabulary alongside a model checkpoint."""
    tokenizer_path = model_path.parent / f"{model_path.stem}_tokenizer.json"
    tokenizer.save_to_json(tokenizer_path)
