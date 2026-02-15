from pathlib import Path
from dataclasses import dataclass

DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
MOVES_FILE = DATA_DIR / "all_uci_moves.txt"
DATASET_PATH = DATA_DIR / "processed/tokenized_games.parquet"
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "chess_model.pt"
PIECES_DIR = Path("assets/pieces")


@dataclass
class ChessGPTConfig:
    block_size: int = 512
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384


@dataclass
class TrainConfig:
    learning_rate: float = 5e-4
    max_iters: int = 5000
    batch_size: int = 32
    num_workers: int = 4


SOS_ID = 0
EOS_ID = 1
PAD_ID = 2
