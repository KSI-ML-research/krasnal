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
    block_size: int = 1024
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = False  # no bias = slightly better and faster


@dataclass
class TrainConfig:
    learning_rate: float = 5e-4
    min_lr: float = 5e-5  # cosine annealing minimum LR
    max_iters: int = 50000
    warmup_iters: int = 100
    batch_size: int = 32
    num_workers: int = 4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    compile: bool = True  # use torch.compile (best for long runs, e.g. 10k+ iters; disable for short/debug runs to avoid compile overhead)
    padding_bucket_sizes: tuple[int, ...] = (64, 128, 192, 256, 384, 512, 768, 1024)


SOS_ID = 0
EOS_ID = 1
PAD_ID = 2
