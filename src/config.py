from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
MOVES_FILE = DATA_DIR / "all_uci_moves.txt"
PRETRAIN_DATASET_PATH = DATA_DIR / "processed/pretrain.parquet"
EVAL_DATASET_PATH = DATA_DIR / "processed/eval.parquet"
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
    epochs: float = 0.1
    warmup_iters: int = 100
    batch_size: int = 32
    num_workers: int = 4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    compile: bool = True
    compile_mode: str = "default"  # "default", "reduce-overhead", "max-autotune"
    compile_dynamic: bool = False  # False since we use explicit padding buckets
    compile_fullgraph: bool = True  # captures the entire model into a single graph if True
    padding_bucket_sizes: tuple[int, ...] = (64, 128, 192, 256, 384, 512, 768, 1024)


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
