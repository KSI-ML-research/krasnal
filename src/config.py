from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
MOVES_FILE = Path("src/uci_moves.txt")
PRETRAIN_DATASET_PATH = DATA_DIR / "processed/pretrain.parquet"
EVAL_DATASET_PATH = DATA_DIR / "processed/eval.parquet"
ARTIFACTS_DIR = Path("artifacts")
PIECES_DIR = Path("assets/pieces")


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int | None = None
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = False


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
