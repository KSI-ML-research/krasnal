from dataclasses import dataclass
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent

DATA_DIR = Path("data")
RAW_UCI_DIR = DATA_DIR / "1_filtered"
MOVES_FILE = _PACKAGE_ROOT / "uci_moves.txt"
ARTIFACTS_DIR = Path("artifacts")
PRETRAIN_DATASET_PATH = DATA_DIR / "2_tokenized/pretrain.parquet"
SFT_COT_SHARDS_DIR = DATA_DIR / "2_tokenized" / "sft_cot_shards"
EVAL_DATASET_PATH = DATA_DIR / "2_tokenized/eval.parquet"

SF_EVAL_BUCKETS = 128
LOSS_IGNORE_INDEX = -100


@dataclass
class GPTConfig:
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    vocab_size: int | None = None
    dropout: float = 0.0


@dataclass
class TrainConfig:
    # Training hyperparameters
    learning_rate: float
    min_lr: float  # cosine annealing minimum LR
    epochs: float
    warmup_iters: int
    batch_size: int
    weight_decay: float
    beta1: float
    beta2: float
    grad_clip: float

    # Logging and evaluation
    log_interval: int
    eval_interval: int
    eval_num_games: int = 300

    # Runtime
    max_iters: int | None = None  # set after computing from epochs and dataset size
    steps_per_epoch: int | None = None  # set after computing from dataset size

    # Data loading
    num_workers: int = 4
    pin_memory: bool = True

    # Compilation
    compile: bool = True
    compile_mode: str = "default"  # "default", "reduce-overhead", "max-autotune"
    compile_dynamic: bool = False  # False since we use explicit padding buckets
    compile_fullgraph: bool = True  # captures the entire model into a single graph if True
    padding_bucket_sizes: tuple[int, ...] = ()
