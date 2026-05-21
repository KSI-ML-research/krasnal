from dataclasses import dataclass
from pathlib import Path
from typing import Literal

MlpActivation = Literal["gelu", "swiglu", "relu2"]

DATA_DIR = Path("data")
RAW_UCI_DIR = DATA_DIR / "1_filtered"
ARTIFACTS_DIR = Path("artifacts")
PRETRAIN_DATASET_PATH = DATA_DIR / "2_tokenized/pretrain.parquet"
EVAL_DATASET_PATH = DATA_DIR / "2_tokenized/eval.parquet"
MOVE_VOCAB_PATH = DATA_DIR / "2_tokenized/move_vocab.json"
CLOCK_IGNORE_ID = 2**32 - 1


@dataclass
class GPTConfig:
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    # When True, ``GPT.forward`` requires ``active_clock_ids`` / ``opponent_clock_ids``
    # shaped like ``idx``; clock residual uses ``time_mlp`` only (no linear-only path).
    use_time_conditioning: bool
    # Hidden width of the 2 -> h -> n_embd GELU MLP; must satisfy 1 <= h <= n_embd when enabled.
    time_conditioning_hidden: int
    vocab_size: int | None = None
    dropout: float = 0.0
    # Transformer block FFN: ``gelu`` / ``relu2`` (4*d MLP) or ``swiglu`` (gated, ~matched params).
    mlp_activation: MlpActivation = "gelu"


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

    # Optimizer selection: "adamw" or "muon"
    optimizer: str = "adamw"
    # Muon-specific hyperparameters (only used when optimizer == "muon")
    muon_lr: float = 0.02
    muon_momentum: float = 0.95

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
