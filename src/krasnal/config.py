import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

MlpActivation = Literal["gelu", "swiglu", "relu2"]

DATA_DIR = Path("data")
RAW_UCI_DIR = DATA_DIR / "1_filtered"
TOKENIZED_DIR = Path(os.environ.get("KRASNAL_TOKENIZED_DIR", DATA_DIR / "2_tokenized"))
ARTIFACTS_DIR = Path(os.environ.get("KRASNAL_ARTIFACTS_DIR", "artifacts"))
PRETRAIN_DATASET_PATH = TOKENIZED_DIR / "pretrain"
EVAL_DATASET_PATH = TOKENIZED_DIR / "eval.parquet"
MOVE_VOCAB_PATH = TOKENIZED_DIR / "move_vocab.json"
WHAT_IS_ON_BASELINE_COUNTS_PATH = TOKENIZED_DIR / "what_is_on_baseline_counts.json"
CLOCK_IGNORE_ID = 2**32 - 1


@dataclass
class GPTConfig:
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    # When True, ``GPT.forward`` requires ``active_clock_ids`` / ``opponent_clock_ids``
    # shaped like ``idx``; clock residual uses the clock MLP only.
    use_clock_encodings: bool
    # Hidden width of the 2 -> h -> n_embd GELU MLP; must satisfy 1 <= h <= n_embd when enabled.
    clock_encoding_hidden: int
    vocab_size: int | None = None
    dropout: float = 0.0
    # Transformer block FFN: ``gelu`` / ``relu2`` (4*d MLP) or ``swiglu`` (gated, ~matched params).
    mlp_activation: MlpActivation = "swiglu"


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
    num_workers: int = 8
    pin_memory: bool = True

    # Compilation
    compile: bool = True
    compile_mode: str = "default"  # "default", "reduce-overhead", "max-autotune"
    compile_dynamic: bool = False  # packed train uses fixed block_size sequences
    compile_fullgraph: bool = True  # captures the entire model into a single graph if True
