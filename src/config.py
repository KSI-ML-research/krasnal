from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
MOVES_FILE = Path("src/uci_moves.txt")
DATASET_PATH = DATA_DIR / "processed/pretrain.parquet"
SFT_DATA_PATH = DATA_DIR / "processed/sft_data.parquet"
RLVR_DATASET_PATH = DATA_DIR / "processed/rlvr.parquet"
EVAL_DATASET_PATH = DATA_DIR / "processed/eval.parquet"
OUTPUTS_DIR = Path("outputs")
RUNS_DIR = OUTPUTS_DIR / "runs"
RESULTS_DIR = OUTPUTS_DIR / "results"
LOGS_DIR = OUTPUTS_DIR / "logs"
PIECES_DIR = Path("assets/pieces")


# @dataclass
# class ChessGPTConfig:
#     block_size: int = 1024
#     n_layer: int = 6
#     n_head: int = 6
#     n_embd: int = 384
#     dropout: float = 0.0
#     bias: bool = False  # no bias = slightly better and faster


@dataclass
class ChessGPTConfig:
    block_size: int = 2048
    n_layer: int = 12
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.1
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


@dataclass
class GRPOConfig:
    group_size: int = 8  # G
    kl_coeff: float = 0.01  # beta
    clip_eps: float = 0.2  # epsilon
    num_samples: int = 4  # prompts per batch
    reward_weights: dict | None = None  # weights for different reward components

    def __post_init__(self):
        if self.reward_weights is None:
            self.reward_weights = {"legality": 1.0, "outcome": 0.0}
