SEED := "42"
export PYTHONPATH := "src"
export UV_CACHE_DIR := ".uv_cache"

# Install dependencies and setup pre-commit hooks
setup:
    uv sync
    uv run pre-commit install
    cargo build

# Run linters for Python and Rust code
lint *args:
    uv run ruff check . {{args}}
    cargo clippy {{args}}

# Format code with ruff and rustfmt
format *args:
    uv run ruff format . {{args}}
    cargo fmt {{args}}

# Run tests for Python and Rust code
test *args:
    uv run pytest {{args}}
    cargo test {{args}}

# Run all pre-commit hooks
pre-commit:
    uv run pre-commit run --all-files

# Run full training pipeline: download games, preprocess, pretrain, evaluate
pipeline:
    uv sync
    just download-games
    just preprocess
    just pretrain
    just evaluate --latest

# Download chess games from Lichess
download-games:
    cargo run --release --bin download-games

# Preprocess downloaded games into training dataset
preprocess *args:
    uv run scripts/preprocess.py {{args}} --seed {{SEED}}

# Generate synthetic CoT dataset with Stockfish
generate-sft-cot *args:
    uv run scripts/generate_sft_cot.py {{args}} --stockfish-path "$(which stockfish)" --seed {{SEED}}

# Stage 1: Pretrain model on large dataset of chess games
pretrain *args:
    uv run scripts/pretrain.py {{args}} --seed {{SEED}}

# Supervised CoT fine-tuning from a pretrained checkpoint
sft-cot *args:
    uv run scripts/sft_cot.py {{args}} --stockfish-path "$(which stockfish)" --seed {{SEED}}

# Evaluate trained model on held-out dataset
evaluate *args:
    uv run scripts/evaluate.py {{args}} --seed {{SEED}}

# Stage 2: RLVR phase 1 fine-tuning from a pretrained checkpoint
rlvr-phase1 *args:
    uv run scripts/rlvr_phase1.py {{args}} --seed {{SEED}}
