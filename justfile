# load .env variables for every command
set dotenv-load := true
import 'lichess.just'

SEED := "42"
export PYTHONPATH := "."
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
    PYTHONPATH=src uv run pytest {{args}}
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
    PYTHONPATH=src uv run scripts/preprocess.py {{args}} --seed {{SEED}}

# Stage 1: Pretrain model on large dataset of chess games
pretrain *args:
    PYTHONPATH=src uv run scripts/pretrain.py {{args}} --seed {{SEED}}

# Evaluate trained model on held-out dataset
evaluate *args:
    PYTHONPATH=src uv run scripts/evaluate.py {{args}} --seed {{SEED}}
