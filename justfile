# Global defaults
SEED := "42"

# ===== SETUP & DEVELOPMENT =====
setup:
    uv sync
    uv run pre-commit install
    cargo build

lint:
    uv run ruff check .
    cargo clippy

format:
    uv run ruff format .
    cargo fmt

test:
    uv run pytest
    cargo test

pre-commit:
    uv run pre-commit run --all-files

# ===== TRAINING =====

pipeline:
    uv sync
    just download-games
    just preprocess
    just pretrain

download-games:
    cargo run --release --bin download-games

preprocess:
    PYTHONPATH=src uv run scripts/preprocess.py

pretrain:
    SEED={{SEED}} PYTHONPATH=src uv run scripts/pretrain.py
    PYTHONPATH=src uv run scripts/evaluate.py

eval:
    PYTHONPATH=src uv run scripts/evaluate.py

# ===== PUZZLES =====
download-puzzles:
    mkdir -p data
    curl -L --progress-bar \
        "https://database.lichess.org/lichess_db_puzzle.csv.zst" \
        -o data/lichess_db_puzzle.csv.zst

prepare-puzzles:
    cargo run --release --bin prepare_puzzles
