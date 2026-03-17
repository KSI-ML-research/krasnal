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
    cargo run --release
    uv run src/preprocess.py
    just pretrain

pretrain:
    SEED={{SEED}} uv run src/train.py
    uv run scripts/evaluate.py

eval:
    uv run scripts/evaluate.py

preprocess:
    uv run src/preprocess.py
