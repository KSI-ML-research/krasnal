# Setup project dependencies and pre-commit hooks
setup:
    uv sync
    uv run pre-commit install
    cargo build

# Run all linting (Python & Rust)
lint:
    uv run ruff check .
    cargo clippy

# Run all formatting (Python & Rust)
format:
    uv run ruff format .
    cargo fmt

# Run all tests (Python & Rust)
test:
    uv run pytest
    cargo test

# Run all pre-commit hooks
pre-commit:
    uv run pre-commit run --all-files

# Run whole pipeline
pipeline:
    uv sync
    cargo run --release
    uv run src/preprocess.py
    uv run src/train.py
