export PYTHONPATH := "."
export UV_CACHE_DIR := ".uv_cache"
default_seed := "42"

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

pipeline:
    uv sync
    cargo run --release
    just preprocess
    just pretrain
    just finetune
    just eval

preprocess *args:
    uv run scripts/preprocess.py --seed {{ default_seed }} {{ args }}

pretrain *args:
    uv run python scripts/pretrain.py --seed {{ default_seed }} {{ args }}

sft *args:
    uv run python scripts/sft.py --seed {{ default_seed }} {{ args }}

rlvr_legal *args:
    uv run python scripts/rlvr_legal_think.py {{ args }}

finetune *args:
    just sft --latest {{ args }}
    just rlvr_legal --latest {{ args }}

eval RUN *args:
	uv run python -m src.evals.run --run {{ RUN }} --seed {{ default_seed }} {{ args }}
