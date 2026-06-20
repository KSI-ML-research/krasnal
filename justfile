# load .env variables for every command
set dotenv-load := true

export UV_CACHE_DIR := ".uv_cache"
export HF_HUB_ENABLE_HF_TRANSFER := "1"
export POLARS_MAX_THREADS := "2"


# Print common project commands
@help:
    @echo "Usage:"
    @echo "  just lint [args]                      - run Ruff"
    @echo "  just format [args]                    - format Python code"
    @echo "  just test [args]                      - run Python tests"
    @echo "  just pre-commit                       - run all pre-commit hooks"
    @echo "  just download-games [args]            - download & filter games"
    @echo "  just preprocess                       - tokenize Aix-filtered games for training"
    @echo "  just pretrain model=large train=cuda  - run pretraining stage"
    @echo ""
    @echo "Bot commands (via make):"
    @echo "  make bot-setup                        - setup lichess-bot"
    @echo "  make bot-run MODEL_PATH=...           - run bot with model"
    @echo "  make bot-clean                        - remove lichess-bot"

# Run linters for Python code
lint *args:
    uv run ruff check . {{args}}

# Format Python code
format *args:
    uv run ruff format . {{args}}

# Run Python tests
test *args:
    uv run pytest {{args}}

# Run all pre-commit hooks
pre-commit:
    uv run pre-commit run --all-files


# Download & filter Aix Lichess database for high-quality games with evals
# Uses DuckDB + Aix extension for fast SQL-based filtering
# Uses cached files first, downloads missing ones automatically
download-games *args:
    uv run scripts/data/download_games.py {{args}}

# Preprocess downloaded games into training dataset
preprocess *args:
    uv run scripts/data/preprocess.py {{args}}

# Pretrain model on large dataset of chess games
pretrain *args:
    uv run scripts/training/pretrain.py {{args}}




# Remove dataset hf-cache
hf-cache-clean:
    hf cache rm dataset/thomasd1/aix-lichess-database -y

clean:
    rm -rf *.log wandb/ artifacts/ outputs/ .uv_cache/ .hf_cache/ .hydra/ .ruff_cache/ .pytest_cache
