# load .env variables for every command
set dotenv-load := true

SEED := "42"
PREPROCESS_WORKERS := "0"
# 0 lets preprocess auto-pick worker count (capped in script)
export UV_CACHE_DIR := ".uv_cache"
export HF_HUB_ENABLE_HF_TRANSFER := "1"
export POLARS_MAX_THREADS := "2"
LICHESS_BOT_REPO := "https://github.com/lichess-bot-devs/lichess-bot.git"
# pinned lichess bot commit so that the setup is deterministic
LICHESS_BOT_REF := "96a8f74d87a42db8039e847548fec0d9528bb079"


# Print common project commands
@help:
    @echo "Usage:"
    @echo "  just setup                            - install deps, hooks"
    @echo "  just lint [args]                      - run Ruff"
    @echo "  just format [args]                    - format Python code"
    @echo "  just test [args]                      - run Python tests"
    @echo "  just pre-commit                       - run all pre-commit hooks"
    @echo "  just pipeline                         - run full training pipeline"
    @echo "  just download-games [target=5000000] - download & filter Aix DB (DuckDB)"
    @echo "  just preprocess [args]                - tokenize Aix-filtered games for training"
    @echo "  just pretrain model=large train=cuda  - run pretraining stage"

# Install dependencies and setup pre-commit hooks
setup:
    uv sync
    uv run pre-commit install


# Run linters for Python code
lint *args:
    uv run ruff check . {{args}}

# Format Python code
format *args:
    uv run ruff format . {{args}}

# Run Python tests
test *args:
    uv run pytest {{args}}

# Run tests with coverage for Python code
test-cov *args:
    uv run pytest --cov=src/krasnal --cov-report=term-missing --cov-report=xml {{args}}

# Run all pre-commit hooks
pre-commit:
    uv run pre-commit run --all-files


# Run full training pipeline: download games, preprocess, pretrain, evaluate
pipeline:
    uv sync
    just download-games
    just preprocess
    just pretrain
    # just generate-sft-cot --depth 10
    # just train-sft-cot


# Download & filter Aix Lichess database for high-quality games with evals
# Uses DuckDB + Aix extension for fast SQL-based filtering
# Uses cached files first, downloads missing ones automatically
# Target: ~5M games by default (configurable)
download-games *args:
    uv run scripts/data/download_games.py {{args}}

# Preprocess downloaded games into training dataset
preprocess *args:
    uv run scripts/data/preprocess.py {{args}} seed={{SEED}} preprocess_workers={{PREPROCESS_WORKERS}}

# Stage 1: Pretrain model on large dataset of chess games
pretrain *args:
    uv run scripts/training/pretrain.py {{args}} seed={{SEED}}


# Download and setup lichess-bot client
bot-setup:
    @if [ ! -d "lichess-bot" ]; then \
        echo "Cloning lichess-bot..."; \
        git clone {{LICHESS_BOT_REPO}}; \
    fi
    @echo "Pinning lichess-bot to {{LICHESS_BOT_REF}}..."
    @cd lichess-bot && git fetch --depth 1 origin {{LICHESS_BOT_REF}} && git checkout --detach FETCH_HEAD
    @echo "Creating isolated virtual environment for lichess-bot..."
    @cd lichess-bot && uv venv .venv
    @echo "Installing lichess-bot dependencies into lichess-bot/.venv..."
    @cd lichess-bot && uv pip install --python .venv/bin/python -r requirements.txt
    @echo "Upgrading account to bot..."
    @curl -s -X POST https://lichess.org/api/bot/account/upgrade -H "Authorization: Bearer ${LICHESS_BOT_TOKEN}" || echo "Account may already be a bot or token is invalid"

# Run the bot locally (requires .env with LICHESS_BOT_TOKEN)
# Usage:
#   just bot-run                       # uses mock (random moves), prints WARNING
#   just bot-run artifacts/pretrain/... # uses model from artifact directory
# Notes:
#   - model_path must be a directory containing model.pt and config.json
#   - path is resolved relative to project root, then passed as absolute path
bot-run +model_path='':
    @if [ ! -x "lichess-bot/.venv/bin/python" ]; then \
        echo "Missing lichess-bot venv. Run: just bot-setup"; \
        exit 1; \
    fi
    @if [ ! -x ".venv/bin/python" ]; then \
        echo "Missing project venv for engine. Run: just setup"; \
        exit 1; \
    fi
    @if [ "$(uname)" = "Darwin" ]; then \
        cp config/config.yml.example lichess-bot/config.yml && \
        sed -i '' "s|TOKEN_PLACEHOLDER|${LICHESS_BOT_TOKEN}|g" lichess-bot/config.yml && \
        sed -i '' "s|ENGINE_INTERPRETER_PLACEHOLDER|../.venv/bin/python|g" lichess-bot/config.yml; \
    else \
        cp config/config.yml.example lichess-bot/config.yml && \
        sed -i "s|TOKEN_PLACEHOLDER|${LICHESS_BOT_TOKEN}|g" lichess-bot/config.yml && \
        sed -i "s|ENGINE_INTERPRETER_PLACEHOLDER|../.venv/bin/python|g" lichess-bot/config.yml; \
    fi
    @echo "Starting bot..."
    @if [ "{{model_path}}" = "" ]; then \
        cd lichess-bot && LICHESS_BOT_TOKEN=${LICHESS_BOT_TOKEN} KRASNAL_ENGINE_PROVIDER=mock .venv/bin/python lichess-bot.py; \
    else \
        bot_model_path=$(realpath {{model_path}}) && \
        cd lichess-bot && LICHESS_BOT_TOKEN=${LICHESS_BOT_TOKEN} KRASNAL_MODEL_ARTIFACT_DIR=${bot_model_path} KRASNAL_ENGINE_PROVIDER=model .venv/bin/python lichess-bot.py; \
    fi

# Remove everything related to local lichess-bot setup
bot-clean:
    @echo "Cleaning lichess-bot runtime artifacts and repository..."
    @rm -rf lichess-bot

# Remvoe dataset hf-cache
hf-cache-clean:
    hf cache rm dataset/thomasd1/aix-lichess-database -y

clean:
    rm -rf *.log wandb/ artifacts/ outputs/ .uv_cache/ .hf_cache/ .hydra/ .ruff_cache/ .pytest_cache
