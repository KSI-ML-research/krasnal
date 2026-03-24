# load .env variables for every command
set dotenv-load := true

SEED := "42"
LICHESS_BOT_REPO := "https://github.com/lichess-bot-devs/lichess-bot.git"
# pinned lichess bot commit so that the setup is deterministic
LICHESS_BOT_REF := "96a8f74d87a42db8039e847548fec0d9528bb079"
export PYTHONPATH := "."
export UV_CACHE_DIR := ".uv_cache"

# Install dependencies and setup pre-commit hooks
setup:
    uv sync
    uv run pre-commit install
    cargo build

# --- Lichess Bot Integration ---

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

# Run the bot locally (requires .env with LICHESS_BOT_TOKEN)
bot-run:
    @if [ ! -x "lichess-bot/.venv/bin/python" ]; then \
        echo "Missing lichess-bot venv. Run: just bot-setup"; \
        exit 1; \
    fi
    @if [ ! -x ".venv/bin/python" ]; then \
        echo "Missing project venv for engine. Run: just setup"; \
        exit 1; \
    fi
    @echo "Preparing configuration..."
    @cp config.yml.example lichess-bot/config.yml
    @sed -i '' "s|TOKEN_PLACEHOLDER|${LICHESS_BOT_TOKEN}|g" lichess-bot/config.yml
    @sed -i '' "s|ENGINE_INTERPRETER_PLACEHOLDER|../.venv/bin/python|g" lichess-bot/config.yml
    @echo "Starting bot..."
    @cd lichess-bot && .venv/bin/python lichess-bot.py

# Remove everything related to local lichess-bot setup
bot-clean:
    @echo "Cleaning lichess-bot runtime artifacts and repository..."
    @rm -rf lichess-bot


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

# ===== PUZZLES =====

# Download Lichess puzzle database (~1GB)
download-puzzles:
    mkdir -p data
    curl -L --progress-bar \
        "https://database.lichess.org/lichess_db_puzzle.csv.zst" \
        -o data/lichess_db_puzzle.csv.zst

# Filter puzzles by rating and export to JSONL
prepare-puzzles:
    cargo run --release --bin prepare-puzzles
    
# Download games for specific chess players from PGN Mentor
download-player-games *args:
    cargo run --release --bin download-player-games -- {{args}}
