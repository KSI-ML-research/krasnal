# load .env variables for every command
set dotenv-load := true

SEED := "42"
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
        git clone --depth 1 https://github.com/lichess-bot-devs/lichess-bot.git; \
    fi
    @echo "Installing lichess-bot dependencies..."
    cd lichess-bot && uv pip install -r requirements.txt

# Run the bot locally (requires .env with LICHESS_BOT_TOKEN)
bot-run:
    @echo "Preparing configuration..."
    @cp config.yml.example lichess-bot/config.yml
    @sed -i '' "s|TOKEN_PLACEHOLDER|${LICHESS_BOT_TOKEN}|g" lichess-bot/config.yml
    @echo "Starting bot..."
    @cd lichess-bot && PYTHONPATH=../src uv run python lichess-bot.py


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
