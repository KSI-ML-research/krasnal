#!/bin/bash
#SBATCH --job-name=krasnal-stockfish
#SBATCH --output=output/%j_stockfish.out
#SBATCH --error=output/%j_stockfish.err
#SBATCH --cpus-per-task=8
#SBATCH --partition=student-cpu
#SBATCH --time=24:00:00

# === User & Paths ===
USER=ijakus
USER_DIRECTORY=/Ziob/$USER
PROJECT_ROOT=$USER_DIRECTORY/krasnal
STOCKFISH_PATH=$USER_DIRECTORY/stockfish

# === Config ===
NUM_PRODUCERS=8
DEPTH=10

# === Stockfish Download ===
STOCKFISH_URL=https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-ubuntu-x86-64-avx2.tar

source ~/.bashrc
cd $PROJECT_ROOT

mkdir -p output

# Fix for XDG_RUNTIME_DIR and UV cache permission
export XDG_RUNTIME_DIR=/tmp/$USER/runtime
export UV_CACHE_DIR=$USER_DIRECTORY/.cache/uv
mkdir -p $XDG_RUNTIME_DIR
mkdir -p $UV_CACHE_DIR

# Create venv with Python 3.13 and install dependencies
uv venv .venv --python 3.13
uv sync

# Download Stockfish if not present
if [[ ! -f $STOCKFISH_PATH ]]; then
    cd $USER_DIRECTORY
    srun wget -O stockfish.tar $STOCKFISH_URL
    srun tar -xf stockfish.tar
    # Find and move the stockfish binary
    srun bash -c "mv \$(find . -maxdepth 1 -type f -name '*stockfish*' | head -1) $STOCKFISH_PATH"
    rm -f stockfish.tar
    cd $PROJECT_ROOT
fi

uv run python scripts/sft_generate.py \
    stockfish_path=$STOCKFISH_PATH \
    num_producers=$NUM_PRODUCERS \
    depth=$DEPTH
