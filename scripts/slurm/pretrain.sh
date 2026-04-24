#!/bin/bash
#SBATCH --job-name=krasnal-pretrain
#SBATCH --output=output/%j_train.out
#SBATCH --error=output/%j_train.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=student-nvidia
#SBATCH --time=24:00:00

# IMPORTANT: Ensure WANDB_API_KEY is set in your ~/.bashrc on HPC
# export WANDB_API_KEY=your_key_here

# === User & Paths ===
USER=ijakus
USER_DIRECTORY=/Ziob/$USER
PROJECT_ROOT=$USER_DIRECTORY/krasnal
STOCKFISH_PATH=$USER_DIRECTORY/stockfish

# === Config ===
MODEL=xsmall
TRAIN_CONFIG=cuda

# === Stockfish Download ===
STOCKFISH_URL=https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-ubuntu-x86-64-avx2.tar

source ~/.bashrc
cd $PROJECT_ROOT

mkdir -p output

# Fix for XDG_RUNTIME_DIR and UV cache permission
export XDG_RUNTIME_DIR=/tmp/$USER/runtime
export UV_CACHE_DIR=$USER_DIRECTORY/.cache/uv
export UV_LINK_MODE=copy
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

uv run python scripts/pretrain.py \
    model=$MODEL \
    train=$TRAIN_CONFIG \
    stockfish_path=$STOCKFISH_PATH
