#!/bin/bash
#SBATCH --job-name=krasnal-pipeline
#SBATCH --output=output/%j_pipeline.out
#SBATCH --error=output/%j_pipeline.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --partition=student-nvidia
#SBATCH --time=24:00:00

# IMPORTANT: Ensure WANDB_API_KEY and HF_TOKEN are set in your ~/.bashrc on HPC
# export WANDB_API_KEY=your_key_here
# export HF_TOKEN=your_token_here

# === User & Paths ===
USER=ijakus
USER_DIRECTORY=/Ziob/$USER
PROJECT_ROOT=$USER_DIRECTORY/krasnal

# === Config ===
MODEL=xsmall
TRAIN_CONFIG=cuda

source ~/.bashrc
cd $PROJECT_ROOT

mkdir -p output

# Fix for XDG_RUNTIME_DIR and UV cache permission
export XDG_RUNTIME_DIR=/tmp/$USER/runtime
export UV_CACHE_DIR=$USER_DIRECTORY/.cache/uv
export UV_LINK_MODE=copy
# Hugging Face cache
export HF_HOME=$USER_DIRECTORY/.cache/huggingface
mkdir -p $XDG_RUNTIME_DIR
mkdir -p $UV_CACHE_DIR
mkdir -p $HF_HOME

# Create venv with Python 3.13 and install dependencies
uv venv .venv --python 3.13
uv sync

# Download games
uv run python scripts/data/download_games.py

# Preprocess games
uv run python scripts/data/preprocess.py

# Pretrain model
uv run python scripts/training/pretrain.py \
    model=$MODEL \
    train=$TRAIN_CONFIG
