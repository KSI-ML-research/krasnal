#!/bin/bash
#SBATCH --job-name=krasnal-download-games
#SBATCH --output=output/%j_download_games.out
#SBATCH --error=output/%j_download_games.err
#SBATCH --cpus-per-task=4
#SBATCH --partition=student-cpu
#SBATCH --time=01:00:00

# IMPORTANT: Ensure HF_TOKEN is set in your ~/.bashrc on HPC

USER=ijakus
USER_DIRECTORY=/Ziob/$USER
PROJECT_ROOT=$USER_DIRECTORY/krasnal

GAMES=20_000_000

source ~/.bashrc
cd $PROJECT_ROOT

mkdir -p output

export XDG_RUNTIME_DIR=/tmp/$USER/runtime
export UV_CACHE_DIR=$USER_DIRECTORY/.cache/uv
export UV_LINK_MODE=copy
export HF_HOME=$USER_DIRECTORY/.cache/huggingface
mkdir -p $XDG_RUNTIME_DIR
mkdir -p $UV_CACHE_DIR
mkdir -p $HF_HOME

test -d .venv || uv venv .venv --python 3.13
uv sync

uv run scripts/data/download_games.py target_games=$GAMES require_evals=false
