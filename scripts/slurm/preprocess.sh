#!/bin/bash
#SBATCH --job-name=krasnal-preprocess
#SBATCH --output=output/%j_preprocess.out
#SBATCH --error=output/%j_preprocess.err
#SBATCH --cpus-per-task=12
#SBATCH --partition=student-cpu
#SBATCH --time=04:00:00

USER=ijakus
USER_DIRECTORY=/Ziob/$USER
PROJECT_ROOT=$USER_DIRECTORY/krasnal

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

uv run scripts/data/preprocess.py preprocess_workers=4
