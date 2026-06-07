#!/bin/bash
#SBATCH --job-name=krasnal-preprocess
#SBATCH --output=output/%j_preprocess.out
#SBATCH --error=output/%j_preprocess.err
#SBATCH --cpus-per-task=12
#SBATCH --partition=student-cpu
#SBATCH --time=08:00:00

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
export POLARS_MAX_THREADS=2
mkdir -p $XDG_RUNTIME_DIR
mkdir -p $UV_CACHE_DIR
mkdir -p $HF_HOME

test -d .venv || uv venv .venv --python 3.13
uv sync

echo "Filtered dir: ${KRASNAL_FILTERED_DIR:-data/1_filtered}"
echo "Tokenized dir: ${KRASNAL_TOKENIZED_DIR:-data/2_tokenized}"
echo "Preprocess overrides: ${PREPROCESS_EXTRA_ARGS:-none}"
uv run scripts/data/preprocess.py ${PREPROCESS_EXTRA_ARGS:-}
