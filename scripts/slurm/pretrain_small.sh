#!/bin/bash
#SBATCH --job-name=krasnal-pretrain-small
#SBATCH --output=output/%j_pretrain_small.out
#SBATCH --error=output/%j_pretrain_small.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --partition=student-nvidia
#SBATCH --time=06:00:00

# IMPORTANT: Ensure WANDB_API_KEY and HF_TOKEN are set in your ~/.bashrc on HPC

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

uv run torchrun --standalone --nproc_per_node=2 scripts/training/pretrain.py \
    model=small \
    train=cuda \
    train.batch_size=64
