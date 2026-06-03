#!/bin/bash
set -euo pipefail

CLUSTER_USER=${KRASNAL_CLUSTER_USER:-ijakus}
USER_DIRECTORY=/Ziob/$CLUSTER_USER
PROJECT_ROOT=$USER_DIRECTORY/krasnal

source ~/.bashrc
cd "$PROJECT_ROOT"

mkdir -p output

export XDG_RUNTIME_DIR=/tmp/$CLUSTER_USER/runtime
export UV_CACHE_DIR=$USER_DIRECTORY/.cache/uv
export UV_LINK_MODE=copy
export HF_HOME=$USER_DIRECTORY/.cache/huggingface
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
mkdir -p "$XDG_RUNTIME_DIR" "$UV_CACHE_DIR" "$HF_HOME"

test -d .venv || uv venv .venv --python 3.13
uv sync

if [ -z "${RUN_NAME:-}" ] || [ -z "${RUN_OVERRIDES:-}" ]; then
    echo "RUN_NAME and RUN_OVERRIDES must be exported by the scheduler." >&2
    exit 1
fi

export WANDB_RUN_GROUP=${RUN_GROUP:-krasnal-ablation}
export WANDB_NAME="${RUN_NAME}-${SLURM_JOB_ID}"

echo "Running ablation: $RUN_NAME"
echo "Overrides: $RUN_OVERRIDES"
echo "Tokenized dir: ${KRASNAL_TOKENIZED_DIR:-data/2_tokenized}"

uv run torchrun --standalone --nproc_per_node="${RUN_NPROC:-1}" scripts/training/pretrain.py \
    train=cuda \
    train.num_workers="${RUN_NUM_WORKERS:-4}" \
    $RUN_OVERRIDES
