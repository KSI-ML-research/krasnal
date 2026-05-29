#!/bin/bash

# Define common directories/configurations
USER=ijakus
USER_DIRECTORY=/Ziob/$USER
PROJECT_ROOT=$USER_DIRECTORY/krasnal

# ---------------------------------------------------------
# Step 1: Handle Interactive / CLI Argument Input
# ---------------------------------------------------------
if [ -z "$SLURM_JOB_ID" ]; then
    MODEL=$1
    EPOCHS=$2

    # Prompt for Model Size if not provided or invalid
    while [[ ! "$MODEL" =~ ^(small|medium|large)$ ]]; do
        read -p "Enter model size (small/medium/large): " MODEL
        MODEL=$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')
    done

    # Prompt for Epochs if not provided or invalid
    while [[ -z "$EPOCHS" || ! "$EPOCHS" =~ ^[0-9]+(\.[0-9]+)?$ ]]; do
        read -p "Enter number of epochs (e.g., 0.1, 1, 3): " EPOCHS
    done

    # Set parameters dynamically based on model size
    if [ "$MODEL" = "small" ]; then
        TIME_LIMIT="6:00:00"
        BATCH_SIZE=192
        NUM_WORKERS=4
        TRAIN_EXTRA_ARGS=""
    elif [ "$MODEL" = "medium" ]; then
        TIME_LIMIT="12:00:00"
        BATCH_SIZE=128
        NUM_WORKERS=4
        TRAIN_EXTRA_ARGS=""
    else
        TIME_LIMIT="24:00:00"
        BATCH_SIZE=32
        NUM_WORKERS=4
        TRAIN_EXTRA_ARGS="train.learning_rate=2.5e-4 train.min_lr=2.5e-5 train.warmup_iters=1000"
    fi

    echo "Submitting Slurm pretraining job..."
    echo "  - Model: $MODEL"
    echo "  - Epochs: $EPOCHS"
    echo "  - Batch Size: $BATCH_SIZE"
    echo "  - Num Workers: $NUM_WORKERS"
    echo "  - Time Limit: $TIME_LIMIT"
    echo "  - Extra Args: ${TRAIN_EXTRA_ARGS:-none}"

    sbatch \
      --job-name="krasnal-pretrain-${MODEL}" \
      --output="output/%j_pretrain_${MODEL}.out" \
      --error="output/%j_pretrain_${MODEL}.err" \
      --gres="gpu:2" \
      --cpus-per-task=24 \
      --partition="student-nvidia" \
      --time="${TIME_LIMIT}" \
      --export="ALL,RUN_MODEL=${MODEL},RUN_EPOCHS=${EPOCHS},RUN_BATCH_SIZE=${BATCH_SIZE},RUN_NUM_WORKERS=${NUM_WORKERS},RUN_TRAIN_EXTRA_ARGS=${TRAIN_EXTRA_ARGS}" \
      "$0"

    exit 0
fi

# ---------------------------------------------------------
# Step 2: Slurm Compute Node Execution
# ---------------------------------------------------------
source ~/.bashrc
cd $PROJECT_ROOT

mkdir -p output

export XDG_RUNTIME_DIR=/tmp/$USER/runtime
export UV_CACHE_DIR=$USER_DIRECTORY/.cache/uv
export UV_LINK_MODE=copy
export HF_HOME=$USER_DIRECTORY/.cache/huggingface
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
mkdir -p $XDG_RUNTIME_DIR
mkdir -p $UV_CACHE_DIR
mkdir -p $HF_HOME

test -d .venv || uv venv .venv --python 3.13
uv sync

uv run torchrun --standalone --nproc_per_node=2 scripts/training/pretrain.py \
    model="${RUN_MODEL}" \
    train=cuda \
    train.epochs="${RUN_EPOCHS}" \
    train.batch_size="${RUN_BATCH_SIZE}" \
    train.num_workers="${RUN_NUM_WORKERS}" \
    ${RUN_TRAIN_EXTRA_ARGS}
