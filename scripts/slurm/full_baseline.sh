#!/bin/bash
#SBATCH --job-name=krasnal-full-baseline
#SBATCH --output=output/%j_full_baseline.out
#SBATCH --error=output/%j_full_baseline.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --partition=student-nvidia
#SBATCH --time=36:00:00

set -euo pipefail

CLUSTER_USER=${KRASNAL_CLUSTER_USER:-ijakus}
USER_DIRECTORY=/Ziob/$CLUSTER_USER
PROJECT_ROOT=$USER_DIRECTORY/krasnal

TARGET_GAMES=${TARGET_GAMES:-20000000}
MODEL=${RUN_MODEL:-large}
EPOCHS=${RUN_EPOCHS:-3}
BATCH_SIZE=${RUN_BATCH_SIZE:-32}
NUM_WORKERS=${RUN_NUM_WORKERS:-4}
NPROC=${RUN_NPROC:-1}
DATA_NAME=${DATA_NAME:-baseline_${TARGET_GAMES}}

export KRASNAL_FILTERED_DIR=${KRASNAL_FILTERED_DIR:-data/1_filtered_${DATA_NAME}}
export KRASNAL_TOKENIZED_DIR=${KRASNAL_TOKENIZED_DIR:-data/2_tokenized_${DATA_NAME}}
export KRASNAL_ARTIFACT_DIR=${KRASNAL_ARTIFACT_DIR:-artifacts/pretrain/${DATA_NAME}-${MODEL}}

DOWNLOAD_EXTRA_ARGS=${DOWNLOAD_EXTRA_ARGS:-target_games=${TARGET_GAMES}}
PREPROCESS_EXTRA_ARGS=${PREPROCESS_EXTRA_ARGS:-target_games=${TARGET_GAMES} report.enabled=false}
TRAIN_EXTRA_ARGS=${RUN_TRAIN_EXTRA_ARGS:-}
if [ "$MODEL" = "large" ] && [ -z "$TRAIN_EXTRA_ARGS" ]; then
    TRAIN_EXTRA_ARGS="train.learning_rate=2.5e-4 train.min_lr=2.5e-5 train.warmup_iters=1000"
fi

source ~/.bashrc
cd "$PROJECT_ROOT"

mkdir -p output

export XDG_RUNTIME_DIR=/tmp/$CLUSTER_USER/runtime
export UV_CACHE_DIR=$USER_DIRECTORY/.cache/uv
export UV_LINK_MODE=copy
export HF_HOME=$USER_DIRECTORY/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export POLARS_MAX_THREADS=2
mkdir -p "$XDG_RUNTIME_DIR" "$UV_CACHE_DIR" "$HF_HOME"

test -d .venv || uv venv .venv --python 3.13
uv sync

echo "=== Krasnal full baseline job ==="
echo "Target games: $TARGET_GAMES"
echo "Data name: $DATA_NAME"
echo "Filtered dir: $KRASNAL_FILTERED_DIR"
echo "Tokenized dir: $KRASNAL_TOKENIZED_DIR"
echo "Artifact dir: $KRASNAL_ARTIFACT_DIR"
echo "Model: $MODEL"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Workers: $NUM_WORKERS"
echo "NPROC: $NPROC"
echo "Download args: $DOWNLOAD_EXTRA_ARGS"
echo "Preprocess args: $PREPROCESS_EXTRA_ARGS"
echo "Train extra args: ${TRAIN_EXTRA_ARGS:-none}"

echo "=== Download/filter games ==="
uv run scripts/data/download_games.py $DOWNLOAD_EXTRA_ARGS

echo "=== Preprocess games ==="
uv run scripts/data/preprocess.py $PREPROCESS_EXTRA_ARGS

echo "=== Train baseline ==="
export WANDB_RUN_GROUP=${RUN_GROUP:-krasnal-full-baseline}
export WANDB_NAME=${WANDB_NAME:-${DATA_NAME}-${MODEL}-${SLURM_JOB_ID}}

uv run torchrun --standalone --nproc_per_node="$NPROC" scripts/training/pretrain.py \
    model="$MODEL" \
    train=cuda \
    train.epochs="$EPOCHS" \
    train.batch_size="$BATCH_SIZE" \
    train.num_workers="$NUM_WORKERS" \
    $TRAIN_EXTRA_ARGS
