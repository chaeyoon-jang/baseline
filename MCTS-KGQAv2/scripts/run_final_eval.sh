#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_PATH=${DATASET_PATH:-/mnt/home/chaeyun-jang/baseline/FINAL/test.json}
TASK_NAME=${TASK_NAME:-final}
PROPOSE_METHOD=${PROPOSE_METHOD:-qwen14b}
START_IDX=${START_IDX:-0}
END_IDX=${END_IDX:--1}
TOPK=${TOPK:-10}
GPU_ID=${GPU_ID:-0}
USE_LOCAL_METHOD=${USE_LOCAL_METHOD:-false}

if [ ! -f "$DATASET_PATH" ]; then
    echo "Dataset not found at $DATASET_PATH" >&2
    exit 1
fi

OUTPUT_ROOT="$PROJECT_ROOT/outputs/$TASK_NAME/mcts/$PROPOSE_METHOD"
mkdir -p "$OUTPUT_ROOT"

LOG_DIR="$PROJECT_ROOT/outputs/$TASK_NAME/logs"
mkdir -p "$LOG_DIR"

echo "Running tree search on $DATASET_PATH (samples ${START_IDX}-${END_IDX})"
CUDA_VISIBLE_DEVICES=$GPU_ID python "$PROJECT_ROOT/evaluate_v2.py" \
    --task_name "$TASK_NAME" \
    --propose_method "$PROPOSE_METHOD" \
    --dataset_path "$DATASET_PATH" \
    --start_idx "$START_IDX" \
    --end_idx "$END_IDX" \
    --shuffle \
    --shuffle_times 2 \
    --num_plan_branch 7 \
    --num_branch 3 \
    --iteration_limit 40

TREE_DIR="$PROJECT_ROOT/outputs/$TASK_NAME/mcts/$PROPOSE_METHOD"
shopt -s nullglob
TREE_FILES=("$TREE_DIR"/*_alltree.json)
shopt -u nullglob
if [ ${#TREE_FILES[@]} -eq 0 ]; then
    echo "No *_alltree.json files found in $TREE_DIR" >&2
    exit 1
fi
LATEST_TREE=$(ls -t "${TREE_FILES[@]}" | head -n 1)

python "$PROJECT_ROOT/split_json.py" --mode shortcut --input "$LATEST_TREE" --topk "$TOPK"
SHORTCUT_FILE="${LATEST_TREE%.json}/shortcut.json"
if [ ! -f "$SHORTCUT_FILE" ]; then
    echo "shortcut.json was not generated at $SHORTCUT_FILE" >&2
    exit 1
fi

echo "Generating answers from $SHORTCUT_FILE"
if [ "$USE_LOCAL_METHOD" = true ]; then
    LOCAL_FLAG="--use_local_method"
else
    LOCAL_FLAG="--no-use_local_method"
fi

CUDA_VISIBLE_DEVICES=$GPU_ID python "$PROJECT_ROOT/answer_generation.py" \
    --datapath_list "$SHORTCUT_FILE" \
    --propose_method "$PROPOSE_METHOD" \
    $LOCAL_FLAG
