#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data/rog_webqsp/chunks}"
DATASET_NAME="${DATASET_NAME:-webqsp}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/webqsp}"
PROPOSE_METHOD="${PROPOSE_METHOD:-qwen14b}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "${OUTPUT_DIR}/logs"
LOG_FILE="${OUTPUT_DIR}/logs/webqsp_chunk_0_$(date +"%Y_%m_%d_%H_%M_%S").log"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python "${ROOT_DIR}/evaluate_v2.py" \
  --task_name webqsp \
  --data_path "${DATA_DIR}/${DATASET_NAME}_chunk_0.json" \
  --propose_method "${PROPOSE_METHOD}" \
  --use_freebase \
  --shuffle \
  --shuffle_times 2 \
  --num_plan_branch 7 \
  --num_branch 4 \
  --iteration_limit 7 \
  > "${LOG_FILE}"
