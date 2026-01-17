#!/bin/bash
# RTSoG 전체 평가 파이프라인 - 빠른 시작 예시
# 사용법: bash quick_eval.sh [SAMPLE_COUNT] [GPU_ID]

set -euo pipefail

SAMPLE_COUNT=${1:-10}  # 기본값: 10개 샘플
GPU_ID=${2:-0}        # 기본값: GPU 0

PROJECT_ROOT="/mnt/home/chaeyun-jang/baseline/MCTS-KGQAv2"
DATASET_PATH="/mnt/home/chaeyun-jang/baseline/FINAL/test.json"

echo "================================"
echo "RTSoG 평가 파이프라인 시작"
echo "================================"
echo "데이터셋: $DATASET_PATH"
echo "샘플 수: $SAMPLE_COUNT"
echo "GPU ID: $GPU_ID"
echo ""

# Step 1: 트리 검색
echo "[Step 1/3] 추론 트리 생성 중..."
CUDA_VISIBLE_DEVICES=$GPU_ID python "$PROJECT_ROOT/evaluate_v2.py" \
    --task_name final \
    --propose_method qwen14b \
    --dataset_path "$DATASET_PATH" \
    --start_idx 0 \
    --end_idx "$SAMPLE_COUNT" \
    --shuffle \
    --shuffle_times 2 \
    --num_plan_branch 7 \
    --num_branch 3 \
    --iteration_limit 40

# Step 2: shortcut 추출
echo ""
echo "[Step 2/3] shortcut.json 생성 중..."
TREE_DIR="$PROJECT_ROOT/outputs/final/mcts/qwen14b"
LATEST_TREE=$(ls -t "$TREE_DIR"/*_alltree.json 2>/dev/null | head -n 1)

if [ -z "$LATEST_TREE" ]; then
    echo "오류: *_alltree.json 파일을 찾을 수 없습니다"
    exit 1
fi

python "$PROJECT_ROOT/split_json.py" --mode shortcut --input "$LATEST_TREE" --topk 10

# Step 3: 답변 생성
echo ""
echo "[Step 3/3] 답변 생성 및 평가 중..."
SHORTCUT_FILE="${LATEST_TREE%.json}/shortcut.json"

CUDA_VISIBLE_DEVICES=$GPU_ID python "$PROJECT_ROOT/answer_generation.py" \
    --datapath_list "$SHORTCUT_FILE" \
    --propose_method qwen14b \
    --no-use_local_method

# 결과 출력
echo ""
echo "================================"
echo "평가 완료!"
echo "================================"

RESULT_FILE="$TREE_DIR/qwen14b_eval_result.txt"
if [ -f "$RESULT_FILE" ]; then
    echo ""
    echo "결과 요약:"
    tail -n 1 "$RESULT_FILE"
    echo ""
    echo "상세 결과 파일:"
    echo "  JSONL: ${SHORTCUT_FILE%/*}/qwen14b_eval_result.jsonl"
    echo "  텍스트: $RESULT_FILE"
else
    echo "경고: 결과 파일을 찾을 수 없습니다"
fi
