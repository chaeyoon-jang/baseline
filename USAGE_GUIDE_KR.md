# RTSoG를 이용한 FINAL/test.json 평가 가이드

## 개요
이 가이드는 `/mnt/home/chaeyun-jang/baseline/FINAL/test.json` 데이터셋을 이용하여 RTSoG(Reward-guided Tree Search for KGQA) 모델을 실행하고 평가 결과(Hit@1, F1 등)를 얻는 방법을 설명합니다.

## 변경 사항 요약

### 1. **evaluate_v2.py** 수정
- JSONL 형식의 데이터셋을 직접 로드할 수 있도록 `--dataset_path` 파라미터 추가
- `--start_idx`, `--end_idx`를 통한 샘플 범위 제어 지원

### 2. **split_json.py** 수정
- CLI 인터페이스 추가: `--input`, `--mode`, `--topk` 옵션 지원
- 트리 출력을 shortcut.json으로 변환

### 3. **answer_generation.py** 수정
- `--datapath_list` 옵션으로 여러 shortcut.json 파일 처리 가능
- `BooleanOptionalAction` 사용으로 부울 플래그 개선

### 4. **utils/tools.py** 수정
- JSONL 데이터셋 로더 함수 `load_jsonl_kgqa_dataset()` 추가
- 누락된 그래프/엔티티 샘플 자동 스킵

### 5. **scripts/run_final_eval.sh** 생성
- 전체 파이프라인 자동화 (트리 검색 → 단축어 추출 → 답변 생성)

## 사용 방법

### 기본 실행 (전체 파이프라인)

```bash
cd /mnt/home/chaeyun-jang/baseline/MCTS-KGQAv2
bash scripts/run_final_eval.sh
```

### 환경 변수를 이용한 커스터마이징

```bash
# GPU 0번 사용, 처음 100개 샘플 처리
DATASET_PATH=/mnt/home/chaeyun-jang/baseline/FINAL/test.json \
TASK_NAME=final \
PROPOSE_METHOD=qwen14b \
START_IDX=0 \
END_IDX=100 \
GPU_ID=0 \
bash scripts/run_final_eval.sh
```

### 단계별 실행

#### Step 1: 추론 트리 생성

```bash
cd /mnt/home/chaeyun-jang/baseline/MCTS-KGQAv2
CUDA_VISIBLE_DEVICES=0 python evaluate_v2.py \
    --task_name final \
    --propose_method qwen14b \
    --dataset_path /mnt/home/chaeyun-jang/baseline/FINAL/test.json \
    --start_idx 0 \
    --end_idx 100 \
    --shuffle \
    --shuffle_times 2 \
    --num_plan_branch 7 \
    --num_branch 3 \
    --iteration_limit 40
```

출력: `outputs/final/mcts/qwen14b/*_alltree.json`

#### Step 2: shortcut.json 생성

```bash
python split_json.py --mode shortcut --input outputs/final/mcts/qwen14b/qwen14b-...-alltree.json --topk 10
```

출력: `outputs/final/mcts/qwen14b/qwen14b-.../shortcut.json`

#### Step 3: 답변 생성 및 평가

```bash
CUDA_VISIBLE_DEVICES=0 python answer_generation.py \
    --datapath_list outputs/final/mcts/qwen14b/qwen14b-.../shortcut.json \
    --propose_method qwen14b \
    --no-use_local_method
```

출력: `{output_dir}/{propose_method}_eval_result.jsonl` (Hit, F1, Precision, Recall 등 포함)

## 매개변수 설명

### evaluate_v2.py
| 매개변수 | 설명 | 기본값 |
|---------|------|-------|
| `--dataset_path` | JSONL 형식의 데이터셋 경로 (예: test.json) | None |
| `--data_dir` | parquet 형식의 데이터 디렉토리 | None |
| `--start_idx` | 처리할 샘플의 시작 인덱스 | 0 |
| `--end_idx` | 처리할 샘플의 끝 인덱스 (-1 = 끝까지) | -1 |
| `--task_name` | 작업 이름 (cwq, webqsp, final 등) | cwq |
| `--propose_method` | LLM 모델 선택 | qwen14b |
| `--shuffle` | 셔플 활성화 | False |
| `--shuffle_times` | 셔플 횟수 | 1 |

### split_json.py
| 매개변수 | 설명 | 기본값 |
|---------|------|-------|
| `--input` | 처리할 *_alltree.json 파일 경로 | (필수) |
| `--mode` | 작업 모드: shortcut, split, step | shortcut |
| `--topk` | 유지할 상위 K개 점수 수 | 10 |

### answer_generation.py
| 매개변수 | 설명 | 기본값 |
|---------|------|-------|
| `--datapath_list` | shortcut.json 파일 경로 (여러 개 가능) | (필수) |
| `--propose_method` | 사용할 LLM | deepseekv3 |
| `--use_local_method` | 로컬 모델 사용 여부 | True |
| `--no-use_local_method` | API 사용 옵션 | - |
| `--temperature` | 샘플링 온도 | 0.7 |

## 평가 결과 확인

### 결과 파일 위치
```
outputs/final/mcts/qwen14b/{propose_method}_eval_result.jsonl
outputs/final/mcts/qwen14b/{propose_method}_eval_result.txt
```

### 결과 파일 형식 (JSONL)
```json
{
  "qid": "question_id",
  "question": "What is...",
  "gt_answer": ["answer1", "answer2"],
  "prediction": ["predicted_answer"],
  "final_ans": true,
  "acc": 1.0,
  "hit": 1,
  "f1": 0.95,
  "precision": 1.0,
  "recall": 0.9
}
```

### 평가 지표
- **Hit**: 예측된 답변이 정답 목록에 포함되었는지 여부 (0 또는 1)
- **F1**: 정밀도와 재현율의 조화평균
- **Precision**: 예측된 답변 중 정확한 비율
- **Recall**: 정답 중 예측된 비율
- **Accuracy**: 완전히 일치하는 비율

## 주요 특징

✅ **JSONL 형식 지원**: `/mnt/home/chaeyun-jang/baseline/FINAL/test.json` 직접 처리
✅ **부분 처리**: `--start_idx`, `--end_idx`를 통한 샘플 범위 제어
✅ **자동 에러 처리**: 누락된 그래프/엔티티 샘플 자동 스킵
✅ **평가 자동화**: Hit@1, F1, Precision, Recall 등 자동 계산
✅ **쉘 스크립트**: 전체 파이프라인 자동화

## 문제 해결

### Dataset not found
```bash
ls -la /mnt/home/chaeyun-jang/baseline/FINAL/test.json
```

### CUDA 에러
```bash
# CUDA_VISIBLE_DEVICES 설정 확인
nvidia-smi
CUDA_VISIBLE_DEVICES=0 python ...
```

### 메모리 부족
```bash
# 샘플 범위 축소
--start_idx 0 --end_idx 50
```

## 예시: 전체 파이프라인 실행

```bash
#!/bin/bash
cd /mnt/home/chaeyun-jang/baseline/MCTS-KGQAv2

# 변수 설정
export DATASET_PATH=/mnt/home/chaeyun-jang/baseline/FINAL/test.json
export TASK_NAME=final
export PROPOSE_METHOD=qwen14b
export GPU_ID=0
export START_IDX=0
export END_IDX=100

# 실행
bash scripts/run_final_eval.sh

# 결과 확인
cat outputs/$TASK_NAME/mcts/$PROPOSE_METHOD/${PROPOSE_METHOD}_eval_result.txt
```

---
**작성일**: 2025년 1월
**버전**: RTSoG v2 with JSON support
