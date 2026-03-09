# MedGemma Fine-tuning with LoRA

MedGemma-4b-it 모델을 방사선 보고서 생성 작업에 맞춰 LoRA(Low-Rank Adaptation)로 fine-tuning하는 프로젝트입니다.

## 프로젝트 구조

```
challenge/
├── train.py          # LoRA fine-tuning 학습 스크립트
├── evaluation.py     # 모델 평가 스크립트
├── models/           # 다운로드한 모델 저장 디렉토리
├── datasets/         # 다운로드한 데이터셋 저장 디렉토리
├── outputs/          # 학습된 모델 저장 디렉토리
└── checkpoints/      # 체크포인트 저장 디렉토리
```

## 설치

### 1. 의존성 설치

```bash
pip install torch transformers datasets peft accelerate mlflow
pip install rouge-score bert-score pandas numpy tqdm
pip install huggingface_hub
```

### 2. HuggingFace 토큰 설정

```bash
export HF_TOKEN="your_huggingface_token_here"
```

또는 `--hf-token` 인자로 전달:

```bash
python train.py --hf_token "your_token"
```

## 사용법

### 1. 모델 및 데이터셋 다운로드

```bash
python train.py --download-model --download-dataset
```

### 2. 테스트 실행 (소량 데이터)

```bash
python train.py \
    --model-dir ./models/medgemma-4b-it \
    --dataset-id rajpurkarlab/ReXGradient-160K \
    --train-size 100 \
    --val-size 20 \
    --epochs 1 \
    --mlflow-experiment test-run
```

### 3. 전체 학습

```bash
python train.py \
    --model-dir ./models/medgemma-4b-it \
    --dataset-id rajpurkarlab/ReXGradient-160K \
    --epochs 3 \
    --batch-size 4 \
    --learning-rate 2e-4 \
    --mlflow-experiment medgemma-finetune
```

### 4. 커스텀 CSV 파일로 학습

```bash
python train.py \
    --model-dir ./models/medgemma-4b-it \
    --data-path ./my_radiology_reports.csv \
    --findings-col "Findings" \
    --impression-col "Impression" \
    --epochs 3
```

CSV 파일 형식 예시:

```csv
Findings,Impression
"The lungs are clear...", "Normal chest radiograph..."
"There is a focal consolidation...", "Right lower lobe pneumonia..."
```

### 5. 학습된 모델 평가

```bash
python evaluation.py \
    --model-dir ./outputs \
    --data-path ./test_data.csv \
    --sample-size 50
```

## 하이퍼파라미터

### 주요 하이퍼파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--epochs` | 3 | 학습 에포크 수 |
| `--batch-size` | 4 | 배치 사이즈 |
| `--learning-rate` | 2e-4 | 학습률 |
| `--gradient-accumulation-steps` | 4 | 그라디언트 누적 스텝 |
| `--max-length` | 512 | 최대 시퀀스 길이 |

### LoRA 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--lora-r` | 8 | LoRA rank |
| `--lora-alpha` | 16 | LoRA alpha |
| `--lora-dropout` | 0.05 | LoRA dropout |
| `--target-modules` | q_proj k_proj v_proj o_proj | LoRA를 적용할 모듈 |

## MLFlow 추적

학습 중 MLFlow로 실험을 추적할 수 있습니다:

```bash
# MLFlow UI 시작
mlflow ui

# 브라우저에서 열기
# http://localhost:5000
```

로그되는 메트릭:
- Training loss
- Validation loss
- Learning rate
- Epoch time

## 명령행 인자 전체 목록

```bash
# 모델 & 데이터
--model-id                 # HuggingFace 모델 ID (default: google/medgemma-4b-it)
--model-dir                # 로컬 모델 디렉토리 (default: ./models/medgemma-4b-it)
--dataset-id               # HuggingFace 데이터셋 ID (default: rajpurkarlab/ReXGradient-160K)
--dataset-dir              # 로컬 데이터셋 디렉토리 (default: ./datasets/ReXGradient-160K)
--data-path                # 커스텀 CSV 파일 경로
--findings-col             # CSV의 findings 컬럼명 (default: findings)
--impression-col           # CSV의 impression 컬럼명 (default: impression)

# 다운로드
--download-model           # 모델 다운로드
--download-dataset         # 데이터셋 다운로드
--hf-token                 # HuggingFace API 토큰

# 학습
--epochs                   # 학습 에포크 수 (default: 3)
--batch-size               # 배치 사이즈 (default: 4)
--learning-rate            # 학습률 (default: 2e-4)
--gradient-accumulation-steps  # 그라디언트 누적 스텝 (default: 4)
--warmup-ratio             # Warmup 비율 (default: 0.1)
--max-length               # 최대 시퀀스 길이 (default: 512)

# LoRA
--lora-r                   # LoRA rank (default: 8)
--lora-alpha               # LoRA alpha (default: 16)
--lora-dropout             # LoRA dropout (default: 0.05)
--target-modules           # 타겟 모듈 (default: q_proj k_proj v_proj o_proj)

# 데이터 처리
--train-split              # 학습용 데이터 분할 (default: train)
--val-split                # 검증용 데이터 분할 (default: validation)
--train-size               # 학습 데이터 제한 샘플 수
--val-size                 # 검증 데이터 제한 샘플 수
--seed                     # 랜덤 시드 (default: 42)

# 출력 & 로깅
--output-dir               # 최종 모델 출력 디렉토리 (default: ./outputs)
--checkpoint-dir           # 체크포인트 디렉토리 (default: ./checkpoints)
--mlflow-experiment        # MLFlow 실험명 (default: medgemma-finetune)
--no-mlflow                # MLFlow 로깅 비활성화
```

## 데이터셋

### ReXGradient-160K

기본 데이터셋은 `rajpurkarlab/ReXGradient-160K`입니다:

- **형식**: HuggingFace Datasets
- **컬럼**: `findings`, `impression`
- **크기**: 약 160,000개의 방사선 보고서
- **분할**: train, validation

### 커스텀 CSV

자신만의 CSV 파일을 사용할 수 있습니다:

```csv
findings,impression
"Findings text here...","Impression text here..."
```

## 문제 해결

### CUDA Out of Memory

```bash
# 배치 사이즈 감소
python train.py --batch-size 2 --gradient-accumulation-steps 8

# 또는 max-length 감소
python train.py --max-length 256
```

### 데이터셋 로딩 오류

```bash
# 데이터셋을 먼저 다운로드
python train.py --download-dataset --dataset-dir ./datasets/ReXGradient-160K
```

### MLFlow 연결 오류

```bash
# MLFlow 비활성화
python train.py --no-mlflow
```

## 성능 최적화 팁

1. **Gradient Accumulation**: 메모리가 부족할 때 `--gradient-accumulation-steps` 증가
2. **Mixed Precision**: `bfloat16`이 기본으로 활성화되어 있어 더 빠른 학습
3. **Batch Size**: GPU 메모리에 따라 조정 (A100: 8-16, V100: 4-8)
4. **LoRA Rank**: `r=8`은 좋은 시작점, 더 큰 `r`은 더 많은 파라미터 학습

## 참고 자료

- [MedGemma Paper](https://arxiv.org/abs/2401.11801)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Transformers Documentation](https://huggingface.co/docs/transformers)

## 라이선스

이 프로젝트는 Apache 2.0 라이선스를 따릅니다.
