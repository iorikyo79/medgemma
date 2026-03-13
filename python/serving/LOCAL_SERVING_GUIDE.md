# MedGemma Local Serving Guide

기존 Triton + Gunicorn 스택을 로컬에서 실행하는 방법입니다.

## 전제 조건

- NVIDIA Docker (nvidia-docker2 or nvidia-container-toolkit)
- Docker가 설치되어 있고 실행 중이어야 함
- 최소 12GB VRAM (권장 16GB)

> **참고**: 이 가이드는 개발/디버깅 목적의 로컬 서빙 방법입니다. 프로덕션 배포는 [DOCKER_DEPLOYMENT.md](./DOCKER_DEPLOYMENT.md)를 참조하세요.

## 환경변수

`entrypoint.sh`는 다음 환경변수를 자동으로 기본값 설정합니다. 로컬 실행 시 별도 설정 없이도 동작합니다:

| 환경변수 | 기본값 | 설명 |
|----------|--------|------|
| `AIP_HTTP_PORT` | `8080` | HTTP API 서버 포트 |
| `AIP_HEALTH_ROUTE` | `/health` | 헬스 체크 엔드포인트 |
| `AIP_PREDICT_ROUTE` | `/predict` | 추론 엔드포인트 |
| `ENABLE_CLOUD_LOGGING` | `false` | Cloud Logging 비활성화 (로컬 기본) |
| `CLOUD_OPS_LOG_PROJECT` | `local` | Cloud Ops 로그 프로젝트 (로컬 기본) |

> 원래 이 변수들은 Vertex AI 환경에서 자동 주입됩니다. 로컬 실행 시에는 `entrypoint.sh`가 기본값을 설정하므로 수동 설정이 불필요합니다.

## 빠른 시작

### 1. 서버 시작

```bash
cd /home/jkuhm/src/MoltsBot-Source/medgemma-clone/medgemma/python/serving
./local_serving.sh start
```

서버가 시작되면 다음과 같이 표시됩니다:

```
============================================================
MedGemma Local Serving
============================================================

Model path: /path/to/medgemma-4b-it
HTTP Port: 8080
Model REST Port: 8600

[1/4] Creating Triton model repository...
Model repository created at /tmp/medgemma_triton_repo

[2/4] Starting Triton Inference Server...
Triton container started

[3/4] Waiting for Triton server to be ready...
.........Triton server is ready!

[4/4] Starting Gunicorn API server...

============================================================
Server Ready!
============================================================

API Server: http://localhost:8080
Health Check: http://localhost:8080/health
Predict Endpoint: http://localhost:8080/predict

Triton Server: http://localhost:8600

Press Ctrl+C to stop both servers
```

### 2. 상태 확인

```bash
./local_serving.sh status
```

출력:
```
MedGemma Local Serving Status

Triton Server: Running
triton-medgemma   Up 2 minutes   0.0.0.0:8500->8500/tcp, 0.0.0.0:8600->8000/tcp

Gunicorn Server: Running (PID: 12345)

Testing API...
API Health Check: OK
{
    "device": "cuda",
    "model": "medgemma-4b-it",
    "status": "healthy"
}
```

### 3. 테스트

```bash
./local_serving.sh test
```

### 4. 서버 중지

```bash
./local_serving.sh stop
```

## 사용 가능한 명령어

| 명령어 | 설명 |
|--------|------|
| `./local_serving.sh start` | 모든 서버 시작 |
| `./local_serving.sh stop` | 모든 서버 중지 |
| `./local_serving.sh status` | 서버 상태 확인 |
| `./local_serving.sh logs triton` | Triton 로그 보기 |
| `./local_serving.sh test` | API 테스트 실행 |

## API 사용 예시

서버가 시작되면 OpenAI 호환 API를 사용할 수 있습니다:

### 헬스 체크
```bash
curl http://localhost:8080/health
```

### 텍스트 생성
```bash
curl -X POST http://localhost:8080/predict \
    -H "Content-Type: application/json" \
    -d '{
      "messages": [
        {"role": "user", "content": "What is this image?"}
      ]
    }'
```

### 채팅 완료
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Explain CT scans briefly."}
    ],
    "max_tokens": 150
  }'
```

### 이미지 + 텍스트
```bash
#이미지 + 텍스트:
  curl -X POST http://localhost:8080/predict \
    -H "Content-Type: application/json" \
    -d '{
      "messages": [
        {"role": "user", "content": [
          {"type": "text", "text": "Describe this image"},
          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        ]}
      ]
    }'
```

## 문제 해결

### Docker 권한 오류
```bash
sudo usermod -aG docker $USER
# 로그아웃 후 다시 로그인
```

### NVIDIA Docker가 작동하지 않음
```bash
# NVIDIA Container Toolkit 설치 확인
docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
```

### 포트 이미 사용 중
```bash
# 포트 8080 사용 중인 프로세스 확인
lsof -i :8080

# 포트 8600 사용 중인 프로세스 확인
lsof -i :8600
```

### GPU 메모리 부족
`local_serving.sh`에서 `gpu_memory_utilization` 값 줄이기:
```bash
# 파일 내에서 수정
"gpu_memory_utilization": 0.7,  # 0.85에서 0.7로
```

## 구조

```
local_serving.sh 실행
        │
        ▼
┌─────────────────────────────────────┐
│  Docker Container (Triton)          │
│  - Port: 8500 (gRPC)                │
│  - Port: 8600 (HTTP)                │
│  - vLLM Backend                     │
│  - Model: medgemma-4b-it            │
└─────────────────────────────────────┘
        │ gRPC
        ▼
┌─────────────────────────────────────┐
│  Gunicorn + Flask (Host)            │
│  - Port: 8080                       │
│  - MedGemmaPredictor                │
│  - Data Accessors                   │
│  - Vertex-style API                 │
└─────────────────────────────────────┘
```

### 모델 감지 우선순위

`entrypoint.sh`는 다음 순서로 모델 파일을 찾습니다:

1. `MODEL_SOURCE` 환경변수 → GCS URI로 모델 다운로드
2. `AIP_STORAGE_URI` 환경변수 → Vertex AI 업로드 모델
3. `/serving/models/medgemma-4b-it` 디렉토리 → Docker 이미지 내 번들 모델 (로컬 실행 시 기본)
4. 위 모두 해당 없으면 → HuggingFace에서 `google/medgemma-4b-it` 직접 다운로드

## Simple Serving과의 차이

| 항목 | Local Serving (Triton) | Simple Serving |
|------|----------------------|----------------|
| 성능 | 높음 (vLLM 최적화) | 낮음 (기본 Transformers) |
| 동시 요청 | 지원 (배칭) | 미지원 |
| DICOM 처리 | 지원 | 미지원 |
| 설정 복잡도 | 높음 (Docker 필요) | 낮음 |
| 용도 | 프로덕션 테스트 | 빠른 개발 |

## 참고

- Triton 로그: `docker logs triton-medgemma`
- Gunicorn 로그: 터미널 출력
- 모델 경로: 스크립트 내 `MODEL_PATH` 변수로 수정 가능
- 포트 변경: 스크립트 내 포트 변수로 수정 가능

## 관련 문서

| 문서 | 설명 |
|------|------|
| [DOCKER_DEPLOYMENT.md](./DOCKER_DEPLOYMENT.md) | Docker 이미지 빌드 및 배포 가이드 (프로덕션 권장) |
| [SIMPLE_SERVING_README.md](./SIMPLE_SERVING_README.md) | Docker 없이 Python으로만 실행하는 간이 서빙 (개발용) |
| [medgemma_docker_guide.md](./medgemma_docker_guide.md) | Docker 초보자를 위한 입문 가이드 |

> **프로덕션 배포 시 권장 옵션**: Docker 컨테이너로 배포할 때는 `--shm-size=32g`, `--max-model-len=4096`, `--gpu-memory-utilization=0.9` 옵션을 권장합니다. 자세한 내용은 [DOCKER_DEPLOYMENT.md](./DOCKER_DEPLOYMENT.md)를 참조하세요.
