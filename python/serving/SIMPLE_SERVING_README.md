# MedGemma Simple Serving Guide

간단하게 MedGemma 4B 모델을 로컬에서 서빙하기 위한 가이드입니다.

## 사전 요구사항

- NVIDIA GPU (최소 12GB VRAM, 권장 16GB)
- Python 3.12
- CUDA 12.x

## 1단계: 의존성 설치

```bash
cd /home/jkuhm/src/MoltsBot-Source/medgemma-clone/medgemma/python/serving

# pip 업그레이드
pip install --upgrade pip

# PyTorch (CUDA 12.1 기준)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# vLLM 및 기타 의존성
pip install vllm transformers opencv-python-headless pillow flask
```

## 2단계: 서버 시작

```bash
cd /home/jkuhm/src/MoltsBot-Source/medgemma-clone/medgemma/python/serving
python simple_serving.py
```

서버가 시작되면 다음 메시지가 표시됩니다:
```
============================================================
MedGemma 4B Server Ready!
============================================================
Server running on http://0.0.0.0:8080

Available endpoints:
  - GET  http://localhost:8080/health
  - POST http://localhost:8080/v1/chat/completions
  - POST http://localhost:8080/predict
============================================================
```

## 3단계: 테스트

**별도의 터미널에서** 테스트 스크립트를 실행하세요:

```bash
cd /home/jkuhm/src/MoltsBot-Source/medgemma-clone/medgemma/python/serving
python test_serving.py
```

## API 사용 예시

### 1. 헬스 체크

```bash
curl http://localhost:8080/health
```

### 2. 간단한 텍스트 생성 (/predict)

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is medical imaging?",
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

### 3. 채팅 완료 (/v1/chat/completions)

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is a chest X-ray?"}
    ],
    "max_tokens": 150,
    "temperature": 0.7
  }'
```

### 4. 이미지와 함께 채팅

```bash
# 이미지를 base64로 인코딩
IMAGE_BASE64=$(base64 -w 0 /path/to/image.jpg)

curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"messages\": [
      {
        \"role\": \"user\",
        \"content\": [
          {\"type\": \"text\", \"text\": \"Describe this image.\"},
          {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/jpeg;base64,$IMAGE_BASE64\"}}
        ]
      }
    ],
    \"max_tokens\": 300
  }"
```

## Python 클라이언트 예시

```python
import requests
import base64

# 이미지 로드 및 인코딩
with open("image.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

# 요청 전송
response = requests.post(
    "http://localhost:8080/v1/chat/completions",
    json={
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                    }
                ]
            }
        ],
        "max_tokens": 300,
        "temperature": 0.7
    }
)

print(response.json()["choices"][0]["message"]["content"])
```

## 문제 해결

### GPU 메모리 부족
`simple_serving.py`에서 `gpu_memory_utilization` 값을 줄이세요:
```python
gpu_memory_utilization=0.7,  # 0.85에서 0.7로 감소
```

### CUDA 버전 불일치
설치된 CUDA 버전을 확인하고 맞는 PyTorch를 설치하세요:
```bash
nvidia-smi  # CUDA 버전 확인
```

### 모델 로딩 오류
모델 경로가 올바른지 확인하세요. `simple_serving.py`는 스크립트 위치 기준 상대 경로 (`./models/medgemma-4b-it`)를 사용합니다:
```bash
ls python/serving/models/medgemma-4b-it
```

## 서버 중지

터미널에서 `Ctrl+C`를 누르세요.

## 관련 문서

| 문서 | 설명 |
|------|------|
| [DOCKER_DEPLOYMENT.md](./DOCKER_DEPLOYMENT.md) | Docker 이미지 빌드 및 배포 가이드 (프로덕션 권장) |
| [LOCAL_SERVING_GUIDE.md](./LOCAL_SERVING_GUIDE.md) | Triton + Gunicorn 스택 로컬 서빙 가이드 |
| [medgemma_docker_guide.md](./medgemma_docker_guide.md) | Docker 초보자를 위한 입문 가이드 |

> **참고**: 이 Simple Serving은 빠른 개발/테스트 용도입니다. 프로덕션 환경이나 동시 요청 처리가 필요한 경우 Docker 기반 서빙(vLLM + Triton)을 권장합니다.
