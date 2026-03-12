# MedGemma 도커 이미지 빌드 및 배포 가이드

이 가이드는 MedGemma 서버를 도커 이미지로 빌드하고 외부 서버에 배포하는 방법을 설명합니다.

---

## 목차

1. [사전 요구사항](#사전-요구사항)
2. [도커 이미지 빌드](#도커-이미지-빌드)
3. [이미지 저장 및 전송](#이미지-저장-및-전송)
4. [외부 서버 배포](#외부-서버-배포)
5. [서버 실행 테스트](#서버-실행-테스트)
6. [문제 해결](#문제-해결)

---

## 이미지 버전 관리 가이드 (Tip!)

도커 이미지를 빌드하실 때 `latest` 대신 지속적으로 버전 태그를 지정하여 버전을 관리하는 방법을 권장합니다:

1. **시맨틱 버저닝 (수동)**: 운영 단계에 명확히 구분된 버전 부여
   ```bash
   docker build -t medgemma-serving:v1.0.0 .
   docker build -t medgemma-serving:v1.0.1 .
   ```
2. **날짜/시간 기반 (자동)**: 빌드한 시간으로 구분 (개발 중 권장)
   ```bash
   docker build -t medgemma-serving:$(date +%Y%m%d) .
   # 결과: medgemma-serving:20240312
   ```
3. **Git 커밋 해시 (자동)**: 소스코드 버전과 정확히 일치시키는 방법
   ```bash
   docker build -t medgemma-serving:$(git rev-parse --short HEAD) .
   # 결과: medgemma-serving:a1b2c3d
   ```

*이 가이드 문서 내의 명령어들은 예시로 `v1.0.0` 태그를 기준으로 작성되어 있습니다. 필요에 따라 원하는 버전 태그명(예: 저장한 날짜)으로 치환해서 사용하시면 됩니다.*

---

## 사전 요구사항

### 빌드 서버 (이미지 생성)

- Docker 설치됨
- NVIDIA GPU Driver 설치됨
- 최소 20GB 디스크 공간

### 배포 서버 (실행 환경)

| 항목 | 요구사항 |
|------|----------|
| OS | Linux (Ubuntu 20.04/22.04 권장) |
| GPU | NVIDIA GPU (Compute Capability 7.0+) |
| GPU Driver | NVIDIA Driver 525+ |
| Docker | Docker 24.0+ |
| NVIDIA Container Toolkit | 설치 필수 |
| RAM | 최소 16GB |
| 디스크 | 최소 30GB 여유 공간 |
| VRAM | 최소 12GB (RTX 4090 권장) |

### NVIDIA Container Toolkit 설치 (배포 서버)

```bash
# NVIDIA 패키지 저장소 추가
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 패키지 업데이트 및 설치
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Docker 설정 재구성
sudo nvidia-ctk runtime configure --runtime=docker

# Docker 재시작
sudo systemctl restart docker

# GPU 통신 테스트
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

---

## 도커 이미지 빌드

### 1. 프로젝트 구조 확인

```
medgemma/
├── python/
│   ├── serving/
│   │   ├── Dockerfile          # 빌드용 Dockerfile
│   │   ├── models/
│   │   │   └── medgemma-4b-it/ # 모델 파일 (8.1GB)
│   │   └── ...
│   ├── .dockerignore           # 빌드 제외 파일
│   └── ...
└── ...
```

### 2. 이미지 빌드

```bash
# 프로젝트 루트 디렉토리로 이동
cd /path/to/medgemma

# 도커 이미지 빌드 (버전 태그 지정, 예: v1.0.0)
docker build -f python/serving/Dockerfile -t medgemma-serving:v1.0.0 .

# 빌드 완료 후 이미지 확인
docker images | grep medgemma
```

### 3. 빌드 옵션

| 옵션 | 설명 |
|------|------|
| `-f python/serving/Dockerfile` | Dockerfile 경로 지정 |
| `-t medgemma-serving:v1.0.0` | 이미지 이름 및 태그 (버전 정보 포함) |
| `--no-cache` | 캐시 없이 처음부터 빌드 (문제 발생 시) |
| `--progress=plain` | 빌드 과정 자세히 보기 |

---

## 이미지 저장 및 전송

### 방법 1: tar.gz 파일로 저장 (권장)

```bash
# 이미지를 tar.gz 파일로 압축 저장 (약 10-15분 소요)
docker save medgemma-serving:v1.0.0 | gzip > medgemma-serving_v1.0.0.tar.gz

# 파일 크기 확인
ls -lh medgemma-serving_v1.0.0.tar.gz
# 예상 크기: 약 10-12GB
```

### 방법 2: tar 파일로 저장 (압축 없음)

```bash
# 압축 없이 저장 (더 빠름, 파일 크기 큼)
docker save medgemma-serving:v1.0.0 -o medgemma-serving_v1.0.0.tar
```

### 외부 서버로 전송

#### SCP 사용 (로컬 → 원격)

```bash
# 로컬에서 원격 서버로 전송
scp medgemma-serving_v1.0.0.tar.gz user@remote-server:/path/to/destination/

# 또는 rsync 사용 (대용량 권장, 중단되어도 이어서 전송)
rsync -avz --progress medgemma-serving_v1.0.0.tar.gz user@remote-server:/path/to/destination/
```

#### S3/GS 사용 (클라우드)

```bash
# AWS S3
aws s3 cp medgemma-serving_v1.0.0.tar.gz s3://your-bucket/

# Google Cloud Storage
gsutil cp medgemma-serving_v1.0.0.tar.gz gs://your-bucket/
```

#### 물리적 전송

```bash
# USB 등에 복사하여 전달
cp medgemma-serving_v1.0.0.tar.gz /path/to/usb/
```

---

## 외부 서버 배포

### 1. 이미지 로드

```bash
# 압축된 이미지 로드
docker load < medgemma-serving_v1.0.0.tar.gz

# 또는 gunzip 후 로드
gunzip -c medgemma-serving_v1.0.0.tar.gz | docker load

# 로드 확인
docker images | grep medgemma
```

### 2. 컨테이너 실행

```bash
# 기본 실행
docker run -d \
  --name medgemma-server \
  --gpus all \
  -e AIP_HTTP_PORT=8080 \
  -p 8080:8080 \
  --shm-size=32g \
  medgemma-serving:v1.0.0 \
  --max-model-len=4096 \
  --gpu-memory-utilization=0.9 \
  --disable-log-stats

# 실행 확인
docker ps | grep medgemma
```

### 3. 실행 옵션

| 옵션 | 설명 |
|------|------|
| `-d` | 백그라운드 실행 |
| `--name medgemma-server` | 컨테이너 이름 지정 |
| `--gpus all` | 모든 GPU 사용 |
| `--gpus '"device=0"'` | 특정 GPU만 사용 (GPU ID 0) |
| `-e AIP_HTTP_PORT=8080` | (필수) 컨테이너 내부 웹 서버 포트 설정 |
| `-p 8080:8080` | 포트 포워딩 (호스트:컨테이너) |
| `--shm-size=32g` | 공유 메모리 크기 (모델 크기에 따라 조절, 다중 요청 시 여유 있게 설정) |
| `--restart always` | 시스템 재부팅 시 자동 시작 |
| `-v /host/path:/container/path` | 볼륨 마운트 |

### 4. 다중 GPU 서버에서 특정 GPU 지정

```bash
# GPU ID 2번만 사용
docker run -d \
  --name medgemma-server \
  --gpus '"device=2"' \
  -e AIP_HTTP_PORT=8080 \
  -p 8080:8080 \
  --shm-size=16g \
  medgemma-serving:v1.0.0

# GPU ID 0,1 사용
docker run -d \
  --name medgemma-server \
  --gpus '"device=0,1"' \
  -e AIP_HTTP_PORT=8080 \
  -p 8080:8080 \
  --shm-size=16g \
  medgemma-serving:v1.0.0
```

---

## 서버 실행 테스트

### 1. Health Check

```bash
# 서버 상태 확인
curl http://localhost:8080/health
# 예상 응답: ok

# 또는 브라우저에서
# http://<서버IP>:8080/health
```

### 2. 추론 테스트

```bash
# 텍스트만 요청
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ]
  }'

# 예상 응답:
# {
#   "choices": [{
#     "message": {
#       "content": "I am doing well, thank you...",
#       "role": "assistant"
#     }
#   }],
#   ...
# }
```

### 3. 로그 확인

```bash
# 컨테이너 로그 실시간 보기
docker logs -f medgemma-server

# 최근 100줄 보기
docker logs --tail 100 medgemma-server
```

### 4. 서버 관리 명령어

```bash
# 서버 중지
docker stop medgemma-server

# 서버 시작
docker start medgemma-server

# 서버 재시작
docker restart medgemma-server

# 서버 삭제
docker stop medgemma-server
docker rm medgemma-server

# 이미지 삭제
docker rmi medgemma-serving:v1.0.0
```

---

## 문제 해결

### 빌드 관련

#### "docker build command not found"
```bash
# Docker 설치 확인
docker --version

# Docker 설치 (Ubuntu)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

#### "no space left on device"
```bash
# Docker 정리
docker system prune -a --volumes

# 디스크 공간 확인
df -h
```

### 실행 관련

#### "could not select device driver"
```bash
# NVIDIA Container Toolkit 설치
# 위 [사전 요구사항] 섹션 참조

# Docker 재시작
sudo systemctl restart docker
```

#### "GPU memory insufficient"
```bash
# GPU 메모리 확인
nvidia-smi

# 다른 프로세스 중지 또는 다른 GPU 사용
docker run -d \
  --name medgemma-server \
  --gpus '"device=1"' \
  -e AIP_HTTP_PORT=8080 \
  -p 8080:8080 \
  medgemma-serving:v1.0.0
```

#### "Port 8080 already in use"
```bash
# 사용 중인 포트 확인
sudo lsof -i :8080

# 다른 포트 사용
docker run -d \
  --name medgemma-server \
  --gpus all \
  -e AIP_HTTP_PORT=8080 \
  -p 8081:8080 \
  medgemma-serving:v1.0.0
```

### 서버 응답 없음

```bash
# 컨테이너 상태 확인
docker ps -a | grep medgemma

# 컨테이너 로그 확인
docker logs medgemma-server

# 컨테이너 내부 진단
docker exec -it medgemma-server bash

# Triton 상태 확인
docker exec medgemma-server curl -s http://localhost:8600/v2/health/ready
```

---

## 참고 사항

### 포트 정보

| 포트 | 용도 |
|------|------|
| 8080 | HTTP API 서버 |
| 8600 | Triton gRPC/HTTP 서버 |

### 리소스 요구사항

| 리소스 | 최소 | 권장 |
|--------|------|------|
| 시스템 RAM | 16GB | 32GB |
| GPU VRAM | 12GB | 24GB (RTX 4090) |
| 디스크 | 30GB | 50GB |
| GPU | RTX 3090+ | RTX 4090 |

### 보안 권장사항

```bash
# 방화벽 설정 (필요한 경우)
sudo ufw allow 8080/tcp

# HTTPS 리버스 프록시 사용 (Nginx 예시)
# 외부 노출 시 Nginx 등을 통해 SSL/TLS 설정 권장
```

---

## 지원 및 문의

문제가 발생하면 다음을 확인하세요:
1. NVIDIA Driver 버전: `nvidia-smi`
2. Docker 버전: `docker --version`
3. 컨테이너 로그: `docker logs medgemma-server`
4. 시스템 리소스: `htop` 또는 `nvidia-smi`
