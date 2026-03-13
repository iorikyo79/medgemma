# MedGemma 프로젝트 수정 내역 보고서

> **기간**: 2026-03-08 ~ 2026-03-13 (6일간)
> **작업자**: iorikyo79 / aipacs

---

## 커밋 타임라인 요약

| 날짜 | 커밋 | 요약 | 주요 영역 |
|------|------|------|-----------|
| 03-08 | `07f92c1` | 한국어버전 노트북 추가 | `notebooks/` |
| 03-09 | `417adee` | MedGemma LoRA 파인튜닝 파이프라인 추가 | `challenge/` |
| 03-11 | `73fac46` | MedGemma 로컬 서빙 환경 구축 | `python/serving/` |
| 03-12 | `352bc21` | 도커 실행 시 속도 저하 문제 수정 | `python/serving/` |
| 03-13 | `89e50a6` | MedGemma 사용자 가이드 문서 추가 | 루트 |

---

## 1. 한국어 노트북 번역 (03-08, `07f92c1`)

**Author**: aipacs · **파일 12개 추가** (+8,556줄)

구글 공식 MedGemma 노트북을 한국어로 번역하여 `_ko` 접미사를 붙여 추가.

| 번역된 노트북 | 원본 |
|---------------|------|
| `cxr_anatomy_localization_with_hugging_face_ko.ipynb` | 흉부 X-ray 해부학 위치 특정 |
| `cxr_longitudinal_comparison_with_hugging_face_ko.ipynb` | 흉부 X-ray 종단 비교 |
| `ehr_navigator_agent_ko.ipynb` | EHR 네비게이터 에이전트 |
| `fine_tune_with_hugging_face_ko.ipynb` | HuggingFace 파인튜닝 |
| `high_dimensional_ct_hugging_face_ko.ipynb` | 고해상도 CT (HuggingFace) |
| `high_dimensional_ct_model_garden_ko.ipynb` | 고해상도 CT (Model Garden) |
| `high_dimensional_pathology_hugging_face_ko.ipynb` | 고해상도 병리 (HuggingFace) |
| `high_dimensional_pathology_model_garden_ko.ipynb` | 고해상도 병리 (Model Garden) |
| `quick_start_with_dicom_ko.ipynb` | DICOM 퀵스타트 |
| `quick_start_with_hugging_face_ko.ipynb` | HuggingFace 퀵스타트 |
| `quick_start_with_model_garden_ko.ipynb` | Model Garden 퀵스타트 |
| `reinforcement_learning_with_hugging_face_ko.ipynb` | 강화학습 |

---

## 2. LoRA 파인튜닝 파이프라인 (03-09, `417adee`)

**Author**: iorikyo79 · **파일 11개 추가** (+31,620줄)

방사선 보고서 **Findings → Impression 변환** 작업을 위한 LoRA 기반 학습·평가 파이프라인을 노트북 코드 기반으로 로컬 실행이 가능하도록 Python 스크립트로 변환.

### 핵심 파일

| 파일 | 역할 |
|------|------|
| [train.py](file:///home/jkuhm/src/MoltsBot-Source/medgemma-clone/medgemma/challenge/train.py) | LoRA 파인튜닝 학습 스크립트 (713줄) |
| [evaluation.py](file:///home/jkuhm/src/MoltsBot-Source/medgemma-clone/medgemma/challenge/evaluation.py) | 모델 평가 스크립트 (438줄) |
| [README.md](file:///home/jkuhm/src/MoltsBot-Source/medgemma-clone/medgemma/challenge/README.md) | 설치·사용법·하이퍼파라미터 문서 |
| [medgemma_lora_finetune_blog.md](file:///home/jkuhm/src/MoltsBot-Source/medgemma-clone/medgemma/challenge/medgemma_lora_finetune_blog.md) | 실험 결과 블로그 포스트 |
| [radiology-report-3.ipynb](file:///home/jkuhm/src/MoltsBot-Source/medgemma-clone/medgemma/challenge/radiology-report-3.ipynb) | 원본 Kaggle 노트북 |
| `radiology-report-3_ko.ipynb` | 한국어 번역 노트북 |
| `radiology-report-3_ko_local.ipynb` | 로컬 실행용 수정 노트북 |

### 주요 기능

- **데이터셋**: ReXGradient-160K 및 커스텀 CSV 지원
- **학습**: Gradient accumulation, Mixed precision (bfloat16), Early stopping
- **평가**: ROUGE-1/2/L, BERTScore, LoRA 어댑터 자동 감지·병합
- **추적**: MLFlow 실험 추적 통합
- **실험 결과**: 20개 샘플, 1 에포크, 3.8초 학습 → ROUGE-L **+9.72%** 향상

---

## 3. 로컬 서빙 환경 구축 (03-11, `73fac46`)

**Author**: iorikyo79 · **파일 14개 추가, 2개 수정** (+2,033줄)

구글 클라우드(Vertex AI) 전용 vLLM+Triton 서빙 코드를 **자체 서버에서 실행 가능하도록** 수정.

### 핵심 변경 내용

#### 코드 수정
| 파일 | 변경 내용 |
|------|-----------|
| [Dockerfile](file:///home/jkuhm/src/MoltsBot-Source/medgemma-clone/medgemma/python/serving/Dockerfile) | 로컬 모델 COPY 경로 추가, 라이선스 미러링 `ADD` 구문 주석 처리 (외부 Git 클론으로 인한 빌드 실패 방지) |
| [data_accessor_definition.py](file:///home/jkuhm/src/MoltsBot-Source/medgemma-clone/medgemma/python/data_accessors/gcs_generic/data_accessor_definition.py) | GCS 의존성 수정 |
| [abstract_handler.py](file:///home/jkuhm/src/MoltsBot-Source/medgemma-clone/medgemma/python/data_accessors/local_file_handlers/abstract_handler.py) | 로컬 핸들러 수정 |
| [.dockerignore](file:///home/jkuhm/src/MoltsBot-Source/medgemma-clone/medgemma/python/.dockerignore) | 빌드 컨텍스트 최적화 (54줄) |

#### 신규 문서
| 파일 | 내용 |
|------|------|
| [DOCKER_DEPLOYMENT.md](file:///home/jkuhm/src/MoltsBot-Source/medgemma-clone/medgemma/python/serving/DOCKER_DEPLOYMENT.md) | Docker 이미지 빌드·배포 전체 가이드 (405줄) |
| [LOCAL_SERVING_GUIDE.md](file:///home/jkuhm/src/MoltsBot-Source/medgemma-clone/medgemma/python/serving/LOCAL_SERVING_GUIDE.md) | Triton+Gunicorn 로컬 서빙 가이드 (217줄) |
| [SIMPLE_SERVING_README.md](file:///home/jkuhm/src/MoltsBot-Source/medgemma-clone/medgemma/python/serving/SIMPLE_SERVING_README.md) | vLLM 직접 서빙 가이드 (169줄) |
| [medgemma_docker_guide.md](file:///home/jkuhm/src/MoltsBot-Source/medgemma-clone/medgemma/python/serving/medgemma_docker_guide.md) | Docker 초보자용 가이드 (126줄) |

#### 신규 스크립트/도구
| 파일 | 역할 |
|------|------|
| [local_serving.sh](file:///home/jkuhm/src/MoltsBot-Source/medgemma-clone/medgemma/python/serving/local_serving.sh) | start/stop/status/test 통합 관리 스크립트 (271줄) |
| [simple_serving.py](file:///home/jkuhm/src/MoltsBot-Source/medgemma-clone/medgemma/python/serving/simple_serving.py) | vLLM 직접 서빙 서버 (232줄) |
| [test_curl.sh](file:///home/jkuhm/src/MoltsBot-Source/medgemma-clone/medgemma/python/serving/test_curl.sh) | cURL 기반 API 테스트 스크립트 (173줄) |
| [test_curl_with_image.sh](file:///home/jkuhm/src/MoltsBot-Source/medgemma-clone/medgemma/python/serving/test_curl_with_image.sh) | 이미지 포함 cURL 테스트 (118줄) |
| [test_serving.py](file:///home/jkuhm/src/MoltsBot-Source/medgemma-clone/medgemma/python/serving/test_serving.py) | Python 기반 API 테스트 (149줄) |

#### 기타
| 파일 | 내용 |
|------|------|
| [README_ko.md](file:///home/jkuhm/src/MoltsBot-Source/medgemma-clone/medgemma/README_ko.md) | 프로젝트 메인 README 한국어 번역 |
| `.gitignore` | 체크포인트, 로그, 임시 파일 등 로컬 개발 관련 항목 추가 |

---

## 4. 도커 속도 저하 문제 수정 (03-12, `352bc21`)

**Author**: iorikyo79 · **파일 3개 수정** (+84 / -27줄)

### 변경 요약

| 파일 | 변경 내용 |
|------|-----------|
| [entrypoint.sh](file:///home/jkuhm/src/MoltsBot-Source/medgemma-clone/medgemma/python/serving/entrypoint.sh) | Vertex AI 환경변수 기본값 설정 (`AIP_HTTP_PORT`, `AIP_HEALTH_ROUTE`, `AIP_PREDICT_ROUTE`), Cloud Logging 비활성화, 로컬 모델 자동 감지 경로 추가, HuggingFace 모델 ID 인자 전달 로직 추가 |
| [DOCKER_DEPLOYMENT.md](file:///home/jkuhm/src/MoltsBot-Source/medgemma-clone/medgemma/python/serving/DOCKER_DEPLOYMENT.md) | Docker 태그 `latest` → `v1.0.0`으로 변경, `-e AIP_HTTP_PORT=8080` 필수 옵션 추가, `--shm-size=32g`로 상향, `--max-model-len`, `--gpu-memory-utilization`, `--disable-log-stats` 옵션 추가 |
| [medgemma_docker_guide.md](file:///home/jkuhm/src/MoltsBot-Source/medgemma-clone/medgemma/python/serving/medgemma_docker_guide.md) | Docker 태그 변경에 맞춰 설명 업데이트 |

### 핵심 기술적 변경

```diff
# entrypoint.sh - 로컬 실행을 위한 Vertex AI 환경변수 기본값 설정
+ export AIP_HTTP_PORT="${AIP_HTTP_PORT:-8080}"
+ export AIP_HEALTH_ROUTE="${AIP_HEALTH_ROUTE:-/health}"
+ export AIP_PREDICT_ROUTE="${AIP_PREDICT_ROUTE:-/predict}"
+ export ENABLE_CLOUD_LOGGING="${ENABLE_CLOUD_LOGGING:-false}"

# Docker run 명령어 최적화
- --shm-size=16g
+ --shm-size=32g
+ --max-model-len=4096
+ --gpu-memory-utilization=0.9
+ --disable-log-stats
```

---

## 5. 사용자 가이드 문서 추가 (03-13, `89e50a6`)

**Author**: iorikyo79 · **파일 3개 추가** (+532줄)

서빙 서버를 활용하기 위한 사용자 대상 문서 두 종 작성.

| 파일 | 대상 | 내용 |
|------|------|------|
| [medgemma-guide-for-developer.md](file:///home/jkuhm/src/MoltsBot-Source/medgemma-clone/medgemma/medgemma-guide-for-developer.md) | 개발자/연구팀 | API 엔드포인트 설명, cURL/Python 테스트 방법, 유즈케이스별 프롬프트 (오탈자·모순 감지, 요약), 성능 평가 기준, 파인튜닝 의사결정 로직 |
| medgemma-guide-for-developer.docx | 개발자/연구팀 | 위 문서의 Word 버전 |
| [medgemma-guide-for-agent.md](file:///home/jkuhm/src/MoltsBot-Source/medgemma-clone/medgemma/medgemma-guide-for-agent.md) | AI 에이전트 | 에이전트가 MedGemma API를 활용하여 연구 작업을 수행하기 위한 가이드 (336줄) |

---

## 문서 현황 및 정리 필요사항

현재 프로젝트에 **20개의 `.md` 파일**이 존재하며, 서빙 관련 문서만 5개가 분산되어 있습니다.

### 서빙 관련 문서 (내용 중복/분산 발생)

| 문서 | 라인수 | 대상 | 상태 |
|------|--------|------|------|
| `DOCKER_DEPLOYMENT.md` | 405 → 수정됨 | 도커 배포 전체 과정 | ✅ 최신 |
| `medgemma_docker_guide.md` | 126 → 수정됨 | 도커 초보자 | ✅ 최신 |
| `LOCAL_SERVING_GUIDE.md` | 217 | Triton+Gunicorn 로컬 | ⚠️ 초기 작성 후 미갱신 |
| `SIMPLE_SERVING_README.md` | 169 | vLLM 직접 서빙 | ⚠️ 경로가 하드코딩, `352bc21` 변경분 미반영 |
| `python/serving/README.md` | — | 구글 원본 | 🔷 원본 유지 |

> [!WARNING]
> **`SIMPLE_SERVING_README.md`**: 모델 경로가 `challenge/models/` 로 하드코딩되어 있어 도커 이미지 내 경로(`/serving/models/`)와 불일치합니다.
> **`LOCAL_SERVING_GUIDE.md`**: `352bc21` 커밋의 `entrypoint.sh` 변경(AIP_HTTP_PORT, Cloud Logging 비활성화 등)이 반영되지 않았습니다.

### 루트 레벨 문서

| 문서 | 상태 |
|------|------|
| `README.md` | 🔷 구글 원본 유지 |
| `README_ko.md` | ✅ 한국어 번역 추가됨 |
| `medgemma-guide-for-developer.md` | ✅ 최신 |
| `medgemma-guide-for-agent.md` | ✅ 최신 |

### Challenge 문서

| 문서 | 상태 |
|------|------|
| `challenge/README.md` | ✅ 최신 |
| `challenge/medgemma_lora_finetune_blog.md` | ✅ 최신 |

---

## 전체 작업 통계

| 항목 | 수치 |
|------|------|
| 총 커밋 수 | 5 |
| 신규 파일 | 39개 |
| 수정 파일 | 5개 |
| 총 추가 줄 수 | ~42,825줄 |
| 작업 영역 | 3개 (challenge, python/serving, 루트 문서) |
