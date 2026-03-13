# MedGemma 서버 활용 가이드 (AI 에이전트용)

이 문서는 AI 에이전트가 사내에 배포된 MedGemma 의료 AI 모델 서버에 접속하여 의료 판독문 분석, 오탈자 검출, 논리적 모순 감지 등의 연구 작업을 수행할 때 참조해야 할 기술 명세서입니다.

> **⚠️ 중요**: 이 서버는 팀 내 연구용으로 운영되며, 예기치 않게 서빙이 중단될 수 있습니다. 서버 접속 실패 시 사용자에게 "서버가 응답하지 않습니다. 담당자(엄정권)에게 문의해 주세요."라고 안내하세요.

---

## 1. 서버 접속 정보

| 항목 | 값 |
| :--- | :--- |
| **서버 IP** | `10.10.40.194` |
| **포트** | `8080` |
| **GPU** | NVIDIA A100 80GB |
| **모델** | MedGemma-4B-IT (medgemma-4b-it) |
| **프레임워크** | vLLM + Triton Inference Server |

### 사용 가능한 엔드포인트

| 엔드포인트 | URL | 용도 |
| :--- | :--- | :--- |
| **Predict** | `http://10.10.40.194:8080/predict` | 기본 예측 API |
| **Chat Completions** | `http://10.10.40.194:8080/v1/chat/completions` | OpenAI 호환 API |
| **Health Check** | `http://10.10.40.194:8080/health` | 서버 상태 확인 |

> 두 예측 엔드포인트(`/predict`, `/v1/chat/completions`)는 내부적으로 완전히 동일한 추론 엔진에 연결됩니다. 어느 것을 사용해도 동일한 결과를 반환합니다. OpenAI Python 라이브러리를 사용할 경우 `/v1/chat/completions`를 사용하세요.

---

## 2. API 요청/응답 명세

### 2.1. 요청 형식 (Request)

- **메서드**: `POST`
- **Header**: `Content-Type: application/json`

#### 요청 Body (JSON)

```json
{
  "messages": [
    {
      "role": "system",
      "content": "시스템 프롬프트 (모델의 역할/페르소나 지정, 선택사항)"
    },
    {
      "role": "user",
      "content": "사용자의 질문 또는 분석 요청 내용"
    }
  ],
  "max_tokens": 1024,
  "temperature": 0.1,
  "top_p": 0.95
}
```

#### 주요 파라미터

| 파라미터 | 타입 | 기본값 | 설명 |
| :--- | :--- | :--- | :--- |
| `messages` | array | (필수) | 대화 메시지 배열. `role`은 `"system"`, `"user"`, `"assistant"` 중 하나. |
| `max_tokens` | int | 500 | 생성할 최대 토큰 수. 긴 답변이 필요하면 1024~2048로 설정. |
| `temperature` | float | - | 0.0~1.0. **의료 분석 작업에서는 반드시 0.1~0.2로 낮게 설정할 것.** 높을수록 창의적이지만 부정확해짐. |
| `top_p` | float | - | 0.0~1.0. nucleus sampling 확률. 일반적으로 0.9~0.95 사용. |
| `frequency_penalty` | float | - | 반복 억제. 0.0~2.0. |
| `seed` | int | - | 재현성을 위한 시드값. 동일 입력에 동일 출력을 원할 때 사용. |
| `stop` | string | - | 생성을 멈출 문자열 지정. |

### 2.2. 응답 형식 (Response)

```json
{
  "choices": [
    {
      "index": 0,
      "message": {
        "content": "모델의 응답 텍스트",
        "role": "assistant"
      }
    }
  ],
  "created": 1773311565,
  "id": "uuid-string",
  "model": "placeholder",
  "object": "chat.completion",
  "usage": {
    "completion_tokens": 161,
    "prompt_tokens": 52,
    "total_tokens": 213
  }
}
```

#### 응답 데이터 추출 방법

- **모델 답변 텍스트**: `response["choices"][0]["message"]["content"]`
- **사용된 입력 토큰 수**: `response["usage"]["prompt_tokens"]`
- **생성된 출력 토큰 수**: `response["usage"]["completion_tokens"]`

---

## 3. Python 코드 예제

### 3.1. 기본 요청 함수

```python
import requests

MEDGEMMA_URL = "http://10.10.40.194:8080/predict"

def query_medgemma(prompt, system_prompt=None, max_tokens=1024, temperature=0.1):
    """MedGemma 서버에 질의하고 응답 텍스트를 반환합니다."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95
    }
    
    try:
        response = requests.post(
            MEDGEMMA_URL,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=120  # 서버 타임아웃은 120초
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.ConnectionError:
        return "[ERROR] MedGemma 서버에 연결할 수 없습니다. 서버 상태를 확인하세요."
    except requests.exceptions.Timeout:
        return "[ERROR] 요청 시간이 초과되었습니다. max_tokens를 줄이거나 입력을 간소화하세요."
    except Exception as e:
        return f"[ERROR] 요청 실패: {str(e)}"
```

### 3.2. 서버 상태 확인

```python
def check_medgemma_health():
    """MedGemma 서버가 정상 동작 중인지 확인합니다."""
    try:
        response = requests.get("http://10.10.40.194:8080/health", timeout=10)
        return response.status_code == 200
    except:
        return False
```

### 3.3. cURL 명령어 (터미널 실행)

```bash
curl -s -X POST http://10.10.40.194:8080/predict \
     -H "Content-Type: application/json" \
     -d '{
       "messages": [
         {"role": "user", "content": "분석할 내용"}
       ],
       "max_tokens": 1024,
       "temperature": 0.1
     }' | jq -r '.choices[0].message.content'
```

---

## 4. 의료 판독문 분석 태스크 가이드

MedGemma 서버의 핵심 연구 목적은 의료 판독문(Radiology Report)의 품질 검증입니다. 아래는 에이전트가 연구 작업 수행 시 활용할 수 있는 태스크별 프롬프트 설계 가이드입니다.

### 태스크 A: 오탈자 및 철자 오류 검출

의학 용어의 철자 오류, 타이핑 실수, 문법적 오류를 감지합니다.

**시스템 프롬프트**:
```
당신은 전문 영상의학과 전문의이자 꼼꼼한 의무기록 검수자입니다.
제공된 판독문에서 명백한 오탈자, 의학 용어의 철자 오류, 또는 문법적으로 어색한 부분을 찾아내고 올바른 단어를 제안하세요.

[출력 형식]
발견된 오류가 있으면:
1. 원본: xxx -> 수정: yyy (이유)
오류가 없으면: "오류 없음"
```

**사용자 프롬프트**: 검사할 판독문 텍스트를 그대로 전달합니다.

### 태스크 B: 논리적 모순 및 의학적 불일치 감지 (핵심 태스크)

환자 정보와 판독문 내용 사이의 논리적 모순을 찾습니다.

- 좌우(Laterality) 반전 오류
- 성별과 맞지 않는 장기/소견 언급
- 이미 절제된 장기를 정상이라고 기술
- 검사 종류와 소견의 불일치

**시스템 프롬프트**:
```
당신은 20년 차 베테랑 임상의입니다.
아래 제공되는 [환자 정보]와 [판독문]을 대조하여 논리적으로 모순되거나 해부학적으로 성립할 수 없는 오류를 찾아내세요.

검증 항목:
1. 좌우(Laterality) 일치 여부
2. 환자 성별과 해부학적 소견의 일치 여부
3. 과거 수술력과 현재 소견의 모순 여부
4. 검사 종류(Modality)와 소견의 적합성

[출력 형식]
- 모순 발견 시: 항목별로 [모순 유형], [원문 내용], [문제점], [권장 수정] 형태로 출력
- 모순 없을 시: "모순 없음"
```

**사용자 프롬프트 구성**:
```
[환자 정보]
성별: {성별}, 나이: {나이}세, 과거력: {수술/질환 이력}

[판독문]
{판독문 전문}
```

### 태스크 C: 판독문 요약 및 임프레션(Impression) 도출

Findings 섹션으로부터 핵심 결론을 요약합니다.

**시스템 프롬프트**:
```
아래 영상 검사의 소견(Findings)을 바탕으로 영상의학과 전문의가 작성하는 임프레션(Impression) 형식으로 핵심 결론을 2~3줄 이내로 요약하세요.
중요도가 높은 소견을 우선 기술하고, 추가 검사가 필요한 경우 권고 사항을 포함하세요.
```

### 태스크 D: 판독문 한영/영한 번역

판독문의 언어를 변환합니다. 의학 용어의 정확성이 핵심입니다.

**시스템 프롬프트**:
```
아래 판독문을 {목표 언어}로 번역하세요.
의학 전문 용어는 국제적으로 통용되는 표준 용어를 사용하고, 약어는 그대로 유지하세요.
```

---

## 5. 프롬프트 엔지니어링 권장사항

MedGemma로부터 최적의 결과를 얻기 위한 프롬프트 작성 원칙입니다.

### 꼭 지켜야 할 원칙

1. **Temperature는 반드시 낮게 (0.1~0.2)**: 의료 분석은 사실 기반이므로 창의성보다 정확성이 중요합니다.
2. **역할(Persona) 명시**: "당신은 영상의학과 전문의입니다"와 같은 전문가 역할 부여가 답변 품질을 크게 향상시킵니다.
3. **출력 형식 지정**: 구조화된 출력(JSON, 번호 목록, 표 등)을 명시적으로 요청하면 분석 결과를 파싱하기 쉬워집니다.
4. **판독문의 메타데이터 제공**: 환자 정보(성별, 나이, 과거력)를 함께 제공하면 논리 검증의 정확도가 높아집니다.
5. **한 번에 하나의 태스크만 요청**: 오탈자 검출과 논리 모순 감지를 동시에 요청하면 정확도가 떨어집니다. 태스크를 분리하세요.

### 피해야 할 패턴

1. **모호한 지시**: "이 판독문 확인해줘" → 무엇을 확인할지 명확하지 않음.
2. **과도한 Temperature**: 0.7 이상으로 설정하면 의학적으로 부정확한 내용을 생성(Hallucination)할 위험이 높아집니다.
3. **max_tokens 부족**: 답변이 잘리면 분석 결과가 불완전해집니다. 최소 1024 이상 권장.

---

## 6. 배치 처리 (다수의 판독문 일괄 분석)

여러 건의 판독문을 순차적으로 분석할 때의 코드 패턴입니다.

```python
import time

def batch_analyze(reports, system_prompt, delay=1.0):
    """여러 판독문을 순차적으로 MedGemma에 보내 분석합니다.
    
    Args:
        reports: 판독문 리스트 (각 항목은 dict: {"id": str, "text": str, "patient_info": str})
        system_prompt: 분석 태스크를 지시하는 시스템 프롬프트
        delay: 요청 간 대기 시간(초). 서버 부하 방지를 위해 최소 1초 권장.
    
    Returns:
        결과 리스트 (각 항목은 dict: {"id": str, "input": str, "output": str})
    """
    results = []
    for i, report in enumerate(reports):
        user_prompt = f"[환자 정보]\n{report['patient_info']}\n\n[판독문]\n{report['text']}"
        
        output = query_medgemma(
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=1024,
            temperature=0.1
        )
        
        results.append({
            "id": report["id"],
            "input": report["text"],
            "output": output
        })
        
        # 서버 부하 방지를 위한 딜레이
        if i < len(reports) - 1:
            time.sleep(delay)
    
    return results
```

---

## 7. 에러 처리 및 문제 해결

| 증상 | 원인 | 해결 방법 |
| :--- | :--- | :--- |
| `ConnectionError` / 연결 거부 | 서버가 꺼져 있거나 네트워크 문제 | `/health` 엔드포인트로 상태 확인. 실패 시 담당자(엄정권)에게 연락. |
| `Timeout` (120초 초과) | 입력이 너무 길거나 서버 과부하 | `max_tokens`를 줄이거나, 입력 텍스트를 간소화. |
| 응답이 중간에 잘림 | `max_tokens` 값 부족 | `max_tokens`를 1024 이상으로 올림. |
| 한글이 유니코드로 출력됨 | JSON 인코딩 이슈(cURL 사용 시에만 발생) | `\| jq -r '.choices[0].message.content'`를 명령어 뒤에 추가. |
| 의학적으로 부정확한 답변 | Temperature가 높거나 프롬프트가 모호함 | Temperature를 0.1로 낮추고, 역할/태스크를 구체적으로 지시. |
| 동일 질문에 다른 답변이 나옴 | Temperature > 0이면 정상 동작 | 재현성이 필요하면 `seed` 파라미터를 고정값으로 설정. |

---

## 8. 모델 특성 및 제한사항

- **모델명**: MedGemma-4B-IT (Instruction-Tuned)
- **개발사**: Google
- **파라미터 수**: 약 40억 (4B)
- **강점**: 의료 영상 및 임상 텍스트에 대한 사전 학습. 판독문 오탈자 감지, 기본적인 의학 용어 이해에 강함.
- **약점 (주의)**:
  - 4B 규모의 경량 모델이므로 고도의 임상 추론(complex clinical reasoning)에는 한계가 있을 수 있습니다.
  - 한국어 판독문에 대한 학습 데이터가 영어 대비 제한적일 수 있습니다.
  - 모델의 답변은 참고용이며, 최종 의학적 판단은 반드시 전문의가 수행해야 합니다.
  - 환자 개인정보(이름, 주민번호 등)가 포함된 데이터를 절대로 서버에 전송하지 마세요.
