# MedGemma

MedGemma는 의료 텍스트 및 이미지 이해 성능을 위해 학습된 [Gemma 3](https://ai.google.dev/gemma/docs/core) 변형 모델들의 모음입니다. 개발자는 MedGemma를 사용하여 헬스케어 기반 AI 애플리케이션 구축을 가속화할 수 있습니다. MedGemma는 4B 멀티모달 버전과 27B 텍스트 전용 버전의 두 가지 변형으로 제공됩니다.

MedGemma 4B는 흉부 X-ray, 피부과 이미지, 안과 이미지, 조직 병리학 슬라이드 등 다양한 비식별 의료 데이터에 대해 특별히 사전 학습된 [SigLIP 이미지 인코더](https://arxiv.org/abs/2303.15343)를 활용합니다. LLM 컴포넌트는 방사선 이미지, 조직 병리학 패치, 안과 이미지, 피부과 이미지, 의료 텍스트를 포함한 다양한 의료 데이터 세트에 대해 학습됩니다.

MedGemma 변형 모델들은 기본 성능을 입증하기 위해 임상적으로 관련된 다양한 벤치마크에서 평가되었습니다. 여기에는 공개 벤치마크 데이터세트와 큐레이션된 데이터세트가 모두 포함되며, 특정 작업에 대한 전문가 인적 평가에 중점을 둡니다. 개발자는 성능 향상을 위해 MedGemma 변형 모델들을 파인튜닝할 수 있습니다. 자세한 내용은 곧 공개될 논문[링크 예정]과 의도된 사용 선언문(Intended Use Statement)을 참고해 주세요.

## 시작하기

*   개발자들을 위한 다음 단계의 전체 범위와 모델에 대한 자세한 내용을 알아보려면 [모델 카드](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card)를 포함한 [개발자 문서](https://developers.google.com/health-ai-developer-foundations/medgemma/get-started)를 읽어보세요.

*   모델을 사용하기 위한 [노트북](./notebooks)이 포함된 이 리포지토리를 탐색해 보세요.

*   [Hugging Face](https://huggingface.co/models?other=medgemma) 또는 [Model Garden](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/medgemma)에서 모델을 방문하세요.

## 기여하기

버그 리포트, 풀 리퀘스트(PR) 및 기타 기여에 열려 있습니다. 자세한 내용은 [CONTRIBUTING](CONTRIBUTING.md) 및 [커뮤니티 가이드라인](https://developers.google.com/health-ai-developer-foundations/community-guidelines)을 확인하세요.

## 라이선스

모델은 [Health AI Developer Foundations 라이선스](https://developers.google.com/health-ai-developer-foundations/terms)에 따라 라이선스가 부여되지만, 이 리포지토리의 모든 항목은 Apache 2.0 라이선스에 따라 라이선스가 부여됩니다. [LICENSE](LICENSE)를 참조하세요.
