# QA Generation Module

정법 문서로부터 질문-답변(QA) 쌍을 생성하고, 통찰력 있는 설명과 정제된 표현으로 후처리하는 기능을 제공합니다.  
이 모듈은 사용자 학습용 QA 카드, GPT 훈련 데이터, 태그 기반 검색 QA 시스템 등 다양한 곳에 활용 가능합니다.

현재는 `converted_tag_schema.yaml` 파일을 기반으로 `TagSchema` 클래스가 태그 구조를 일관되게 관리하며, 각 태그의 설명, 키워드, 연관 질문 등도 함께 처리합니다.

---

## 📁 주요 파일

| 파일명 | 설명 |
|--------|------|
| `generate_qa.py` | 문서 청크에서 질문-답변 쌍을 생성하고 태그 및 인용문 기반 정보를 추가 |
| `generate_refined_qa.py` | 생성된 QA를 요약, 정제, 통찰 문장 추가 등의 방식으로 다듬는 후처리 스크립트 |
| `generate_insightful_qa.py` | OpenAI API를 활용하여 고차원 통찰력 요약, 키워드, 문맥 노트 등을 생성 |
| `qa_analysis.py` | QA 데이터셋에 대해 태그 적절성 평가 및 태그 후보를 분석하여 CSV로 저장 |

---

## 🧠 주요 기능

- 텍스트 청크 → 다중 질문-답변 생성
- 태그 기반 질문 생성 및 설명 연동
- 인용문, 통찰 요약, 키워드 자동 추출
- JSON 및 JSONL 포맷 저장 지원
- QA 데이터의 분석 및 후보 태그 평가
- YAML 기반 태그 스키마(`TagSchema`)를 통해 키워드 추출, 설명 생성, 태그 유효성 검사 수행

---

## 🧪 실행 예시

```bash
# 기본 QA 생성 (TagSchema를 사용한 태그 기반 질문 생성)
python generate_qa.py --input_dir data/processed_chunks --output_file data/generated_qa.json

# QA 후처리 및 정제 (태그 설명 및 인용문 기반 통찰 보강)
python generate_refined_qa.py --input data/generated_qa.json --output data/refined_qa.json

# 통찰 요약 추가 (OpenAI + 태그 기반 설명 부여)
python generate_insightful_qa.py --input data/refined_qa.json --output data/insightful_qa.json

# 태그 분석 및 추천 (TagSchema로 후보 태그 비교 및 평가)
python qa_analysis.py --input data/refined_qa.json --output data/qa_tag_analysis.csv
```

---

## 🛠 향후 개선 방향

- GPT 기반 질문 다변화 기능 추가
- 태그별 질문 템플릿 자동 추천
- 통찰 요약 자동 평가 지표 도입
- 주제별 QA 그룹핑 및 카드형 UI 내보내기 지원