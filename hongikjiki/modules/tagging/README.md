

# Tagging Module

이 모듈은 정법 문서 및 질문에서 주제 태그를 자동으로 추출하고,  
태그 체계를 계층적으로 정의 및 관리합니다.

---

## 📁 주요 구성

| 파일명 | 역할 |
|--------|------|
| `tag_schema.py` | 태그의 구조, 계층, 키워드, 문구, 연관성 등을 정의하는 중심 온톨로지 클래스 |
| `tag_extractor.py` | `TagSchema`를 기반으로 텍스트에서 관련 태그를 추출하는 엔진 |
| `tag_keywords.json` | 키워드 기반 태그 정의 (현재는 `tag_schema`로 점차 통합 중) |
| `tag_patterns.json` | 과거의 태그 추출용 패턴 정의 (현재는 `TagSchema`로 마이그레이션 진행) |
| `converted_tag_schema.yaml` | 기존 패턴 정보를 변환하여 저장한 YAML 기반 스키마 파일 |
| `convert_pattern_to_schema.py` | `pattern.json`을 `TagSchema` 형식의 YAML로 변환하는 스크립트 |

---

## 🧠 핵심 기능

- 계층 기반 태그 구조 정의 및 직렬화 (`TagSchema`)
- 키워드/문장 기반 유사도 추출 (`TagExtractor`)
- 추후 관련 질문 생성, 문서 추천 등과 연계 가능

---

## 🔁 변환 예시

```bash
python convert_pattern_to_schema.py
```

실행 시 `data/config/tag_patterns.json` → `data/converted_tag_schema.yaml`로 변환됩니다.

---

## 🛠 향후 개선 사항

- 정규표현식 기반 패턴(`patterns`) 활용 기능 추가
- 태그 간 유사도 기반 자동 연결 기능 확장
- 사용자 정의 태그 관리 UI 또는 YAML 편집기 연결