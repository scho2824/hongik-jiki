# Text Processing Module

이 모듈은 정법 기반 문서의 전처리와 청크화 작업을 수행하는 핵심 컴포넌트입니다.  
원문 텍스트를 로드하고 정규화하며, 의미 단위로 분할하고 필요한 메타데이터를 추출합니다.

---

## 📁 주요 구성

| 파일명 | 역할 |
|--------|------|
| `document_loader.py` | `.txt`, `.json` 등의 원시 문서 파일을 로드하고 `content` 필드로 정리 |
| `text_normalizer.py` | 타임스탬프 제거, 공백 및 구문 정리 등 텍스트 정규화 수행 |
| `document_chunker.py` | 고정 길이 및 의미 단위 기반으로 텍스트 청크 분할 |
| `metadata_extractor.py` | 태그 추출 및 문서 수준의 메타데이터 자동 생성 |
| `document_processor.py` | 위 기능들을 통합하여 전체 전처리 파이프라인 실행 |
| `document_manager.py` | 문서 수집, 벡터 DB 색인, 태그 추출 및 QA 생성 자동화 파이프라인 |

---

## 🧪 테스트

해당 모듈은 `test_text_processing.py`를 통해 개별 단위 테스트가 가능하며,  
`run_processor.py` 스크립트를 통해 CLI 기반 전체 흐름 테스트도 지원합니다.

```bash
python run_processor.py \
  --input-dir data/raw_docs \
  --output-file data/chunked_docs.json \
  --chunk-size 1000 \
  --overlap 200
```

`document_manager.py`를 통해 전체 문서 수집 및 전처리 → 임베딩 → 태그 추출 → QA 생성을 일괄 처리할 수 있습니다.

```bash
python document_manager.py \
  --dir data/docs \
  --reindex
```

---

## 🛠 향후 개선 사항

- 의미 기반 문장 분리 정확도 향상
- 타임스탬프 제거 및 인코딩 정규화 고도화
- 테스트 케이스 세분화 및 문서별 커스터마이징 로직 지원
- 문서 재색인 여부 자동 판단 로직 보완