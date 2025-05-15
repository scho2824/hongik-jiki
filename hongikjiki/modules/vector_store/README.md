# Vector Store Module

이 모듈은 벡터 기반 문서 검색을 위한 저장소를 구현하며,  
ChromaDB를 백엔드로 사용하여 문서를 임베딩하고 유사도 기반 검색을 수행합니다.

---

## 📁 주요 파일

| 파일명 | 설명 |
|--------|------|
| `base.py`          | 모든 벡터 저장소 구현이 따르는 추상 베이스 클래스 (`VectorStoreBase`) 정의 |
| `chroma_store.py`  | ChromaDB를 이용한 벡터 저장소 구현 (기본 검색, 태그 기반 검색, 실험용 임베딩 생성기 포함) |
| `embeddings.py`    | OpenAI 또는 HuggingFace 기반 임베딩 모델 인터페이스 |
| `tag_index.py`     | 문서-태그 관계를 저장하는 로컬 인덱스 및 태그 기반 랭킹 기능 제공 |

---

## 🧠 주요 기능

- 텍스트 → 임베딩 변환 → 저장 및 검색
- 중복 문서 자동 필터링 (`source_id` 기준)
- 메타데이터 정리 및 복잡한 타입 처리
- 태그 기반 검색 지원 (연관 태그, 가중치 재정렬 등)
- 다양한 초기화 시도 및 예외 처리 강화
- `build_embedding_model()` 메서드를 통해 간단한 실험용 임베딩 모델 생성 가능

---

## 🔍 검색 API 예시

```python
vector_store = ChromaVectorStore(...)
results = vector_store.search("인공지능의 윤리 문제")
results = vector_store.search_with_tags("인공지능", tags=["윤리", "기술"], tag_boost=0.5)
results = vector_store.advanced_search("기술 발전의 방향은?")
```

---

## 🛠 향후 개선 사항

- 메타데이터 기반 필터링 기능 추가
- 벡터 삭제 후 Chroma 컬렉션 압축 기능 도입
- 태그 추출 성능 개선 및 자동화 연동 강화
- `TagSchema`를 통한 쿼리-태그 연동 최적화
- 다중 쿼리 비교를 위한 `batch_search()` 기능 확장

---

## 🧪 테스트 및 활용 예시

이 모듈은 다음과 같이 독립적으로 또는 통합 파이프라인에서 활용될 수 있습니다:

```python
from hongikjiki.modules.vector_store.chroma_store import ChromaVectorStore
from hongikjiki.modules.tagging.tag_schema import TagSchema

vector_store = ChromaVectorStore(...)
schema = TagSchema.load_from_file("data/converted_tag_schema.yaml")

query = "인간의 자유와 책임에 대해 알려줘"
cleaned_query, tags = vector_store.tag_aware_search.extract_tags_from_query(query, schema)
results = vector_store.search_with_tags(cleaned_query, tags)
```

```python
# 임베딩 모델 직접 사용 예시
from hongikjiki.modules.vector_store.chroma_store import ChromaVectorStore

embedding_model = ChromaVectorStore.build_embedding_model()
sentence_embedding = embedding_model.encode("정신의 본질은 무엇인가?")
```
