# 벡터 저장소 진단 도구 (Vector Store Diagnostics)

이 도구는 Hongik-Jiki 챗봇의 벡터 저장소 성능을 진단하고 모니터링하기 위한 유틸리티입니다. 벡터 검색 속도, 정확성, 중복 문서 감지, 태그 분석 등 다양한 성능 지표를 측정하여 벡터 저장소의 최적화를 돕습니다.

## 주요 기능

- **성능 벤치마크**: 쿼리 검색 시간, 결과 품질 측정
- **태그 분석**: 태그 분포 및 사용 현황 분석
- **중복 문서 감지**: 유사하거나 중복된 문서 식별
- **진단 보고서 생성**: 분석 결과를 JSON 파일로 저장

## 설치 방법

이 모듈은 Hongik-Jiki 프로젝트의 일부로, 프로젝트 클론 후 사용할 수 있습니다.

```bash
# 프로젝트 루트 디렉토리에서 설치
pip install -e .
```

## 사용 방법

### 명령줄에서 실행

```bash
# 프로젝트 루트 디렉토리에서 실행
python -m hongikjiki.modules.vector_store.diagnostics
```

### 코드에서 사용

```python
from hongikjiki.modules.vector_store.chroma_store import ChromaVectorStore
from hongikjiki.modules.vector_store.embeddings import get_embeddings
from hongikjiki.modules.vector_store.diagnostics import VectorStoreDiagnostics

# 벡터 저장소 초기화
vector_store = ChromaVectorStore(
    collection_name="hongikjiki_jungbub",
    persist_directory="./data/vector_store",
    embeddings=get_embeddings("openai", model="text-embedding-3-small")
)

# 진단 도구 초기화
diagnostics = VectorStoreDiagnostics(vector_store)

# 전체 진단 실행 및 보고서 생성
report = diagnostics.run_full_diagnostics("vector_store_report.json")

# 또는 개별 진단 실행
benchmark_results = diagnostics.run_benchmarks()
tag_analysis = diagnostics.analyze_tags()
duplicate_check = diagnostics.check_duplicates()
```

## 진단 보고서 형식

진단 보고서는 JSON 형식으로 저장되며, 다음과 같은 주요 섹션을 포함합니다:

1. **benchmarks**: 검색 쿼리 성능 벤치마크
   - 평균 검색 시간
   - 쿼리별 결과 수
   - 검색 시간 통계 (최소, 최대, 표준편차)

2. **tag_analysis**: 태그 분포 분석
   - 문서별 태그 비율
   - 가장 많이 사용된 태그
   - 태그 다양성 통계

3. **duplicate_analysis**: 중복 문서 분석
   - 중복 문서 수
   - 중복 source_id 수
   - 중복 문서 예시

## 예시 사용 시나리오

### 1. 벡터 저장소 성능 모니터링

정기적으로 벡터 저장소 성능을 모니터링하여 변경 사항의 영향을 추적합니다.

```bash
# 매일 성능 보고서 생성
python -m hongikjiki.modules.vector_store.diagnostics --output reports/vector_store_$(date +%Y%m%d).json
```

### 2. 중복 문서 정리

중복 문서를 감지하고 정리하여 검색 품질과 성능을 향상시킵니다.

```python
# 중복 문서 감지
duplicates = diagnostics.check_duplicates()

# 중복 문서가 있으면 처리
if duplicates["content_duplicates"] > 0:
    # 중복 문서 삭제 등의 처리 로직
    pass
```

### 3. 태그 최적화

태그 분석 결과를 바탕으로 태그 시스템을 최적화합니다.

```python
# 태그 분석
tag_analysis = diagnostics.analyze_tags()

# 사용 빈도가 낮은 태그 파악
low_usage_tags = [tag for tag, count in tag_analysis["tag_distribution"].items() if count < 5]
```

## 주의사항

- 대규모 벡터 저장소에서는 진단 과정이 오래 걸릴 수 있습니다.
- 진단 중에는 CPU 및 메모리 사용량이 증가할 수 있습니다.
- OpenAI API를 사용하는 경우 API 사용량 및 비용에 주의하세요.

## 라이선스

이 도구는 Hongik-Jiki 프로젝트의 일부로 제공됩니다.