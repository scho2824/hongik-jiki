# Core 모듈 README

## 개요

`core` 모듈은 홍익지기 챗봇의 핵심 기능을 담당하는 컴포넌트들을 포함합니다. 이 모듈은 챗봇의 중심 로직, 데이터 모델, 응답 포맷팅, 그리고 관련 질문 생성 기능을 제공합니다.

## 구조

```
core/
├── __init__.py      # 패키지 초기화 및 공개 인터페이스
├── chatbot.py       # 챗봇 핵심 로직
├── formatter.py     # 응답 포맷팅
├── models.py        # 데이터 모델 정의
└── related_questions.py  # 관련 질문 생성
```

## 주요 컴포넌트

### chatbot.py

`HongikJikiChatbot` 클래스를 정의하며, 사용자 질문 처리, 문서 검색, 답변 생성의 핵심 로직을 포함합니다. 이 클래스는 LLM, 벡터 저장소, 태그 추출기를 사용하여 정법 기반 질의응답을 처리합니다.

**주요 기능:**
- 사용자 질문 처리 및 답변 생성
- 질문에서 태그 추출
- 관련 문서 검색
- 검색 결과 기반 답변 생성

### formatter.py

챗봇 응답의 일관된 형식을 정의하고, 검색 결과와 생성된 답변을 사용자 친화적인 형태로 포맷팅합니다.

**주요 기능:**
- 검색 결과와 답변 통합
- 출처 정보 포맷팅
- 관련 인용문 추출 및 표시
- 태그 정보 포맷팅

### models.py

챗봇에서 사용되는 주요 데이터 구조를 정의합니다. 이 모델들은 애플리케이션 전체에서 일관된 데이터 형식을 보장합니다.

**주요 데이터 클래스:**
- `SearchResult`: 검색 결과 표현
- `ChatResponse`: 챗봇 응답 데이터 구조화

### related_questions.py

사용자 질문과 태그를 기반으로 관련된 후속 질문을 생성합니다. 이 기능은 사용자가 대화를 계속 이어나갈 수 있도록 도와줍니다.

**주요 기능:**
- 태그 기반 관련 질문 생성
- 질문 중복성 검사
- 관련 인사이트 제공

## 사용 방법

기본적인 사용 예시:

```python
from hongikjiki.core import HongikJikiChatbot
from hongikjiki.vector_store import ChromaVectorStore
from hongikjiki.modules.tagging import TagExtractor

# 필요한 컴포넌트 초기화
llm = get_llm("openai")  # LLM 가져오기
vector_store = ChromaVectorStore()  # 벡터 저장소 초기화
tag_extractor = TagExtractor()  # 태그 추출기 초기화

# 챗봇 인스턴스 생성
chatbot = HongikJikiChatbot(llm, vector_store, tag_extractor)

# 질문 처리
response = chatbot.answer_question("정법에서 말하는 자주독립이란 무엇인가요?")

# 응답 출력
print(response.answer)
```

## 의존성

- 외부에서 주입되는 LLM (언어 모델)
- 벡터 저장소 (문서 검색용)
- 태그 추출기 (선택 사항)

## 개발 참고사항

- `chatbot.py`의 `HongikJikiChatbot` 클래스는 필요한 모든 의존성을 생성자를 통해 주입받도록 설계되었습니다.
- 모든 예외는 적절히 처리되며, 오류 상황에서도 사용자에게 의미 있는 응답을 제공합니다.
- 향후 확장을 위해 모듈 간 결합도를 낮게 유지하고 있습니다.

---

이 문서는 `core` 모듈의 구조와 기능을 설명합니다. 각 컴포넌트의 더 자세한 사용법은 해당 파일의 문서와 주석을 참조하세요.