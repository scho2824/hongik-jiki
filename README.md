# 홍익지기 (Hongik-Jiki) 챗봇

정법 교육 자료 기반의 대화형 AI 비서

## 프로젝트 소개

홍익지기는 천공 스승님의 정법 가르침에 기반한 대화형 AI 비서입니다. 정법 강의 자료를 벡터화하여 사용자의 질문에 대해 가장 관련성 높은 내용을 찾아 답변합니다. 이 프로젝트는 정법 교육을 더 많은 사람들에게 접근성 있게 제공하고, 개인의 삶의 문제에 대한 통찰력 있는 안내를 제공하는 것을 목표로 합니다.

## 주요 기능

- **텍스트 기반 대화**: 자연어로 정법에 관한 질문을 하면 관련 강의 내용을 기반으로 답변
- **태그 기반 검색**: 주제별 태그 시스템을 통해 더 정확한 검색 결과 제공
- **관련 강의 추천**: 각 답변과 관련된 정법 강의를 추천하여 더 깊은 학습 경로 제공
- **다양한 문서 형식 지원**: TXT, RTF, PDF, DOCX 등 다양한 형식의 정법 교육 자료 처리

## 시스템 구조

홍익지기는 다음과 같은 주요 모듈로 구성되어 있습니다:

1. **텍스트 처리 모듈**: 문서 로드, 정규화, 메타데이터 추출, 문서 분할
2. **벡터 저장소 모듈**: 텍스트 임베딩 및 의미 기반 검색
3. **태깅 모듈**: 주제별 분류 및 태그 기반 검색 강화
4. **챗봇 모듈**: 사용자 질문 처리 및 응답 생성
5. **파이프라인 모듈**: 문서 수집, 처리, 태깅 등의 전체 프로세스 자동화

## 설치 및 실행

### 필요 조건

- Python 3.8 이상
- 필수 패키지: 아래의 `requirements.txt` 참조

### 설치 방법

1. 저장소 클론
   ```bash
   git clone https://github.com/yourusername/hongikjiki.git
   cd hongikjiki
   ```

2. 가상 환경 생성 및 활성화
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. 의존성 설치
   ```bash
   pip install -r requirements.txt
   ```

4. 환경 변수 설정 (`.env` 파일 생성)
   ```
   OPENAI_API_KEY=your_api_key_here
   EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
   DATA_DIR=./data/jungbub_teachings
   CHATBOT_NAME=Hongik-Jiki
   DEVELOPER_NAME=your_name
   ```

### 실행 방법

#### 문서 처리 및 벡터화

```bash
# 문서 처리 및 벡터 데이터베이스 구축
python -m hongikjiki.pipeline.run_ingest
```

#### 태그 시스템 구축

```bash
# 문서 자동 태깅
python -m hongikjiki.pipeline.run_tagging
```

#### 챗봇 실행

```bash
# CLI 모드로 챗봇 실행
python -m hongikjiki.main

# 웹 서버 실행
python -m hongikjiki.app
```

## 태그 시스템

홍익지기는 주제별 태그 시스템을 통해 검색 및 응답 품질을 향상시킵니다. 태그는 다음과 같은 주요 카테고리로 구성됩니다:

1. **🌌 우주와 진리 (Universe & Truth)**
   - 정법, 우주법칙, 진리, 자연의 섭리 등

2. **🧍 인간 본성과 삶 (Human Nature & Life)**
   - 본성, 존재와 죽음, 자유와 선택 등

3. **🔍 탐구와 인식 (Inquiry & Awareness)**
   - 자기성찰, 의식 성장, 깨달음, 자각과 각성 등

4. **🧘 실천과 방법 (Practice & Method)**
   - 수행, 행공, 기도와 명상, 생활 실천 등

5. **🌏 사회와 현실 (Society & Reality)**
   - 관계, 공동체, 구조와 제도 등

6. **❤️ 감정 상태 (Emotional States)**
   - 불안, 분노, 슬픔, 평온, 기쁨 등

7. **🕰️ 삶의 단계 (Life Stages)**
   - 유년기, 청년기, 중년의 위기, 노년의 지혜 등

## 디렉토리 구조

```
hongikjiki/
│
├── chatbot.py                         # 메인 챗봇 클래스
│
├── text_processing/                   # 텍스트 전처리 및 구조화
│   ├── __init__.py
│   ├── document_loader.py             # 텍스트 로딩
│   ├── text_normalizer.py             # 정제 및 정규화
│   ├── metadata_extractor.py          # 메타데이터 추출
│   ├── document_chunker.py            # 문서 분할
│   ├── document_processor.py          # 처리 파이프라인 조정
│
├── vector_store/                      # 벡터 DB (Chroma 등)
│   ├── __init__.py
│   ├── base.py
│   ├── embeddings.py
│   ├── chroma_store.py
│   └── tag_index.py                   # 태그 기반 색인/검색
│
├── tagging/                           # 태그 시스템 (독립 도메인)
│   ├── __init__.py
│   ├── tag_schema.py                  # 계층적 태그 정의
│   ├── tag_extractor.py               # 텍스트 기반 태그 추출
│   ├── tag_analyzer.py                # 태그 통계 및 관계 분석
│   └── tagging_tools.py               # 수동 태깅 및 툴
│
├── pipeline/                          # 전체 파이프라인 정의 및 실행
│   ├── __init__.py
│   ├── run_ingest.py                  # 문서 수집 및 벡터화 전체 실행
│   ├── run_tagging.py                 # 자동 태깅 실행
│   ├── run_analysis.py                # 태그 관계 분석
│   └── run_demo_server.py             # Streamlit or FastAPI 데모 서버 실행
│
└── langchain_integration/             # LLM 연결 및 QA 체인
    ├── __init__.py
    ├── base.py
    ├── llm.py
    ├── memory.py
    └── chain.py

data/
├── config/                            # 설정 파일
│   ├── tag_schema.yaml                # 태그 계층 구조
│   └── tag_patterns.json              # 태그 추출 패턴
│
├── jungbub_teachings/                 # 정법 강의 텍스트 파일
│
├── vector_store/                      # 벡터 저장소 파일
│
└── tag_data/                          # 태그 관련 데이터
    ├── manually_tagged/               # 수동 태깅 문서
    ├── auto_tagged/                   # 자동 태깅 문서
    ├── input_chunks/                  # 태깅 입력 청크
    ├── tag_statistics.json            # 태그 통계
    └── tag_relationships.json         # 태그 관계 데이터
```

## 기여 방법

1. 이 저장소를 Fork합니다.
2. 새로운 기능 브랜치를 생성합니다 (`git checkout -b feature/amazing-feature`).
3. 변경사항을 커밋합니다 (`git commit -m 'Add some amazing feature'`).
4. 브랜치에 Push합니다 (`git push origin feature/amazing-feature`).
5. Pull Request를 생성합니다.

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다 - 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 연락처

프로젝트 관리자 - [@yourusername](https://github.com/yourusername)

프로젝트 링크: [https://github.com/yourusername/hongikjiki](https://github.com/yourusername/hongikjiki)