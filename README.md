# Hongik-Jiki (홍익지기)

천공 스승님의 정법 가르침을 바탕으로 한 AI 챗봇입니다.

## 소개

Hongik-Jiki는 정법의 통찰과 역설을 통해 자신과 세상의 본질을 깨닫고,
홍익인간의 삶을 실현하도록 돕는 AI 비서입니다.

정법은 통찰로 자신과 세상의 본질을 깨닫고, 역설로 우리의 상식을 뒤집어
홍익인간의 삶을 실현하는 데 목적이 있습니다.

이 챗봇의 역할은 삶을 살아가는데 부딪히는 문제나 자주독립을 정법강의의 
통찰과 역설을 통해 찾아나갈 수 있도록 돕는 것입니다.

# 설치 방법
요구 사항

Python 3.8 이상
pip

## 설치

저장소 복제:
bashgit clone https://github.com/username/hongik-jiki.git
cd hongik-jiki

의존성 설치:
bashpip install -r requirements.txt

개발 모드로 설치:
bashpip install -e .


# 사용 방법
명령줄 인터페이스
bash
# 명령줄에서 실행
hongikjiki-cli

웹 인터페이스
bash
# 웹 서버 실행
hongikjiki-web
그 후 웹 브라우저에서 http://localhost:5000으로 접속하세요.

프로젝트 구조
hongik-jiki/
├── hongikjiki/              # 메인 패키지
│   ├── __init__.py
│   ├── chatbot.py           # 챗봇 핵심 클래스
│   ├── text_processor.py    # 텍스트 전처리 및 청크 분할
│   ├── vector_store.py      # 벡터 DB(ChromaDB) 관리
│   ├── utils.py             # 공통 유틸리티 함수
│   ├── cli.py               # 명령줄 인터페이스
│   └── web.py               # 웹 인터페이스
├── data/                    # 정법 문서 저장 디렉토리
│   └── jungbub_teachings/
├── output/                  # 처리 결과 저장 디렉토리
├── tests/                   # 테스트 코드
├── .env.example             # 환경 변수 예시
├── pyproject.toml           # 프로젝트 메타데이터
├── requirements.txt         # 의존성 목록
├── setup.py                 # 설치 스크립트
└── README.md                # 이 파일

## 기여 방법

이슈 등록 또는 기능 요청
포크 후 변경 사항 개발
테스트 실행 및 코드 스타일 확인
풀 리퀘스트 제출

## 라이선스
[라이선스 정보]
## 감사의 말
이 프로젝트는 천공 스승님의 정법에 기반하고 있습니다.