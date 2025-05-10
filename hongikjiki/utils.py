import os
import logging
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# 챗봇 및 개발자 정보
CHATBOT_NAME = os.getenv('CHATBOT_NAME', 'Hongik-Jiki')
DEVELOPER_NAME = os.getenv('DEVELOPER_NAME', '조성우')

def setup_logging():
    """로깅 설정 강화"""
    logger = logging.getLogger("HongikJikiChatBot")
    logger.setLevel(logging.DEBUG)

    # 이미 핸들러가 있으면 제거
    if logger.handlers:
        logger.handlers.clear()

    # 파일 핸들러
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, "chatbot.log"))
    file_handler.setLevel(logging.DEBUG)

    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # 포매터
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 핸들러 추가
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def print_welcome_message():
    """시작 메시지 출력"""
    print("=" * 60)
    print(f"{CHATBOT_NAME} - 개발자: {DEVELOPER_NAME}")
    print("천공 스승님의 가르침에 기반한 인공지능 비서")
    print("정법은 통찰로 자신과 세상의 본질을 깨닫고, 역설로 우리의 상식을 뒤집어")
    print("홍익인간의 삶을 실현하는 데 목적이 있습니다.")
    print("=" * 60)


# 문서 파일 검색 함수
from pathlib import Path
from typing import List, Union, Optional

def find_documents(data_dir: Union[str, Path], exts: Optional[List[str]] = None) -> List[Path]:
    """지정된 디렉토리에서 문서 파일(.txt 등) 목록을 재귀적으로 검색"""
    exts = exts or [".txt", ".md", ".rtf"]
    data_dir = Path(data_dir)
    return [p for p in data_dir.rglob("*") if p.suffix in exts and p.is_file()]

def ensure_dir(path: Union[str, Path]) -> None:
    """디렉토리가 존재하지 않으면 생성"""
    Path(path).mkdir(parents=True, exist_ok=True)