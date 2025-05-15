# hongikjiki/utils/file_utils.py
import os
import tempfile
import logging
from pathlib import Path
from typing import Optional

CHATBOT_NAME = os.getenv("CHATBOT_NAME", "Hongik-Jiki")
DEVELOPER_NAME = os.getenv("DEVELOPER_NAME", "개발연구원 조성우")

logger = logging.getLogger("HongikJikiChatBot")

def ensure_dir(directory: str) -> None:
    """디렉토리가 존재하지 않으면 생성"""
    os.makedirs(directory, exist_ok=True)

def setup_logging(log_level: str = "INFO") -> None:
    """기본 로깅 설정"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )

def create_temp_file(content, suffix=".txt", encoding="utf-8"):
    """임시 파일 생성 (다운로드용)"""
    try:
        temp_dir = tempfile.gettempdir()
        temp = tempfile.NamedTemporaryFile(
            delete=False, 
            mode="w", 
            suffix=suffix, 
            dir=temp_dir, 
            encoding=encoding
        )
        temp.write(content)
        temp.close()
        logger.info(f"임시 파일 생성: {temp.name}")
        return temp.name
    except Exception as e:
        logger.error(f"임시 파일 생성 오류: {e}")
        return None