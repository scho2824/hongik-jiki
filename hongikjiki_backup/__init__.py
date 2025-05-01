"""
홍익지기 (Hongik-Jiki) 챗봇 패키지

이 패키지는 정법 가르침 기반 질의응답 챗봇을 구현합니다.
"""

import logging
from logging.handlers import RotatingFileHandler
import os

# 로깅 설정
logger = logging.getLogger("HongikJikiChatBot")
logger.setLevel(logging.INFO)

# 콘솔 핸들러 추가
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_format)
logger.addHandler(console_handler)

# 로그 디렉토리 확인 및 생성
log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)

# 파일 핸들러 추가
file_handler = RotatingFileHandler(
    os.path.join(log_dir, "hongik_jiki.log"),
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
file_handler.setLevel(logging.DEBUG)
file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_format)
logger.addHandler(file_handler)

# 버전 정보
__version__ = "0.1.0"