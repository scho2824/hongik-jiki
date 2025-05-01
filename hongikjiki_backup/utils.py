import os
import logging
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# 챗봇 및 개발자 정보
CHATBOT_NAME = os.getenv('CHATBOT_NAME', 'Hongik-Jiki')
DEVELOPER_NAME = os.getenv('DEVELOPER_NAME', '조성우')

def setup_logging():
    """로깅 설정"""
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    log_file = os.getenv('LOG_FILE', './logs/hongik_jiki.log')
    
    # 로그 디렉토리 생성
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # 로깅 레벨 설정
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    
    # 로거 설정
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("HongikJiki")
    logger.info(f"로깅 시스템 초기화 완료 (챗봇: {CHATBOT_NAME}, 개발자: {DEVELOPER_NAME})")
    return logger

def print_welcome_message():
    """시작 메시지 출력"""
    print("=" * 60)
    print(f"{CHATBOT_NAME} - 개발자: {DEVELOPER_NAME}")
    print("천공 스승님의 가르침에 기반한 인공지능 비서")
    print("정법은 통찰로 자신과 세상의 본질을 깨닫고, 역설로 우리의 상식을 뒤집어")
    print("홍익인간의 삶을 실현하는 데 목적이 있습니다.")
    print("=" * 60)