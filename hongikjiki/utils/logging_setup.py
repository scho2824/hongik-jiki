# hongikjiki/utils/logging_setup.py
import os
import logging

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