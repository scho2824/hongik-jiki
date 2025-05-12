# hongikjiki/utils/file_utils.py
import tempfile
import os
import logging

logger = logging.getLogger("HongikJikiChatBot")

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