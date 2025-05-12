#!/usr/bin/env python
"""
홍익지기 챗봇 애플리케이션 실행 스크립트
환경 변수 설정 및 디버깅 정보를 제공합니다.
"""

import os
import sys
import logging
import subprocess
from dotenv import load_dotenv

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("홍익지기 런처")

def setup_environment():
    """환경 변수 설정 및 검증"""
    # .env 파일 로드
    load_dotenv()
    
    # OpenAI API 키 확인
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("오류: OpenAI API 키가 설정되지 않았습니다.")
        print("다음 방법 중 하나로 API 키를 설정하세요:")
        print("1. export OPENAI_API_KEY=your_api_key")
        print("2. .env 파일에 OPENAI_API_KEY=your_api_key 추가")
        return False
    
    logger.info(f"API 키 확인: {api_key[:5]}...{api_key[-5:]}")
    return True

def check_vector_store():
    """벡터 스토어 상태 확인"""
    vector_store_path = "data/vector_store"
    if not os.path.exists(vector_store_path):
        logger.warning(f"벡터 스토어 디렉토리가 존재하지 않습니다: {vector_store_path}")
        return False
    
    logger.info(f"벡터 스토어 디렉토리 확인: {vector_store_path}")
    return True

def run_application():
    """애플리케이션 실행"""
    try:
        # 환경 설정
        if not setup_environment():
            return False
        
        # 벡터 스토어 확인
        check_vector_store()
        
        # 경로 디버깅
        current_dir = os.getcwd()
        logger.info(f"현재 디렉토리: {current_dir}")
        
        # Gradio 앱 실행
        logger.info("홍익지기 챗봇 앱을 시작합니다...")
        app_path = os.path.join(current_dir, "gradio_app.py")
        
        env = os.environ.copy()
        result = subprocess.run([sys.executable, app_path], env=env)
        
        if result.returncode != 0:
            logger.error(f"애플리케이션 실행 실패: 반환 코드 {result.returncode}")
            return False
        
        return True
    except KeyboardInterrupt:
        logger.info("사용자에 의해 애플리케이션이 중단되었습니다.")
        return True
    except Exception as e:
        logger.error(f"애플리케이션 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    successful = run_application()
    sys.exit(0 if successful else 1)