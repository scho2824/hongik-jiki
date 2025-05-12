#!/usr/bin/env python3
"""
홍익지기 챗봇 시작 스크립트
"""
import os
import sys
import traceback
from hongikjiki.utils import load_dotenv, setup_logging

def main():
    # 로깅 설정
    logger = setup_logging()
    logger.info("홍익지기 챗봇 시작")
    
    # 환경 변수 로드
    load_dotenv()
    logger.info("환경 변수 로드 완료")
    
    # API 키 확인
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("오류: OPENAI_API_KEY가 설정되지 않았습니다.")
        print("다음 방법 중 하나로 API 키를 설정하세요:")
        print("1. export OPENAI_API_KEY=your_api_key")
        print("2. .env 파일에 OPENAI_API_KEY=your_api_key 추가")
        logger.error("API 키가 설정되지 않았습니다")
        return 1
    
    logger.info(f"API 키 확인: {api_key[:5]}...{api_key[-5:]}")
    print(f"API 키 확인: {api_key[:5]}...{api_key[-5:]}")
    
    try:
        print("Gradio 앱 시작 중...")
        from gradio_app import iface
        iface.launch(prevent_thread_lock=True)
        # 앱이 백그라운드에서 실행됨
        print("\n앱이 백그라운드에서 실행 중입니다.")
        print("앱을 종료하려면 Ctrl+C를 누르세요.")

        # 메인 스레드가 종료되지 않도록 대기
        import signal
        def signal_handler(sig, frame):
            print("\n앱을 종료합니다...")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.pause()  # 신호가 오기를 기다림

    except KeyboardInterrupt:
        print("\n사용자에 의해 앱이 종료되었습니다.")
        return 0
    except Exception as e:
        print(f"오류 발생: {e}")
        print("\n상세 오류:")
        traceback.print_exc()
        logger.error(f"앱 실행 오류: {e}\n{traceback.format_exc()}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())