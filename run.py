# run.py (프로젝트 루트에 생성)
#!/usr/bin/env python
"""
홍익지기 챗봇 시작 스크립트
환경 변수 설정 및 애플리케이션 실행
"""

import os
import sys
from dotenv import load_dotenv

def main():
    """메인 함수: 환경 설정 및 앱 실행"""
    
    # 현재 경로 확인
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    print(f"작업 디렉토리: {current_dir}")
    
    # 환경 변수 로드
    load_dotenv()
    
    # API 키 확인
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("오류: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다")
        print("다음 방법 중 하나로 API 키를 설정하세요:")
        print("1. export OPENAI_API_KEY=your_api_key")
        print("2. .env 파일에 OPENAI_API_KEY=your_api_key 추가")
        return False
    
    print(f"API 키 확인: {api_key[:5]}...{api_key[-5:]}")
    
    # PYTHONPATH에 현재 디렉토리 추가 (모듈 임포트를 위해)
    sys.path.insert(0, current_dir)
    
    # 애플리케이션 실행
    try:
        print("홍익지기 챗봇을 시작합니다...")
        
        # 모듈 임포트 및 실행
        from hongikjiki.app.main import main as app_main
        app_main()
        
        return True
    except KeyboardInterrupt:
        print("\n사용자에 의해 애플리케이션이 중단되었습니다")
        return True
    except Exception as e:
        print(f"애플리케이션 실행 중 오류 발생: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)