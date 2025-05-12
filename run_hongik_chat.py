#!/usr/bin/env python3
"""
홍익지기 챗봇 통합 실행 스크립트
다양한 모드와 오류 처리 기능 포함
"""
import os
import sys
import argparse
import subprocess
import traceback
from dotenv import load_dotenv

def setup_environment():
    """환경 변수 설정 및 검증"""
    # .env 파일 로드
    load_dotenv()
    
    # OpenAI API 키 확인
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("오류: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("다음 방법 중 하나로 API 키를 설정하세요:")
        print("1. export OPENAI_API_KEY=your_api_key")
        print("2. .env 파일에 OPENAI_API_KEY=your_api_key 추가")
        return False
    
    print(f"API 키 확인: {api_key[:5]}...{api_key[-5:]}")
    return True

def verify_dependencies():
    """필요한 패키지 확인"""
    required_packages = ["gradio", "openai", "python-dotenv"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"다음 패키지가 설치되어 있지 않습니다: {', '.join(missing_packages)}")
        install = input("지금 설치하시겠습니까? (y/n): ")
        if install.lower() == 'y':
            subprocess.run([sys.executable, "-m", "pip", "install"] + missing_packages)
            print("패키지 설치 완료")
            return True
        else:
            print("패키지 설치를 건너뜁니다.")
            return False
    
    return True

def run_script(script_path):
    """스크립트 실행"""
    if not os.path.exists(script_path):
        print(f"오류: 스크립트 파일을 찾을 수 없습니다: {script_path}")
        return False
    
    print(f"스크립트 실행: {script_path}")
    try:
        result = subprocess.run([sys.executable, script_path], env=os.environ.copy())
        if result.returncode != 0:
            print(f"스크립트 실행 중 오류 발생: 종료 코드 {result.returncode}")
            return False
        return True
    except Exception as e:
        print(f"스크립트 실행 중 오류 발생: {e}")
        traceback.print_exc()
        return False

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="홍익지기 챗봇 통합 실행 스크립트")
    parser.add_argument("--mode", choices=["full", "demo", "simple", "minimal"], default="full",
                       help="실행 모드 선택: full(기본값), demo, simple, minimal")
    args = parser.parse_args()
    
    # 환경 설정
    if not setup_environment():
        return 1
    
    # 의존성 확인
    if not verify_dependencies():
        print("경고: 일부 의존성이 누락되었습니다. 실행이 실패할 수 있습니다.")
    
    # 모드에 따른 스크립트 선택
    script_path = None
    if args.mode == "full":
        script_path = "gradio_app.py"
    elif args.mode == "demo":
        script_path = "gradio_app_demo.py"
    elif args.mode == "simple":
        script_path = "simple_hongik_chat.py"
    elif args.mode == "minimal":
        script_path = "gradio_app_fixed.py"
    
    if not script_path:
        print(f"오류: 알 수 없는 모드: {args.mode}")
        return 1
    
    # 스크립트 실행
    success = run_script(script_path)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())