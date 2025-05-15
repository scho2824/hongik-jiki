"""
interface/utils.py

CLI 및 사용자 인터페이스에서 사용하는 유틸리티 함수 모음
"""

from hongikjiki.utils.file_utils import CHATBOT_NAME, DEVELOPER_NAME

def print_welcome_message() -> None:
    """콘솔에 챗봇 환영 메시지를 출력합니다."""
    print("=" * 60)
    print(f"🤖 {CHATBOT_NAME} - 개발자: {DEVELOPER_NAME}")
    print("천공 스승님의 가르침에 기반한 인공지능 비서입니다.")
    print("질문을 입력하거나 종료하려면 'q', 'quit', 'exit', '종료'를 입력하세요.")
    print("=" * 60)
