"""
hongikjiki.core.__init__.py

코어 패키지를 초기화하는 파일.
모듈 가져오기를 위한 Python 패키지 마커로 사용됩니다.
"""

# 선택적: 특정 클래스나 함수를 가져와서 __all__에 추가할 수 있습니다
from .chatbot import HongikJikiChatbot

# 이 패키지에서 가져올 수 있는 심볼 목록 정의
__all__ = ['HongikJikiChatbot']