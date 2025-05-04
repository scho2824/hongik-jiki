"""
LangChain 통합을 위한 기본 클래스 정의
"""

import logging
from typing import List, Dict, Any, Optional, Union, Callable
from abc import ABC, abstractmethod

logger = logging.getLogger("HongikJikiChatBot")

class LLMBase(ABC):
    """대규모 언어 모델(LLM) 통합을 위한 기본 인터페이스"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        단일 프롬프트로부터 텍스트 생성
        
        Args:
            prompt: 입력 프롬프트
            **kwargs: 추가 매개변수
            
        Returns:
            str: 생성된 텍스트
        """
        pass
    
    @abstractmethod
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """
        여러 프롬프트로부터 텍스트 일괄 생성
        
        Args:
            prompts: 입력 프롬프트 리스트
            **kwargs: 추가 매개변수
            
        Returns:
            List[str]: 생성된 텍스트 리스트
        """
        pass


class ChainBase(ABC):
    """LangChain Chain 통합을 위한 기본 인터페이스"""
    
    @abstractmethod
    def run(self, input_data: Any) -> Any:
        """
        체인 실행
        
        Args:
            input_data: 입력 데이터
            
        Returns:
            Any: 체인 실행 결과
        """
        pass


class MemoryBase(ABC):
    """대화 기록 관리를 위한 메모리 기본 인터페이스"""
    
    @abstractmethod
    def add_user_message(self, message: str) -> None:
        """
        사용자 메시지 추가
        
        Args:
            message: 사용자 메시지
        """
        pass
    
    @abstractmethod
    def add_ai_message(self, message: str) -> None:
        """
        AI 메시지 추가
        
        Args:
            message: AI 메시지
        """
        pass
    
    @abstractmethod
    def get_chat_history(self, max_tokens: Optional[int] = None) -> str:
        """
        대화 기록 조회
        
        Args:
            max_tokens: 최대 토큰 수 (제한 필요시)
            
        Returns:
            str: 포맷팅된 대화 기록
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """대화 기록 초기화"""
        pass