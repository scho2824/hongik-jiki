"""
대화 기록 관리를 위한 메모리 구현
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import datetime
from dataclasses import dataclass, field

from hongikjiki.langchain_integration.base import MemoryBase

logger = logging.getLogger("HongikJikiChatBot")

@dataclass
class Message:
    """대화 메시지를 나타내는 데이터 클래스"""
    role: str  # 'user' 또는 'ai'
    content: str
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)


class ConversationMemory(MemoryBase):
    """대화 기록을 관리하는 기본 메모리 구현"""
    
    def __init__(self, max_history: int = 10):
        """
        대화 메모리 초기화
        
        Args:
            max_history: 저장할 최대 메시지 수
        """
        self.messages: List[Message] = []
        self.max_history = max_history
    
    def add_user_message(self, message: str) -> None:
        """
        사용자 메시지 추가
        
        Args:
            message: 사용자 메시지
        """
        self.messages.append(Message(role="user", content=message))
        self._trim_history()
    
    def add_ai_message(self, message: str) -> None:
        """
        AI 메시지 추가
        
        Args:
            message: AI 메시지
        """
        self.messages.append(Message(role="ai", content=message))
        self._trim_history()
    
    def get_chat_history(self, max_tokens: Optional[int] = None) -> str:
        """
        대화 기록 조회
        
        Args:
            max_tokens: 최대 토큰 수 (제한 필요시)
            
        Returns:
            str: 포맷팅된 대화 기록
        """
        formatted_history = ""
        
        # 간단하게 "Human: " 및 "AI: " 형식으로 포맷팅
        for msg in self.messages:
            prefix = "Human: " if msg.role == "user" else "AI: "
            formatted_history += f"{prefix}{msg.content}\n\n"
        
        # 토큰 제한이 있는 경우 처리 (간단한 구현)
        if max_tokens and len(formatted_history) > max_tokens * 4:  # 문자당 평균 토큰 수를 대략 0.25로 가정
            # 뒷부분에서 일정 길이만 가져오기
            formatted_history = formatted_history[-int(max_tokens * 4):]
            # 메시지 중간에서 잘리지 않도록 첫 번째 완전한 메시지부터 시작
            first_prefix_pos = formatted_history.find("Human: ")
            if first_prefix_pos > 0:
                formatted_history = formatted_history[first_prefix_pos:]
            else:
                first_prefix_pos = formatted_history.find("AI: ")
                if first_prefix_pos > 0:
                    formatted_history = formatted_history[first_prefix_pos:]
        
        return formatted_history.strip()
    
    def get_messages(self) -> List[Dict[str, str]]:
        """
        OpenAI 형식의 메시지 리스트 반환
        
        Returns:
            List[Dict[str, str]]: OpenAI 형식의 메시지 리스트
        """
        openai_messages = []
        
        for msg in self.messages:
            # OpenAI 메시지 형식으로 변환 (user/assistant)
            role = "user" if msg.role == "user" else "assistant"
            openai_messages.append({
                "role": role,
                "content": msg.content
            })
        
        return openai_messages
    
    def clear(self) -> None:
        """대화 기록 초기화"""
        self.messages = []
    
    def _trim_history(self) -> None:
        """대화 기록을 최대 길이로 제한"""
        if len(self.messages) > self.max_history:
            # 가장 오래된 메시지 제거
            self.messages = self.messages[-self.max_history:]


class ContextualMemory(ConversationMemory):
    """컨텍스트 정보를 포함하는 확장된 대화 메모리"""
    
    def __init__(self, max_history: int = 10):
        """
        컨텍스트 메모리 초기화
        
        Args:
            max_history: 저장할 최대 메시지 수
        """
        super().__init__(max_history=max_history)
        self.context: Dict[str, Any] = {}
    
    def add_context(self, key: str, value: Any) -> None:
        """
        컨텍스트 정보 추가
        
        Args:
            key: 컨텍스트 키
            value: 컨텍스트 값
        """
        self.context[key] = value
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """
        컨텍스트 정보 조회
        
        Args:
            key: 컨텍스트 키
            default: 기본값
            
        Returns:
            Any: 컨텍스트 값 또는 기본값
        """
        return self.context.get(key, default)
    
    def remove_context(self, key: str) -> None:
        """
        컨텍스트 정보 제거
        
        Args:
            key: 제거할 컨텍스트 키
        """
        if key in self.context:
            del self.context[key]
    
    def clear(self) -> None:
        """대화 기록 및 컨텍스트 초기화"""
        super().clear()
        self.context = {}
    
    def get_formatted_context(self) -> str:
        """
        포맷팅된 컨텍스트 정보 문자열 반환
        
        Returns:
            str: 포맷팅된 컨텍스트 정보
        """
        if not self.context:
            return ""
        
        context_str = "컨텍스트 정보:\n"
        for key, value in self.context.items():
            if isinstance(value, str):
                # 문자열 값은 그대로 추가
                context_str += f"- {key}: {value}\n"
            else:
                # 다른 타입은 간략히 표시
                context_str += f"- {key}: {str(value)[:100]}\n"
        
        return context_str
    
    def get_chat_history_with_context(self, max_tokens: Optional[int] = None) -> str:
        """
        컨텍스트 정보가 포함된 대화 기록 조회
        
        Args:
            max_tokens: 최대 토큰 수 (제한 필요시)
            
        Returns:
            str: 컨텍스트 정보와 대화 기록
        """
        context_str = self.get_formatted_context()
        history_str = self.get_chat_history(max_tokens)
        
        if context_str:
            return f"{context_str}\n\n{history_str}"
        else:
            return history_str