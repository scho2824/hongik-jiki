from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class VectorStoreBase(ABC):
    """
    벡터 저장소 기본 인터페이스
    모든 벡터 저장소 구현체는 이 클래스를 상속해야 함
    """
    
    def __init__(self):
        """VectorStoreBase 초기화"""
        pass
    
    @abstractmethod
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        텍스트 리스트를 벡터 저장소에 추가
        
        Args:
            texts: 추가할 텍스트 리스트
            metadatas: 각 텍스트에 대한 메타데이터 리스트 (옵션)
            
        Returns:
            List[str]: 추가된 문서 ID 리스트
        """
        pass
    
    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        문서 객체 리스트를 벡터 저장소에 추가
        
        Args:
            documents: 문서 객체 리스트 (content 및 metadata 필드 포함)
            
        Returns:
            List[str]: 추가된 문서 ID 리스트
        """
        pass
    
    @abstractmethod
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        쿼리에 가장 관련성 높은 문서 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            
        Returns:
            List[Dict]: 관련 문서 리스트
        """
        pass
    
    @abstractmethod
    def count(self) -> int:
        """
        벡터 저장소의 문서 수 반환
        
        Returns:
            int: 저장된 문서 수
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """벡터 저장소 초기화"""
        pass