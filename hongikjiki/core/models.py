from typing import List, Dict, Optional, Any
from dataclasses import dataclass

@dataclass
class SearchResult:
    """검색 결과를 나타내는 데이터 클래스"""
    content: str
    metadata: Dict[str, Any]
    score: float = 0.0

@dataclass
class ChatResponse:
    """챗봇 응답을 나타내는 데이터 클래스"""
    answer: str
    related_questions: List[Dict[str, str]]
    source_documents: List[SearchResult]
    tags: Optional[List[str]] = None