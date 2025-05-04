import re
import logging

logger = logging.getLogger("HongikJikiChatBot")

class TextNormalizer:
    """
    텍스트 정규화를 담당하는 클래스
    타임스탬프 제거, 특수문자 처리, 텍스트 정리 등을 수행
    """
    
    def __init__(self):
        """TextNormalizer 초기화"""
        pass
    
    def normalize(self, text: str) -> str:
        """
        텍스트 정규화 과정을 수행
        
        Args:
            text: 정규화할 원본 텍스트
            
        Returns:
            str: 정규화된 텍스트
        """
        if not text:
            return ""
        
        # 1. 타임스탬프 제거
        text = self.remove_timestamps(text)
        
        # 2. 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        # 3. 연속된 개행 정리
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # 4. 줄 앞뒤 공백 제거
        text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)
        
        return text.strip()
    
    def remove_timestamps(self, text: str) -> str:
        """
        내용에서 타임스탬프 제거
        
        Args:
            text: 타임스탬프가 포함된 텍스트
            
        Returns:
            str: 타임스탬프가 제거된 텍스트
        """
        if not text:
            return ""
        
        # 다양한 타임스탬프 패턴 제거
        # 1. 범위 타임스탬프 (예: 10.4 - 16.9:)
        text = re.sub(r'\d+\.\d+\s*-\s*\d+\.\d+[:：]\s*', '', text)
        
        # 2. 단일 타임스탬프 (예: 10:30:)
        text = re.sub(r'^\d+:\d+[:：]\s*', '', text, flags=re.MULTILINE)
        
        # 3. 영상 시간 표시 (예: [00:15])
        text = re.sub(r'\[\d+:\d+\]', '', text)
        
        # 4. 시간 패턴 (hh:mm:ss)
        text = re.sub(r'\b\d{1,2}:\d{2}(:\d{2})?\b', '', text)
        
        # 5. 강의 패턴 (예: 00:00, 01:04) - 줄 시작 타임스탬프
        text = re.sub(r'^\d{2}:\d{2}\s*', '', text, flags=re.MULTILINE)
        
        return text

    def chunk_text(self, text: str, max_length: int = 500) -> list[str]:
        """
        텍스트를 의미 단위로 나누는 간단한 청크 생성 함수
        기본적으로 문단 단위로 분할하며, 길이 제한을 고려해 추가 분할

        Args:
            text (str): 전체 텍스트
            max_length (int): 각 청크의 최대 길이 (기본값 500자)

        Returns:
            list[str]: 청크 리스트
        """
        if not text:
            return []

        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        for para in paragraphs:
            if len(para) <= max_length:
                chunks.append(para)
            else:
                # 너무 긴 문단은 문장 단위로 더 분할
                sentences = re.split(r'(?<=[.!?])\s+', para)
                chunk = ""
                for sentence in sentences:
                    if len(chunk) + len(sentence) <= max_length:
                        chunk += sentence + " "
                    else:
                        chunks.append(chunk.strip())
                        chunk = sentence + " "
                if chunk:
                    chunks.append(chunk.strip())
        return chunks