import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger("HongikJikiChatBot")

class DocumentChunker:
    """
    문서를 적절한 크기의 청크로 분할하는 클래스
    검색 및 처리를 위한 청킹 로직 구현
    """
    
    def __init__(self, default_chunk_size: int = 1000, default_overlap: int = 200):
        """
        DocumentChunker 초기화
        
        Args:
            default_chunk_size: 기본 청크 크기 (문자 수)
            default_overlap: 기본 중복 영역 크기 (문자 수)
        """
        self.default_chunk_size = default_chunk_size
        self.default_overlap = default_overlap
    
    def split_documents(self, documents: List[Dict[str, Any]], 
                        chunk_size: int = None, 
                        overlap: int = None) -> List[Dict[str, Any]]:
        """
        문서를 청크로 분할하고 메타데이터 유지
        
        Args:
            documents: 문서 딕셔너리 리스트
            chunk_size: 각 청크의 최대 문자 수 (기본값 사용 시 None)
            overlap: 연속된 청크 간의 중복 문자 수 (기본값 사용 시 None)
                
        Returns:
            List[Dict]: 분할된 청크와 메타데이터
        """
        chunk_size = chunk_size or self.default_chunk_size
        overlap = overlap or self.default_overlap
        
        chunks = []
        
        for doc in documents:
            content = doc["content"]
            metadata = doc["metadata"]
            
            # 짧은 콘텐츠는 분할하지 않음
            if len(content) < chunk_size:
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = 0
                chunk_metadata["chunk_info"] = f"전체 문서 (1/1)"
                chunks.append({
                    "content": content,
                    "metadata": chunk_metadata
                })
                continue
            
            # 일반 문서는 문단 기준으로 분할
            doc_chunks = self._split_by_paragraph(content, metadata, chunk_size, overlap)
            chunks.extend(doc_chunks)
        
        # 청크 정보 업데이트
        for i, chunk in enumerate(chunks):
            chunks[i]["metadata"]["total_chunks"] = len(chunks)
        
        logger.info(f"{len(documents)}개 문서를 {len(chunks)}개 청크로 분할")
        return chunks
    
    def _split_by_paragraph(self, content: str, metadata: Dict[str, Any], 
                           chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        """
        문단 기반 분할
        
        Args:
            content: 분할할 문서 내용
            metadata: 유지할 메타데이터
            chunk_size: 각 청크의 최대 문자 수
            overlap: 연속된 청크 간의 중복 문자 수
            
        Returns:
            List[Dict]: 분할된 청크와 메타데이터
        """
        chunks = []
        
        # 문단으로 먼저 분할
        paragraphs = re.split(r'\n\s*\n', content)
        
        current_chunk = ""
        chunk_paragraphs = []  # 현재 청크에 포함된 문단 수 추적
        
        for para in paragraphs:
            para = para.strip()
            if not para:  # 빈 문단 건너뛰기
                continue
                
            # 현재 문단이 너무 긴 경우
            if len(para) > chunk_size:
                # 현재 청크가 있으면 추가
                if current_chunk:
                    chunk_metadata = metadata.copy()
                    chunk_metadata["chunk_index"] = len(chunks)
                    chunk_metadata["chunk_info"] = f"청크 {len(chunks) + 1} (문단 {', '.join(map(str, chunk_paragraphs))})"
                    chunks.append({
                        "content": current_chunk.strip(),
                        "metadata": chunk_metadata
                    })
                    current_chunk = ""
                    chunk_paragraphs = []
                
                # 긴 문단 문장 단위로 분할
                sentences = re.split(r'(?<=[.!?])\s+', para)
                temp_chunk = ""
                sentence_count = 0
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                        
                    sentence_count += 1
                    
                    if len(temp_chunk) + len(sentence) <= chunk_size:
                        temp_chunk += sentence + " "
                    else:
                        # 청크 추가
                        if temp_chunk:
                            chunk_metadata = metadata.copy()
                            chunk_metadata["chunk_index"] = len(chunks)
                            chunk_metadata["chunk_info"] = f"청크 {len(chunks) + 1} (긴 문단 분할: 문장 {sentence_count-1}개)"
                            chunks.append({
                                "content": temp_chunk.strip(),
                                "metadata": chunk_metadata
                            })
                        temp_chunk = sentence + " "
                
                # 남은 문장 처리
                if temp_chunk:
                    current_chunk = temp_chunk
                    chunk_paragraphs = [len(paragraphs)]  # 마지막 문단 표시
            else:
                # 현재 청크에 문단 추가 가능한지 확인
                if len(current_chunk) + len(para) <= chunk_size:
                    current_chunk += para + "\n\n"
                    chunk_paragraphs.append(paragraphs.index(para) + 1)  # 1-based 문단 번호
                else:
                    # 현재 청크 추가
                    chunk_metadata = metadata.copy()
                    chunk_metadata["chunk_index"] = len(chunks)
                    chunk_metadata["chunk_info"] = f"청크 {len(chunks) + 1} (문단 {', '.join(map(str, chunk_paragraphs))})"
                    chunks.append({
                        "content": current_chunk.strip(),
                        "metadata": chunk_metadata
                    })
                    
                    # 중복 영역 처리 - 마지막 문단을 다시 포함시키기 위한 로직
                    if overlap > 0 and chunk_paragraphs:
                        # 중복되는 마지막 문단(들)을 찾아 새 청크의
                        # 시작 부분에 포함
                        overlap_size = 0
                        overlap_paragraphs = []
                        
                        # 뒤에서부터 문단을 검사
                        for p_idx in reversed(chunk_paragraphs):
                            p_content = paragraphs[p_idx - 1]  # 0-based 인덱스 조정
                            if overlap_size + len(p_content) <= overlap:
                                overlap_paragraphs.insert(0, p_content)
                                overlap_size += len(p_content)
                            else:
                                break
                        
                        if overlap_paragraphs:
                            current_chunk = "\n\n".join(overlap_paragraphs) + "\n\n" + para + "\n\n"
                            chunk_paragraphs = [paragraphs.index(p) + 1 for p in overlap_paragraphs]
                            chunk_paragraphs.append(paragraphs.index(para) + 1)
                        else:
                            current_chunk = para + "\n\n"
                            chunk_paragraphs = [paragraphs.index(para) + 1]
                    else:
                        current_chunk = para + "\n\n"
                        chunk_paragraphs = [paragraphs.index(para) + 1]
        
        # 마지막 청크 추가
        if current_chunk:
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = len(chunks)
            chunk_metadata["chunk_info"] = f"청크 {len(chunks) + 1} (문단 {', '.join(map(str, chunk_paragraphs))})"
            chunks.append({
                "content": current_chunk.strip(),
                "metadata": chunk_metadata
            })
        
        # 중복 영역을 고려한 청크 정보 업데이트
        for i in range(len(chunks)):
            chunks[i]["metadata"]["total_chunks"] = len(chunks)
        
        return chunks