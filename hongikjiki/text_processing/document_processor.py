import os
import logging
from typing import List, Dict, Any, Optional
import hashlib
import json
from datetime import datetime

from hongikjiki.text_processing.document_loader import DocumentLoader
from hongikjiki.text_processing.text_normalizer import TextNormalizer
from hongikjiki.text_processing.metadata_extractor import MetadataExtractor
from hongikjiki.text_processing.document_chunker import DocumentChunker

logger = logging.getLogger("HongikJikiChatBot")

class DocumentProcessor:
    """
    문서 처리 워크플로우를 관리하는 클래스
    다른 컴포넌트들을 조합하여 전체 문서 처리 과정을 조율함
    """
    
    def __init__(self, 
                 document_loader: Optional[DocumentLoader] = None,
                 text_normalizer: Optional[TextNormalizer] = None,
                 metadata_extractor: Optional[MetadataExtractor] = None,
                 document_chunker: Optional[DocumentChunker] = None,
                 chunk_size: int = 1000,
                 overlap: int = 200):
        """
        DocumentProcessor 초기화 및 의존성 주입
        
        Args:
            document_loader: 문서 로더 인스턴스 (없으면 생성)
            text_normalizer: 텍스트 정규화 인스턴스 (없으면 생성)
            metadata_extractor: 메타데이터 추출 인스턴스 (없으면 생성)
            document_chunker: 문서 청킹 인스턴스 (없으면 생성)
            chunk_size: 기본 청크 크기
            overlap: 기본 중복 영역 크기
        """
        # 의존성 주입 또는 생성
        self.document_loader = document_loader or DocumentLoader()
        self.text_normalizer = text_normalizer or TextNormalizer()
        self.metadata_extractor = metadata_extractor or MetadataExtractor()
        self.document_chunker = document_chunker or DocumentChunker(chunk_size, overlap)
        
        # 기본 설정값
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # 처리된 문서 해시 추적 (중복 감지용)
        self.processed_hashes = []
    
    def process_directory(self, directory: str) -> List[Dict[str, Any]]:
        """
        디렉토리 내 모든 문서 처리
        
        Args:
            directory: 처리할 문서가 있는 디렉토리 경로
            
        Returns:
            List[Dict]: 처리 및 청킹된 문서 리스트
        """
        all_chunks = []
        
        # 각 파일 처리
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            
            # 디렉토리 건너뛰기
            if os.path.isdir(file_path):
                continue
                
            # 파일 확장자 확인
            ext = os.path.splitext(filename)[1].lower()
            if ext in ['.txt', '.md', '.pdf', '.docx', '.rtf']:
                chunks = self.process_file(file_path)
                all_chunks.extend(chunks)
        
        return all_chunks
    
    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        단일 파일 처리
        
        Args:
            file_path: 처리할 파일 경로
            
        Returns:
            List[Dict]: 처리 및 청킹된 문서 리스트 (청킹 결과에 따라 여러 개일 수 있음)
        """
        try:
            doc_data = self.document_loader.load_document(file_path) or {}
        except Exception as e:
            logger.error(f"파일 처리 오류: {e}")
            return []
        if isinstance(doc_data, dict):
            docs_list = [doc_data]
        elif isinstance(doc_data, list):
            docs_list = doc_data
        else:
            docs_list = []
        all_new_chunks = []
        for doc in docs_list:
            content = doc["content"]
            metadata_in = doc["metadata"]
            normalized = self.text_normalizer.normalize(content)
            content_hash = hashlib.md5(normalized.encode('utf-8')).hexdigest()
            if content_hash in self.processed_hashes:
                continue
            meta = self.metadata_extractor.extract_metadata(
                normalized,
                file_path,
                metadata_in
            )
            # Ensure filename stays as the original basename, not full path
            meta['filename'] = os.path.basename(file_path)
            docs_to_chunk = [{"content": normalized, "metadata": meta}]
            chunks = self.document_chunker.split_documents(docs_to_chunk)
            for chunk in chunks:
                all_new_chunks.append(chunk)
            self.processed_hashes.append(content_hash)
        return all_new_chunks
    
    def is_duplicate(self, content: str) -> bool:
        """
        내용 중복 여부 확인
        
        Args:
            content: 확인할 내용
            
        Returns:
            bool: 중복이면 True, 아니면 False
        """
        normalized = self.text_normalizer.normalize(content)
        content_hash = hashlib.md5(normalized.encode('utf-8')).hexdigest()
        return content_hash in self.processed_hashes