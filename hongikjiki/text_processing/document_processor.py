import os
import logging
from typing import List, Dict, Any, Optional
import hashlib

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
    
    def process_directory(self, directory: str, chunk_size: int = None, overlap: int = None) -> List[Dict[str, Any]]:
        """
        디렉토리 내 모든 문서 처리
        
        Args:
            directory: 처리할 문서가 있는 디렉토리 경로
            chunk_size: 청크 크기 (기본값 사용 시 None)
            overlap: 중복 영역 크기 (기본값 사용 시 None)
            
        Returns:
            List[Dict]: 처리 및 청킹된 문서 리스트
        """
        # 기본값 설정
        chunk_size = chunk_size or self.chunk_size
        overlap = overlap or self.overlap
        
        logger.info(f"{directory} 폴더에서 정법 문서 처리 시작...")
        
        documents = []
        processed_files = 0
        skipped_files = 0
        
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            
            # 디렉토리 건너뛰기
            if os.path.isdir(file_path):
                logger.debug(f"디렉토리 건너뛰기: {filename}")
                continue
            
            try:
                # 1. 문서 로드
                doc_data = self.document_loader.load_document(file_path)
                if not doc_data:
                    logger.warning(f"문서 로드 실패: {filename}")
                    skipped_files += 1
                    continue
                
                # 2. 텍스트 정규화
                normalized_content = self.text_normalizer.normalize(doc_data["content"])
                
                # 3. 중복 확인
                content_hash = hashlib.md5(normalized_content.encode('utf-8')).hexdigest()
                if content_hash in self.processed_hashes:
                    logger.info(f"중복 문서 건너뛰기: {filename}")
                    skipped_files += 1
                    continue
                
                self.processed_hashes.append(content_hash)
                
                # 4. 메타데이터 추출
                metadata = self.metadata_extractor.extract_metadata(
                    normalized_content, 
                    filename, 
                    doc_data["metadata"]
                )
                
                # 해시 업데이트
                metadata["file_hash"] = content_hash
                
                # 처리된 문서 추가
                documents.append({
                    "content": normalized_content,
                    "metadata": metadata
                })
                
                processed_files += 1
                logger.debug(f"문서 처리 완료: {filename}")
                
            except Exception as e:
                logger.error(f"문서 처리 오류: {filename}, 오류: {e}")
                logger.exception(e)
                skipped_files += 1
        
        logger.info(f"총 {processed_files}개 문서 처리 완료, {skipped_files}개 문서 건너뜀")
        
        # 5. 문서 분할 (청킹)
        chunked_docs = self.document_chunker.split_documents(documents, chunk_size, overlap)
        logger.info(f"총 {len(chunked_docs)}개 청크 생성 완료")
        
        return chunked_docs
    
    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        단일 파일 처리
        
        Args:
            file_path: 처리할 파일 경로
            
        Returns:
            List[Dict]: 처리 및 청킹된 문서 리스트 (청킹 결과에 따라 여러 개일 수 있음)
        """
        filename = os.path.basename(file_path)
        logger.info(f"파일 처리 중: {filename}")
        
        try:
            # 1. 문서 로드
            doc_data = self.document_loader.load_document(file_path)
            if not doc_data:
                logger.warning(f"문서 로드 실패: {filename}")
                return []
            
            # 2. 텍스트 정규화
            normalized_content = self.text_normalizer.normalize(doc_data["content"])
            
            # 3. 메타데이터 추출
            metadata = self.metadata_extractor.extract_metadata(
                normalized_content, 
                filename, 
                doc_data["metadata"]
            )
            
            # 해시 계산
            content_hash = hashlib.md5(normalized_content.encode('utf-8')).hexdigest()
            metadata["file_hash"] = content_hash
            
            document = {
                "content": normalized_content,
                "metadata": metadata
            }
            
            # 4. 문서 분할 (청킹)
            # 강제로 작은 청크 크기(500) 사용하여 토큰 한도를 넘지 않도록 함
            # 특히 RTF, PDF 파일의 경우 더 작은 청크 사용
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext in ['.rtf', '.pdf', '.docx']:
                forced_chunk_size = 400
                forced_overlap = 100
            else:
                forced_chunk_size = 800
                forced_overlap = 200
                
            chunked_docs = self.document_chunker.split_documents(
                [document], 
                forced_chunk_size,
                forced_overlap
            )
            
            logger.info(f"파일 '{filename}' 처리 완료: {len(chunked_docs)}개 청크 생성")
            return chunked_docs
            
        except Exception as e:
            logger.error(f"파일 처리 오류: {filename}, 오류: {e}")
            logger.exception(e)
            return []
    
    def is_duplicate(self, content: str) -> bool:
        """
        내용 중복 여부 확인
        
        Args:
            content: 확인할 내용
            
        Returns:
            bool: 중복이면 True, 아니면 False
        """
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        return content_hash in self.processed_hashes