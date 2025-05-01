"""
홍익지기 챗봇 클래스 구현
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union
import traceback

from hongikjiki.text_processing import DocumentProcessor
from hongikjiki.vector_store import ChromaVectorStore, get_embeddings
from hongikjiki.langchain_integration import (
    get_llm,
    ConversationMemory,
    ContextualMemory,
    ConversationalQAChain
)

# 로거 설정
logger = logging.getLogger("HongikJikiChatBot")

class HongikJikiChatBot:
    """홍익지기 챗봇 클래스"""
    
    def __init__(self, 
                 persist_directory: str = "./data",
                 embedding_type: str = "huggingface",
                 llm_type: str = "openai",
                 collection_name: str = "hongikjiki_documents",
                 max_history: int = 10,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 **kwargs):
        """
        홍익지기 챗봇 초기화
        
        Args:
            persist_directory: 데이터 저장 디렉토리
            embedding_type: 임베딩 모델 타입
            llm_type: 언어 모델 타입
            collection_name: 벡터 저장소 컬렉션 이름
            max_history: 최대 대화 기록 수
            chunk_size: 문서 청크 크기
            chunk_overlap: 문서 청크 중복 영역 크기
            **kwargs: 추가 설정 매개변수
        """
        self.persist_directory = persist_directory
        self.embedding_type = embedding_type
        self.llm_type = llm_type
        self.collection_name = collection_name
        self.max_history = max_history
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 디렉토리 생성
        os.makedirs(persist_directory, exist_ok=True)
        vector_store_dir = os.path.join(persist_directory, "vector_store")
        os.makedirs(vector_store_dir, exist_ok=True)
        
        # 임베딩 모델 초기화
        embedding_kwargs = kwargs.get("embedding_kwargs", {})
        self.embeddings = get_embeddings(embedding_type, **embedding_kwargs)
        logger.info(f"임베딩 모델 초기화 완료: {embedding_type}")
        
        # 벡터 저장소 초기화
        self.vector_store = ChromaVectorStore(
            collection_name=collection_name,
            persist_directory=vector_store_dir,
            embeddings=self.embeddings
        )
        logger.info("벡터 저장소 초기화 완료")
        
        # 문서 처리기 초기화
        self.document_processor = DocumentProcessor(
            chunk_size=chunk_size,
            overlap=chunk_overlap
        )
        logger.info("문서 처리기 초기화 완료")
        
        # LLM 초기화
        llm_kwargs = kwargs.get("llm_kwargs", {})
        self.llm = get_llm(llm_type, **llm_kwargs)
        logger.info(f"언어 모델 초기화 완료: {llm_type}")
        
        # 메모리 초기화
        self.memory = ContextualMemory(max_history=max_history)
        logger.info("대화 메모리 초기화 완료")
        
        # 대화 체인 초기화
        self.chain = ConversationalQAChain(
            llm=self.llm,
            vector_store=self.vector_store,
            memory=self.memory,
            k=kwargs.get("search_results_count", 4)
        )
        logger.info("대화 체인 초기화 완료")
        
        logger.info("홍익지기 챗봇 초기화 완료")
    
    def load_documents(self, directory: str) -> int:
        """
        문서 로드 및 벡터 저장소에 추가
        
        Args:
            directory: 문서 디렉토리 경로
            
        Returns:
            int: 처리된 문서 수
        """
        try:
            logger.info(f"{directory} 디렉토리에서 문서 로드 중...")
            
            # 문서 처리
            documents = self.document_processor.process_directory(
                directory, 
                self.chunk_size, 
                self.chunk_overlap
            )
            
            if not documents:
                logger.warning("처리된 문서가 없습니다.")
                return 0
            
            # 벡터 저장소에 추가
            ids = self.vector_store.add_documents(documents)
            
            logger.info(f"총 {len(documents)}개 청크 벡터 저장소에 추가 완료")
            return len(documents)
            
        except Exception as e:
            logger.error(f"문서 로드 오류: {e}")
            logger.error(traceback.format_exc())
            return 0
    
    def load_document(self, file_path: str) -> int:
        """
        단일 문서 로드 및 벡터 저장소에 추가
        
        Args:
            file_path: 문서 파일 경로
            
        Returns:
            int: 처리된 청크 수
        """
        try:
            logger.info(f"{file_path} 파일 처리 중...")
            
            # 문서 처리
            chunks = self.document_processor.process_file(file_path)
            
            if not chunks:
                logger.warning("처리된 청크가 없습니다.")
                return 0
            
            # 벡터 저장소에 추가
            ids = self.vector_store.add_documents(chunks)
            
            logger.info(f"총 {len(chunks)}개 청크 벡터 저장소에 추가 완료")
            return len(chunks)
            
        except Exception as e:
            logger.error(f"문서 처리 오류: {e}")
            logger.error(traceback.format_exc())
            return 0
    
    def chat(self, message: str) -> str:
        """
        사용자 메시지에 대한 응답 생성
        
        Args:
            message: 사용자 메시지
            
        Returns:
            str: 챗봇 응답
        """
        try:
            logger.info(f"사용자 메시지 수신: {message[:50]}...")
            
            # 체인을 통해 응답 생성
            response = self.chain.run(message)
            
            logger.info(f"응답 생성 완료: {response[:50]}...")
            return response
            
        except Exception as e:
            logger.error(f"응답 생성 오류: {e}")
            logger.error(traceback.format_exc())
            return f"죄송합니다. 응답 생성 중 오류가 발생했습니다: {str(e)}"
    
    def chat_with_context(self, message: str, context: Optional[str] = None) -> str:
        """
        컨텍스트를 포함한 사용자 메시지에 대한 응답 생성
        
        Args:
            message: 사용자 메시지
            context: 추가 컨텍스트 (선택 사항)
            
        Returns:
            str: 챗봇 응답
        """
        try:
            input_data = {
                "question": message,
                "context": context
            }
            
            # 체인을 통해 응답 생성
            response = self.chain.run(input_data)
            
            return response
            
        except Exception as e:
            logger.error(f"컨텍스트 응답 생성 오류: {e}")
            logger.error(traceback.format_exc())
            return f"죄송합니다. 응답 생성 중 오류가 발생했습니다: {str(e)}"
    
    def add_context(self, key: str, value: Any) -> None:
        """
        대화 컨텍스트에 정보 추가
        
        Args:
            key: 컨텍스트 키
            value: 컨텍스트 값
        """
        if isinstance(self.memory, ContextualMemory):
            self.memory.add_context(key, value)
            logger.debug(f"컨텍스트 추가: {key}")
    
    def clear_history(self) -> None:
        """대화 기록 초기화"""
        self.memory.clear()
        logger.info("대화 기록 초기화 완료")
    
    def search_documents(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """
        문서 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            
        Returns:
            List[Dict]: 검색 결과
        """
        try:
            results = self.vector_store.search(query, k=k)
            return results
        except Exception as e:
            logger.error(f"문서 검색 오류: {e}")
            return []
    
    def get_document_count(self) -> int:
        """
        벡터 저장소의 문서 수 반환
        
        Returns:
            int: 문서 수
        """
        try:
            return self.vector_store.count()
        except Exception as e:
            logger.error(f"문서 개수 조회 오류: {e}")
            return 0
    
    def reset_vector_store(self) -> bool:
        """
        벡터 저장소 초기화
        
        Returns:
            bool: 성공 여부
        """
        try:
            self.vector_store.reset()
            logger.info("벡터 저장소 초기화 완료")
            return True
        except Exception as e:
            logger.error(f"벡터 저장소 초기화 오류: {e}")
            return False