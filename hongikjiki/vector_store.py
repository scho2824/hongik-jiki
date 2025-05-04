"""
벡터 데이터베이스 모듈
정법 교육 자료 벡터화 및 검색 기능 제공
"""
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorStore:
    """벡터 데이터베이스 관리 클래스"""
    
    def __init__(self, 
                 persist_directory: Union[str, Path] = "./data/chroma_db",
                 embedding_model_name: str = "jhgan/ko-sroberta-multitask"):
        """
        VectorStore 초기화
        
        Args:
            persist_directory: ChromaDB 영구 저장 경로
            embedding_model_name: 임베딩 모델 이름 (한국어 SRoBERTa 권장)
        """
        self.persist_directory = Path(persist_directory)
        self.embedding_model_name = embedding_model_name
        
        # 저장 디렉토리 생성
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # ChromaDB 클라이언트 초기화
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # 임베딩 모델 로드
        try:
            self.embedding_model = SentenceTransformer(embedding_model_name)
            logger.info(f"임베딩 모델 로드됨: {embedding_model_name}")
        except Exception as e:
            logger.error(f"임베딩 모델 로드 오류: {e}")
            raise
        
        # Wrapping embedding function for ChromaDB
        def _chroma_embedding_function(texts: List[str]) -> List[List[float]]:
            return [self.embedding_model.encode(text).tolist() for text in texts]
        embeddings = _chroma_embedding_function
        
        # 컬렉션 이름
        self.collection_name = "hongikjiki_jungbub"
        
        # 컬렉션 초기화 또는 로드
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=embeddings,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"컬렉션 로드됨: {self.collection_name}")
        except Exception as e:
            logger.error(f"컬렉션 생성 오류: {e}")
            raise
    
    def get_embedding(self, text: str) -> List[float]:
        """
        텍스트의 임베딩 벡터를 생성
        
        Args:
            text: 임베딩할 텍스트
            
        Returns:
            임베딩 벡터
        """
        return self.embedding_model.encode(text).tolist()
    
    def add_documents(self, document_chunks: List[Dict[str, Any]], batch_size: int = 100) -> None:
        """
        문서 청크를 벡터 데이터베이스에 추가
        
        Args:
            document_chunks: 문서 청크 리스트 (텍스트와 메타데이터 포함)
            batch_size: 배치 크기
        """
        total_chunks = len(document_chunks)
        logger.info(f"총 {total_chunks}개 청크를 벡터 데이터베이스에 추가합니다.")
        
        # 이미 존재하는 문서 ID 목록 가져오기
        existing_ids = set(self.collection.get()["ids"])
        
        # 배치로 나누어 처리
        for i in range(0, total_chunks, batch_size):
            batch = document_chunks[i:i+batch_size]
            
            ids = []
            texts = []
            metadatas = []
            embeddings = []
            
            for j, chunk in enumerate(batch):
                # 고유 ID 생성 (파일명_청크인덱스)
                chunk_id = f"{chunk['metadata']['file_name']}_{chunk['metadata']['chunk_index']}"
                
                # 이미 존재하는 ID면 건너뛰기
                if chunk_id in existing_ids:
                    continue
                
                # 데이터 준비
                text = chunk["text"]
                metadata = chunk["metadata"]
                
                # 임베딩 생성
                embedding = self.get_embedding(text)
                
                # 배치에 추가
                ids.append(chunk_id)
                texts.append(text)
                metadatas.append(metadata)
                embeddings.append(embedding)
            
            # 빈 배치가 아닌 경우에만 추가
            if ids:
                self.collection.add(
                    ids=ids,
                    documents=texts,
                    metadatas=metadatas,
                    embeddings=embeddings
                )
                logger.info(f"배치 추가 완료: {len(ids)} 청크")
        
        logger.info(f"벡터 데이터베이스 업데이트 완료. 현재 총 {self.collection.count()} 청크 저장됨.")
    
    def search(self, 
              query: str, 
              top_k: int = 5, 
              threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        쿼리와 유사한 문서 검색
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            threshold: 유사도 임계값 (이 값 이상만 반환)
            
        Returns:
            검색 결과 리스트 (텍스트, 메타데이터, 유사도 점수 포함)
        """
        # 쿼리 임베딩
        query_embedding = self.get_embedding(query)
        
        # 검색 실행
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # 결과 포맷팅
        formatted_results = []
        for i in range(len(results["ids"][0])):
            # 거리를 유사도 점수로 변환 (1 - 코사인 거리)
            # ChromaDB는 코사인 거리를 반환하므로 1에서 빼서 유사도로 변환
            similarity = 1 - results["distances"][0][i]
            
            # 임계값 이상인 결과만 포함
            if similarity >= threshold:
                formatted_results.append({
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "similarity": similarity
                })
        
        return formatted_results
    
    def get_total_documents(self) -> int:
        """
        저장된 총 문서 수 반환
        
        Returns:
            저장된 문서 수
        """
        return self.collection.count()
    
    def reset(self) -> None:
        """벡터 데이터베이스 컬렉션 초기화"""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"컬렉션 삭제됨: {self.collection_name}")
            
            # 컬렉션 재생성
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"컬렉션 재생성됨: {self.collection_name}")
        except Exception as e:
            logger.error(f"컬렉션 초기화 오류: {e}")
            raise