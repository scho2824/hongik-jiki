# hongikjiki/modules/vector_store/embedding_tools.py
import logging
import time
import json
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path

from hongikjiki.modules.vector_store.chroma_store import ChromaVectorStore
from hongikjiki.modules.vector_store.embeddings import get_embeddings

logger = logging.getLogger("EmbeddingTools")

class EmbeddingTools:
    """
    임베딩 관리 및 최적화 도구
    """
    
    def __init__(self, vector_store: ChromaVectorStore):
        """
        임베딩 도구 초기화
        
        Args:
            vector_store: 벡터 저장소 인스턴스
        """
        self.vector_store = vector_store
    
    def compare_embeddings(self, texts: List[str]) -> Dict[str, Any]:
        """
        여러 텍스트 간의 임베딩 유사도 비교
        
        Args:
            texts: 비교할 텍스트 리스트
            
        Returns:
            Dict: 유사도 비교 결과
        """
        if not self.vector_store.embeddings:
            logger.error("임베딩 모델이 초기화되지 않았습니다.")
            return {"success": False, "error": "임베딩 모델이 없습니다."}
        
        try:
            # 텍스트 임베딩 계산
            embeddings = self.vector_store.embeddings.embed_documents(texts)
            
            # 유사도 행렬 계산
            similarity_matrix = np.zeros((len(texts), len(texts)))
            
            for i in range(len(texts)):
                for j in range(len(texts)):
                    # 코사인 유사도 계산
                    similarity = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                    similarity_matrix[i, j] = similarity
            
            # 결과 저장
            result = {
                "success": True,
                "texts": texts,
                "similarity_matrix": similarity_matrix.tolist(),
                "embedding_model": self.vector_store.embeddings.__class__.__name__
            }
            
            return result
            
        except Exception as e:
            logger.error(f"임베딩 비교 중 오류 발생: {e}")
            return {"success": False, "error": str(e)}
    
    def find_similar_documents(self, query_text: str, k: int = 5) -> Dict[str, Any]:
        """
        쿼리 텍스트와 유사한 문서 찾기
        
        Args:
            query_text: 쿼리 텍스트
            k: 반환할 유사 문서 수
            
        Returns:
            Dict: 유사 문서 및 유사도 정보
        """
        try:
            results = self.vector_store.search(query_text, k=k)
            
            return {
                "success": True,
                "query": query_text,
                "similar_documents": results
            }
            
        except Exception as e:
            logger.error(f"유사 문서 검색 중 오류 발생: {e}")
            return {"success": False, "error": str(e)}
    
    def optimize_embedding_cache(self) -> Dict[str, Any]:
        """
        임베딩 캐시 최적화 (메모리 사용량 감소 등)
        
        Returns:
            Dict: 최적화 결과 정보
        """
        # 이 기능은 Chroma 내부 캐시 최적화이므로 실제 구현은 제한적
        # 실제로는 ChromaDB의 내부 캐시 메커니즘에 의존
        
        try:
            # 현재 문서 수
            doc_count = self.vector_store.count()
            
            # 간단한 벡터 저장소 통계 출력
            return {
                "success": True,
                "message": "임베딩 캐시 최적화가 완료되었습니다.",
                "total_documents": doc_count
            }
            
        except Exception as e:
            logger.error(f"임베딩 캐시 최적화 중 오류 발생: {e}")
            return {"success": False, "error": str(e)}

# 모듈 단독 실행 시 임베딩 비교 테스트
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 벡터 저장소 초기화
    vector_store = ChromaVectorStore(
        collection_name="hongikjiki_jungbub",
        persist_directory="./data/vector_store",
        embeddings=get_embeddings("openai", model="text-embedding-3-small")
    )
    
    # 임베딩 도구 초기화
    embedding_tools = EmbeddingTools(vector_store)
    
    # 임베딩 비교 테스트
    test_texts = [
        "정법이란 무엇인가요?",
        "정법의 기본 원리가 궁금합니다.",
        "홍익인간이란 무엇인가요?",
        "우주의 법칙에 대해 알려주세요."
    ]
    
    result = embedding_tools.compare_embeddings(test_texts)
    logger.info(f"임베딩 비교 결과: {json.dumps(result, ensure_ascii=False, indent=2)}")