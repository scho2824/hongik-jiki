"""
벡터 저장소 검사 도구 - Hongik-Jiki 챗봇

이 스크립트는 Hongik-Jiki 챗봇의 벡터 저장소를 검사하고 문제를 진단합니다.
"""

import os
import sys
import logging
import json
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("VectorStoreInspector")

# 모듈 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # 홍익지기 모듈 임포트
    from hongikjiki.vector_store.chroma_store import ChromaVectorStore
    from hongikjiki.vector_store.embeddings import get_embeddings
    
    logger.info("Hongik-Jiki 모듈 임포트 성공")
except ImportError as e:
    logger.error(f"모듈 임포트 오류: {e}")
    sys.exit(1)

def inspect_vector_store():
    """벡터 저장소 검사 및 진단"""
    
    # 벡터 저장소 경로
    persist_directory = "./data/vector_store"
    collection_name = "hongikjiki_documents"
    embedding_model = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    try:
        # 임베딩 모델 로드
        logger.info("임베딩 모델 로드 중...")
        embeddings = get_embeddings("huggingface", model_name=embedding_model)
        
        # 벡터 저장소 연결
        logger.info("벡터 저장소 연결 중...")
        vector_store = ChromaVectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embeddings=embeddings
        )
        
        # 컬렉션 정보 조회
        doc_count = vector_store.count()
        logger.info(f"벡터 저장소 문서 수: {doc_count}")
        
        if doc_count == 0:
            logger.error("벡터 저장소에 문서가 없습니다. 문서를 로드해야 합니다.")
            return
        
        # 샘플 쿼리로 검색 테스트
        sample_queries = [
            "홍익인간이란 무엇인가요?",
            "정법이란 무엇인가요?",
            "천공 스승님은 누구인가요?",
            "자비란 무엇인가요?",
            "선과 악의 기준은 무엇인가요?"
        ]
        
        logger.info("=== 검색 테스트 시작 ===")
        
        for query in sample_queries:
            logger.info(f"\n📝 쿼리: {query}")
            
            # 벡터 검색 수행
            results = vector_store.search(query, k=3)
            
            if not results:
                logger.warning(f"'{query}'에 대한 검색 결과가 없습니다.")
                continue
            
            # 검색 결과 출력
            logger.info(f"총 {len(results)} 개의 결과를 찾았습니다:")
            
            for i, result in enumerate(results):
                content = result.get("content", "")[:100]  # 앞부분만 출력
                score = result.get("score", 0)
                metadata = result.get("metadata", {})
                
                logger.info(f"  결과 {i+1}:")
                logger.info(f"  - 유사도 점수: {score:.4f}")
                logger.info(f"  - 내용: {content}...")
                logger.info(f"  - 메타데이터: {metadata}")
        
        # 벡터 저장소 구조 확인
        logger.info("\n=== 벡터 저장소 구조 검사 ===")
        
        # ChromaDB 구조 탐색
        if hasattr(vector_store, 'collection') and hasattr(vector_store.collection, '_collection'):
            coll = vector_store.collection._collection
            logger.info(f"컬렉션 이름: {coll.name}")
            logger.info(f"컬렉션 메타데이터: {coll.metadata}")
            
            # 저장된 임베딩 차원 확인
            try:
                sample_query_embedding = embeddings.embed_query(sample_queries[0])
                logger.info(f"임베딩 차원: {len(sample_query_embedding)}")
            except Exception as e:
                logger.error(f"임베딩 차원 확인 실패: {e}")
        
        logger.info("===== 벡터 저장소 검사 완료 =====")
        
    except Exception as e:
        logger.error(f"벡터 저장소 검사 중 오류 발생: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    logger.info("Hongik-Jiki 벡터 저장소 검사 도구 시작")
    inspect_vector_store()