#!/usr/bin/env python3
"""
Verify Vector Store

벡터 저장소 상태 확인 및 테스트 검색 수행
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from hongikjiki.modules.vector_store.chroma_store import ChromaVectorStore
from hongikjiki.modules.vector_store.embeddings import get_embeddings

def verify_vector_store():
    """벡터 저장소 상태 확인 및 테스트 검색 수행"""
    # 벡터 저장소 초기화
    embeddings = get_embeddings("openai", model="text-embedding-ada-002")
    vector_store = ChromaVectorStore(
        collection_name="hongikjiki_jungbub",
        persist_directory=str(ROOT_DIR / "data" / "vector_store"),
        embeddings=embeddings
    )
    
    # 문서 수 확인
    doc_count = vector_store.count()
    print(f"벡터 저장소 문서 수: {doc_count}")
    
    # 문서가 있는 경우 테스트 검색 수행
    if doc_count > 0:
        test_queries = [
            "정법이란 무엇인가요?",
            "자기성찰의 중요성",
            "마음을 다스리는 방법",
            "홍익인간의 의미"
        ]
        
        for query in test_queries:
            print(f"\n검색 쿼리: '{query}'")
            results = vector_store.search(query, k=2)
            
            if results:
                print(f"{len(results)}개 결과 찾음:")
                for i, result in enumerate(results):
                    print(f"\n결과 {i+1} (score: {result.get('score', 0):.4f})")
                    content = result.get('content', '')
                    print(f"내용 미리보기: {content[:150]}...")
            else:
                print("결과 없음")
    else:
        print("벡터 저장소가 비어 있습니다. run_ingest.py를 먼저 실행하세요.")

if __name__ == "__main__":
    verify_vector_store()