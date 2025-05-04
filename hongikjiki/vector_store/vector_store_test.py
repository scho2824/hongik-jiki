# vector_store_test.py
import os
from dotenv import load_dotenv
from hongikjiki.vector_store.chroma_store import ChromaVectorStore
from hongikjiki.vector_store.embeddings import get_embeddings

# 환경 변수 로드
load_dotenv()

# 벡터 저장소 경로
persist_directory = "./data/vector_store"

# 임베딩 모델 초기화
embeddings = get_embeddings("openai", model="text-embedding-3-small")

# 벡터 저장소 초기화
vector_store = ChromaVectorStore(
    collection_name="hongikjiki_jungbub",
    persist_directory=persist_directory,
    embeddings=embeddings
)

# 저장된 문서 수 확인
print(f"저장된 문서 수: {vector_store.count()}")

# 간단한 검색 테스트
test_queries = [
    "홍익인간이란 무엇인가요?",
    "정법이란 무엇인가요?",
    "천공 스승님에 대해 알려주세요",
    "용서에 대해 알려주세요",
    "명상은 어떻게 하나요?"
]

for query in test_queries:
    print(f"\n검색 쿼리: '{query}'")
    results = vector_store.search(query, k=3)
    
    if results:
        print(f"검색 결과 {len(results)}개 찾음:")
        for i, result in enumerate(results):
            score = result.get("score", 0)
            content = result.get("content", "")[:100] + "..." if len(result.get("content", "")) > 100 else result.get("content", "")
            print(f"{i+1}. 점수: {score:.4f}, 내용: {content}")
    else:
        print("검색 결과 없음")