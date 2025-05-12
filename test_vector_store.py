# test_vector_store.py
from hongikjiki.vector_store.chroma_store import ChromaVectorStore
from hongikjiki.vector_store.embeddings import get_embeddings

vector_store = ChromaVectorStore(
    collection_name="hongikjiki_jungbub",
    persist_directory="./data/vector_store",
    embeddings=get_embeddings("openai", model="text-embedding-3-small")
)

print("Document Count:", vector_store.count())

# 쿼리 테스트
query = "정법이란 무엇인가요?"
results = vector_store.search(query, k=3)
for i, doc in enumerate(results):
    print(f"\n[{i+1}] {doc['content'][:200]}...\nMetadata: {doc['metadata']}")