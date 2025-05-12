# scripts/query_vector_store.py

from hongikjiki.vector_store.chroma_store import ChromaVectorStore
from hongikjiki.vector_store.embeddings import get_embeddings

def main():
    query = input("질문을 입력하세요: ")

    store = ChromaVectorStore(
        collection_name="hongikjiki_jungbub",
        persist_directory="data/vector_store",
        embeddings=get_embeddings("openai", model="text-embedding-3-small")
    )

    results = store.similarity_search(query, k=3)

    for i, doc in enumerate(results, 1):
        print(f"\n🔹 [Top {i}] 유사 문서:")
        print(doc.page_content)
        print(f"메타데이터: {doc.metadata}")

if __name__ == "__main__":
    main()