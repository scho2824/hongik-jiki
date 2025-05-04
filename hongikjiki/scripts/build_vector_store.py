

import json
import os
from langchain.schema import Document
from hongikjiki.vector_store.chroma_store import ChromaVectorStore
from hongikjiki.vector_store.embeddings import get_embeddings

def load_qa_pairs(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def convert_to_documents(qa_pairs):
    documents = []
    for pair in qa_pairs:
        content = f"Q: {pair['question']}\nA: {pair['answer']}"
        documents.append(Document(page_content=content, metadata={"source": "jungbub_qa"}))
    return documents

def build_vector_store(qa_file, persist_dir="./data/vector_store", collection_name="hongikjiki_jungbub"):
    print("🔹 QA 데이터 로드 중...")
    qa_pairs = load_qa_pairs(qa_file)
    print(f"🔹 총 {len(qa_pairs)}개의 QA 항목")

    docs = convert_to_documents(qa_pairs)

    print("🔹 벡터 저장소 초기화 중...")
    vector_store = ChromaVectorStore(
        collection_name=collection_name,
        persist_directory=persist_dir,
        embeddings=get_embeddings("openai", model="text-embedding-3-small")
    )

    print("🔹 문서 임베딩 및 저장 중...")
    vector_store.add_texts([doc.page_content for doc in docs], metadatas=[doc.metadata for doc in docs])
    vector_store.persist()
    print("✅ 벡터 저장소 구축 완료!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_file", type=str, required=True, help="Path to QA JSON file")
    args = parser.parse_args()

    build_vector_store(args.qa_file)