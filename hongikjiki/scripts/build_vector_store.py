import logging
import os
import json
import hashlib
from langchain.schema import Document
from hongikjiki.vector_store.chroma_store import ChromaVectorStore
from hongikjiki.vector_store.embeddings import get_embeddings

def hash_text(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def load_qa_pairs(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def convert_to_documents(qa_pairs):
    documents = []
    for pair in qa_pairs:
        question = pair["question"]
        answer = pair["answer"]
        tags = pair.get("tags", [])
        content = f"Q: {question}\nA: {answer}"
        base_string = question + ",".join(sorted(tags))
        source_id = "jungbub_qa_" + hash_text(base_string)
        documents.append(Document(
            page_content=content,
            metadata={
                "source": source_id,
                "tags": tags
            }
        ))
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

    # 🔍 기존 문서 source 목록 가져오기
    existing_data = vector_store.get_all_documents()
    existing_metadatas = existing_data.get("metadatas", [])
    existing_sources = set(meta.get("source", "") for meta in existing_metadatas)

    # 🆕 새로운 문서만 필터링
    new_docs = [doc for doc in docs if doc.metadata["source"] not in existing_sources]

    print(f"🔹 신규 추가할 문서 수: {len(new_docs)}개")

    if new_docs:
        print("🔹 문서 임베딩 및 저장 중...")
        vector_store.add_texts(
            [doc.page_content for doc in new_docs],
            metadatas=[doc.metadata for doc in new_docs]
        )
    else:
        print("✅ 새롭게 추가할 문서가 없습니다.")

    # print("🔹 문서 임베딩 및 저장 중...")
    # vector_store.add_texts([doc.page_content for doc in docs], metadatas=[doc.metadata for doc in docs])
    vector_store.persist()
    print("✅ 벡터 저장소 구축 완료!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_file", type=str, required=True, help="Path to QA JSON file")
    args = parser.parse_args()

    build_vector_store(args.qa_file)
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/vector_store.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)