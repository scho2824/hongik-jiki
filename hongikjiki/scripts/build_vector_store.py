import logging
import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from langchain.schema import Document
from hongikjiki.modules.vector_store.chroma_store import ChromaVectorStore
from hongikjiki.modules.vector_store.embeddings import get_embeddings

ROOT_DIR = Path(__file__).resolve().parents[3]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VectorBuilder")

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/vector_store.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def hash_text(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def load_qa_pairs(file_path):
    with file_path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def convert_to_documents(qa_pairs):
    documents = []
    for pair in qa_pairs:
        try:
            question = pair["question"]
            answer = pair["answer"]
            tags = pair.get("tags", [])
            content = f"Q: {question}\nA: {answer}"
            base_string = question + ",".join(sorted(tags))
            source_id = "jungbub_qa_" + hash_text(base_string)

            metadata = {
                "source": source_id,
                "tags": tags
            }

            # 추가 메타데이터가 있다면 병합
            if "metadata" in pair:
                metadata.update(pair["metadata"])

            documents.append(Document(
                page_content=content,
                metadata=metadata
            ))
        except KeyError as e:
            logger.warning(f"⚠️ QA 필드 누락으로 문서 생성 실패: {e}")
            continue
    return documents

def build_vector_store(
    qa_file: Path,
    persist_dir: Path = Path("./data/vector_store"),
    collection_name: str = "hongikjiki_jungbub",
    append_log: bool = False
):

    logger.info("🔹 QA 데이터 로드 중...")
    qa_pairs = load_qa_pairs(qa_file)
    logger.info(f"🔹 총 {len(qa_pairs)}개의 QA 항목")

    docs = convert_to_documents(qa_pairs)

    logger.info("🔹 벡터 저장소 초기화 중...")
    vector_store = ChromaVectorStore(
        collection_name=collection_name,
        persist_directory=str(persist_dir),
        embeddings=get_embeddings("openai", model="text-embedding-3-small")
    )

    # 🔍 기존 문서 source 목록 가져오기
    existing_data = vector_store.get_all_documents()
    existing_metadatas = existing_data.get("metadatas", [])
    existing_sources = set(meta.get("source", "") for meta in existing_metadatas)

    # 🆕 새로운 문서만 필터링
    new_docs = [doc for doc in docs if doc.metadata["source"] not in existing_sources]

    logger.info(f"🔹 신규 추가할 문서 수: {len(new_docs)}개")

    if new_docs:
        logger.info("🔹 문서 임베딩 및 저장 중...")
        vector_ids = vector_store.add_texts(
            [doc.page_content for doc in new_docs],
            metadatas=[doc.metadata for doc in new_docs]
        ) or []

        if not vector_ids:
            logger.warning("❗️ vector_ids가 비어 있습니다. 저장 실패 가능성 있음.")
    else:
        logger.info("✅ 새롭게 추가할 문서가 없습니다.")

    # print("🔹 문서 임베딩 및 저장 중...")
    # vector_store.add_texts([doc.page_content for doc in docs], metadatas=[doc.metadata for doc in docs])

    # --- Extended metadata logging ---
    processed_log_path = ROOT_DIR / "data/processed_files.json"
    processed_log_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing log if it exists
    if processed_log_path.exists() and append_log:
        with processed_log_path.open("r", encoding="utf-8") as f:
            processed_log = json.load(f)
    else:
        processed_log = {}

    timestamp = datetime.now().isoformat()

    if new_docs:
        for doc, vector_id in zip(new_docs, vector_ids):
            source_id = doc.metadata.get("source", "")
            hash_val = hash_text(doc.page_content)
            tags = doc.metadata.get("tags", [])
            processed_log[source_id] = {
                "hash": hash_val,
                "processed_time": timestamp,
                "chunks_count": 1,
                "tags": tags,
                "vector_ids": [vector_id],
                "source_text": doc.page_content[:100]  # Optional preview
            }

    # Save back to file
    with processed_log_path.open("w", encoding="utf-8") as f:
        json.dump(processed_log, f, ensure_ascii=False, indent=2)

    vector_store.persist()
    logger.info("✅ 벡터 저장소 구축 완료!")

def test_vector_query(prompt: str, k: int = 3):
    vector_store = ChromaVectorStore(
        collection_name="hongikjiki_jungbub",
        persist_directory="data/vector_store",
        embeddings=get_embeddings("openai", model="text-embedding-3-small")
    )
    assert vector_store.embeddings is not None
    embedding = vector_store.embeddings.embed_query(prompt)

    results = []
    if (method := getattr(vector_store, "similarity_search_by_vector", None)) and callable(method):
        results = method(embedding, k=k)
    elif hasattr(vector_store, "vectorstore") and vector_store.vectorstore is not None:
        results = vector_store.vectorstore.similarity_search_by_vector(embedding, k=k)
    elif hasattr(vector_store, "store") and vector_store.store is not None:
        results = vector_store.store.similarity_search_by_vector(embedding, k=k)
    elif callable(getattr(vector_store, "query", None)):
        results = vector_store.query(prompt, top_k=k)
    else:
        raise AttributeError("No available similarity search method found in ChromaVectorStore")

    if not isinstance(results, list):
        results = []

    for i, doc in enumerate(results, 1):
        logger.info(f"\n🔹 Top {i}\n{doc['content']}\n📎 Metadata: {doc['metadata']}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_file", type=str, help="Path to QA JSON file (required when building vector store)")
    parser.add_argument("--persist_dir", type=str, default="./data/vector_store", help="Path to save Chroma vector store")
    parser.add_argument("--collection_name", type=str, default="hongikjiki_jungbub", help="Name of the Chroma collection")
    parser.add_argument("--query", type=str, help="Run a similarity query")
    parser.add_argument("--top_k", type=int, default=3, help="Top K results to return")
    parser.add_argument("--append_log", action="store_true", help="기존 processed_files.json에 병합 저장")

    args = parser.parse_args()

    # validation: require qa_file unless running query mode
    if not args.query:
        if not args.qa_file:
            parser.error("--qa_file is required unless --query is provided")
        qa_path = Path(args.qa_file)
    persist_path = Path(args.persist_dir)

    if args.query:
        test_vector_query(args.query, args.top_k)
    else:
        build_vector_store(qa_path, persist_dir=persist_path, collection_name=args.collection_name, append_log=args.append_log)