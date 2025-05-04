from hongikjiki.vector_store.chroma_store import ChromaVectorStore
from hongikjiki.vector_store.embeddings import get_embeddings

# 벡터 저장소 초기화
vector_store = ChromaVectorStore(
    collection_name="hongikjiki_jungbub",
    persist_directory="./data/vector_store",
    embeddings=get_embeddings("huggingface", model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
)

print("총 문서 수:", vector_store.count())

# 전체 문서 가져오기
docs = vector_store.collection.get(include=["documents", "metadatas"])

print("\n🔍 '홍익인간'이 포함된 문서:")
found = 0
for i, (doc, meta) in enumerate(zip(docs['documents'], docs['metadatas'])):
    if "홍익인간" in doc:
        print(f"\n--- 문서 {i+1} ---")
        print(doc.strip())
        print("📎 메타데이터:", meta)
        found += 1

if found == 0:
    print("\n❗ '홍익인간'이 포함된 문서를 찾지 못했습니다.")