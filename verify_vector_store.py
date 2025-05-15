# verify_vector_store.py
from pathlib import Path
from hongikjiki.modules.vector_store.chroma_store import ChromaVectorStore
from hongikjiki.modules.vector_store.embeddings import get_embeddings

# Define paths
ROOT_DIR = Path(__file__).resolve().parent
PERSIST_DIR = ROOT_DIR / "data" / "vector_store"

# Initialize vector store
embeddings = get_embeddings("openai", model="text-embedding-ada-002")
vector_store = ChromaVectorStore(
    collection_name="hongikjiki_jungbub",
    persist_directory=str(PERSIST_DIR),
    embeddings=embeddings
)

# Check count
doc_count = vector_store.count()
print(f"Vector store contains {doc_count} documents")

# Test search
if doc_count > 0:
    test_queries = [
        "정법이란 무엇인가요?",
        "자기성찰의 중요성",
        "마음을 다스리는 방법",
        "홍익인간의 의미"
    ]
    
    for query in test_queries:
        print(f"\nTesting query: '{query}'")
        results = vector_store.search(query, k=2)
        
        if results:
            print(f"Found {len(results)} results")
            for i, result in enumerate(results):
                print(f"\nResult {i+1} (score: {result.get('score', 0):.4f})")
                content = result.get('content', '')
                print(f"Content preview: {content[:150]}...")
        else:
            print("No results found")
else:
    print("Vector store is empty. Run load_documents_fixed.py first.")