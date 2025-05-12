# fixed_check_vector_store.py
import os
import sys
from hongikjiki.utils import load_dotenv, setup_logging

# Set up logging
logger = setup_logging()
logger.info("Vector store diagnostic starting")

# Load environment variables
load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("Error: OPENAI_API_KEY environment variable not found")
    sys.exit(1)

# Import necessary modules
try:
    import chromadb
    from chromadb.config import Settings
    from hongikjiki.vector_store.embeddings import get_embeddings
    from hongikjiki.vector_store.chroma_store import ChromaVectorStore
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)

# Vector store settings
PERSIST_DIR = "data/vector_store"
COLLECTION_NAME = "hongikjiki_jungbub"

print(f"Vector store diagnostic: {PERSIST_DIR}, collection: {COLLECTION_NAME}")

# 1. Check if directory exists
if os.path.exists(PERSIST_DIR):
    print(f"✅ Vector store directory exists: {PERSIST_DIR}")
    # Check directory contents
    print("Directory contents:")
    for item in os.listdir(PERSIST_DIR):
        print(f"  - {item}")
else:
    print(f"❌ Vector store directory doesn't exist: {PERSIST_DIR}")
    # Create directory
    print("Creating directory...")
    os.makedirs(PERSIST_DIR, exist_ok=True)
    print(f"✅ Directory created: {PERSIST_DIR}")

# 2. Try to create ChromaDB client
try:
    print("\nAttempting to create client...")
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    print("✅ Client created successfully")
except Exception as e:
    print(f"❌ Failed to create client: {e}")
    sys.exit(1)

# 3. Check collections using the compatible method for v0.6.0
try:
    print("\nChecking collections (compatible with v0.6.0):")
    collection_names = client.list_collections()
    if collection_names:
        print(f"Found {len(collection_names)} collections:")
        for coll_name in collection_names:
            print(f"  - {coll_name}")
    else:
        print("  - No collections found")
except Exception as e:
    print(f"❌ Failed to check collections: {e}")

# 4. Try to use ChromaVectorStore with proper initialization
try:
    print("\nAttempting to create ChromaVectorStore...")
    embeddings = get_embeddings("openai", model="text-embedding-ada-002", api_key=api_key)
    vector_store = ChromaVectorStore(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
        embeddings=embeddings
    )
    print("✅ ChromaVectorStore created successfully")
    
    # Get document count directly
    try:
        doc_count = vector_store.count()
        print(f"Collection document count: {doc_count}")
        
        if doc_count == 0:
            print("\n⚠️ No documents found.")
            print("You need to add documents. Run your document ingestion script.")
        else:
            print("\nAttempting sample query...")
            results = vector_store.search("정법이란 무엇인가요?", k=2)
            print(f"Search results count: {len(results)}")
            for i, result in enumerate(results):
                print(f"Result {i+1}:")
                print(f"  - Score: {result.get('score', 'N/A')}")
                content = result.get('content', 'N/A')
                print(f"  - Content: {content[:100]}...")
    except Exception as e:
        print(f"❌ Failed to get document count or query: {e}")
    
except Exception as e:
    print(f"❌ Failed to use ChromaVectorStore: {e}")

print("\nDiagnostic complete.")