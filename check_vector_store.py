# check_vector_store.py
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
    from hongikjiki.vector_store import get_embeddings
    from hongikjiki.vector_store.chroma_store import ChromaVectorStore
    from hongikjiki.vector_store import load_vector_store
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

# 3. Check collection list
try:
    print("\nChecking collection list:")
    collections = client.list_collections()
    if collections:
        for coll in collections:
            print(f"  - Name: {coll.name}, Metadata: {coll.metadata}")
    else:
        print("  - No collections found")
except Exception as e:
    print(f"❌ Failed to check collection list: {e}")

# 4. Try to get the collection
try:
    print(f"\nAttempting to get collection '{COLLECTION_NAME}'...")
    
    # Check if collection exists
    exists = False
    for coll in client.list_collections():
        if coll.name == COLLECTION_NAME:
            exists = True
            break
    
    if exists:
        collection = client.get_collection(name=COLLECTION_NAME)
        print(f"✅ Successfully retrieved existing collection: {COLLECTION_NAME}")
    else:
        collection = client.create_collection(name=COLLECTION_NAME)
        print(f"✅ Successfully created new collection: {COLLECTION_NAME}")
    
    # Check document count
    count = collection.count()
    print(f"Collection document count: {count}")
    
    if count == 0:
        print("\n⚠️ No documents found.")
        print("You need to add documents. Run your document ingestion script.")
    else:
        print("\nChecking some documents:")
        results = collection.get(limit=3)
        if results and len(results.get('ids', [])) > 0:
            for i, doc_id in enumerate(results['ids']):
                print(f"  - ID: {doc_id}")
                if 'metadatas' in results and results['metadatas'] and i < len(results['metadatas']):
                    print(f"    Metadata: {results['metadatas'][i]}")
                if 'documents' in results and results['documents'] and i < len(results['documents']):
                    doc_text = results['documents'][i]
                    print(f"    Text: {doc_text[:100]}...")
        else:
            print("  - Unable to retrieve documents.")
    
except Exception as e:
    print(f"❌ Failed to get collection: {e}")

# 5. Try to use ChromaVectorStore
try:
    print("\nAttempting to create ChromaVectorStore...")
    embeddings = get_embeddings("openai", model="text-embedding-ada-002", api_key=api_key)
    vector_store = ChromaVectorStore(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
        embeddings=embeddings
    )
    print("✅ ChromaVectorStore created successfully")
    
    # Try a sample query (if documents exist)
    if count > 0:
        print("\nAttempting sample query...")
        results = vector_store.search("정법이란 무엇인가요?", k=2)
        print(f"Search results count: {len(results)}")
        for i, result in enumerate(results):
            print(f"Result {i+1}:")
            print(f"  - Score: {result.get('score', 'N/A')}")
            print(f"  - Content: {result.get('content', 'N/A')[:100]}...")
    
except Exception as e:
    print(f"❌ Failed to use ChromaVectorStore: {e}")

print("\nDiagnostic complete.")