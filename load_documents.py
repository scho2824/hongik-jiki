# load_documents.py
from pathlib import Path
from hongikjiki.modules.text_processing.document_processor import DocumentProcessor
from hongikjiki.modules.vector_store.chroma_store import ChromaVectorStore
from hongikjiki.modules.vector_store.embeddings import get_embeddings

# Define paths
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data" / "jungbub_teachings"

# Initialize components
processor = DocumentProcessor()
embeddings = get_embeddings("openai", model="text-embedding-ada-002")
vector_store = ChromaVectorStore(
    collection_name="hongikjiki_jungbub",
    persist_directory=str(ROOT_DIR / "data" / "vector_store"),
    embeddings=embeddings
)

# Process files
print(f"Loading documents from {DATA_DIR}...")
document_chunks = processor.process_directory(str(DATA_DIR))
print(f"Processed {len(document_chunks)} chunks.")

# Add to vector store
if document_chunks:
    vector_ids = vector_store.add_documents(document_chunks)
    print(f"Added {len(vector_ids)} chunks to vector store.")
    
# Check count
count = vector_store.count()
print(f"Vector store now contains {count} documents.")