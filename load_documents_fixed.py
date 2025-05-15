# load_documents_fixed.py
from pathlib import Path
import os
import logging
from hongikjiki.modules.text_processing.document_processor import DocumentProcessor
from hongikjiki.modules.vector_store.chroma_store import ChromaVectorStore
from hongikjiki.modules.vector_store.embeddings import get_embeddings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DocumentLoader")

# Define paths
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data" / "jungbub_teachings"
SUPPORTED_EXTENSIONS = ['.txt', '.pdf', '.docx', '.rtf', '.md']

# Initialize components
processor = DocumentProcessor()
embeddings = get_embeddings("openai", model="text-embedding-ada-002")
vector_store = ChromaVectorStore(
    collection_name="hongikjiki_jungbub",
    persist_directory=str(ROOT_DIR / "data" / "vector_store"),
    embeddings=embeddings
)

def process_files():
    """Process all files in the data directory"""
    total_chunks = 0
    total_files = 0
    processed_files = 0
    
    # Find all files recursively
    for root, _, files in os.walk(DATA_DIR):
        for filename in files:
            file_path = os.path.join(root, filename)
            ext = os.path.splitext(filename)[1].lower()
            
            if ext in SUPPORTED_EXTENSIONS:
                total_files += 1
                
                try:
                    # Process file
                    logger.info(f"Processing file {total_files}: {file_path}")
                    chunks = processor.process_file(file_path)
                    
                    if chunks:
                        # Add to vector store in batches
                        vector_ids = vector_store.add_documents(chunks)
                        logger.info(f"Added {len(chunks)} chunks from {filename}")
                        
                        total_chunks += len(chunks)
                        processed_files += 1
                    else:
                        logger.warning(f"No chunks generated for {filename}")
                        
                except Exception as e:
                    logger.error(f"Error processing {filename}: {e}")
    
    logger.info(f"Processed {processed_files} files out of {total_files}")
    logger.info(f"Added {total_chunks} chunks to vector store")
    
    # Check final count
    count = vector_store.count()
    logger.info(f"Vector store now contains {count} documents")

if __name__ == "__main__":
    logger.info(f"Processing files in {DATA_DIR}")
    process_files()