#!/usr/bin/env python3
"""
Document Repair Script for Hongik-Jiki Chatbot

This script checks and repairs documents in the vector store,
ensuring proper indexing and retrieval capabilities.
"""

import os
import sys
import logging
import shutil
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("document_repair.log")
    ]
)
logger = logging.getLogger("DocumentRepair")

# Load environment variables
load_dotenv()

# Add module path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

# Import Hongik-Jiki modules
try:
    from hongikjiki.text_processing import DocumentProcessor
    from hongikjiki.vector_store import ChromaVectorStore
    logger.info("Successfully imported Hongik-Jiki modules")
except ImportError:
    try:
        # Alternative import paths
        from hongikjiki.text_processing.document_processor import DocumentProcessor
        from hongikjiki.vector_store.chroma_store import ChromaVectorStore
        logger.info("Successfully imported Hongik-Jiki modules using alternative paths")
    except ImportError as e:
        logger.error(f"Failed to import Hongik-Jiki modules: {e}")
        sys.exit(1)

def check_environment():
    """Check environment configuration"""
    logger.info("Checking environment configuration...")
    
    # Check essential environment variables
    env_vars = {
        'DATA_DIR': os.getenv('DATA_DIR', './data/jungbub_teachings'),
        'PERSIST_DIRECTORY': os.getenv('PERSIST_DIRECTORY', './data/vector_store'),
        'EMBEDDING_MODEL': os.getenv('EMBEDDING_MODEL', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'),
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY')
    }
    
    # Log environment variables (without API key)
    for key, value in env_vars.items():
        if key == 'OPENAI_API_KEY':
            if value:
                logger.info(f"  {key}: [Set]")
            else:
                logger.error(f"  {key}: [Not Set] - Required for OpenAI LLM")
        else:
            logger.info(f"  {key}: {value}")
    
    # Check if data directory exists
    data_dir = env_vars['DATA_DIR']
    if not os.path.exists(data_dir):
        logger.error(f"Data directory does not exist: {data_dir}")
        logger.info(f"Creating directory: {data_dir}")
        os.makedirs(data_dir, exist_ok=True)
    
    # Check if vector store directory exists
    persist_dir = env_vars['PERSIST_DIRECTORY']
    if not os.path.exists(persist_dir):
        logger.warning(f"Vector store directory does not exist: {persist_dir}")
        logger.info(f"Creating directory: {persist_dir}")
        os.makedirs(persist_dir, exist_ok=True)
    
    return env_vars

def scan_document_directory(data_dir: str) -> List[str]:
    """
    Scan the document directory for files
    
    Args:
        data_dir: Path to the document directory
        
    Returns:
        List of file paths
    """
    logger.info(f"Scanning document directory: {data_dir}")
    
    if not os.path.exists(data_dir):
        logger.error(f"Directory does not exist: {data_dir}")
        return []
    
    # Get all files recursively
    all_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            # Skip hidden files and non-text files
            if file.startswith('.') or not any(file.endswith(ext) for ext in ['.txt', '.pdf', '.docx', '.rtf']):
                continue
            
            file_path = os.path.join(root, file)
            all_files.append(file_path)
    
    # Log summary
    logger.info(f"Found {len(all_files)} document files")
    
    if all_files:
        # Log first few files
        logger.info("Sample files:")
        for file in all_files[:5]:
            logger.info(f"  {file}")
    else:
        logger.warning("No document files found!")
    
    return all_files

def check_vector_store(persist_dir: str, embedding_model: str) -> Tuple[Optional[ChromaVectorStore], int]:
    """
    Check the vector store for existing documents
    
    Args:
        persist_dir: Path to the vector store directory
        embedding_model: Name of the embedding model
        
    Returns:
        Tuple of (vector store instance, document count)
    """
    logger.info(f"Checking vector store at: {persist_dir}")
    
    try:
        # Initialize vector store
        vector_store = ChromaVectorStore(
            model_name=embedding_model,
            persist_directory=persist_dir
        )
        
        # Check document count
        doc_count = vector_store.collection.count()
        logger.info(f"Vector store contains {doc_count} documents")
        
        return vector_store, doc_count
    except Exception as e:
        logger.error(f"Error checking vector store: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, 0

def backup_vector_store(persist_dir: str) -> bool:
    """
    Create a backup of the vector store
    
    Args:
        persist_dir: Path to the vector store directory
        
    Returns:
        True if backup successful, False otherwise
    """
    logger.info(f"Creating backup of vector store: {persist_dir}")
    
    if not os.path.exists(persist_dir):
        logger.warning(f"Vector store directory does not exist: {persist_dir}")
        return False
    
    try:
        # Create backup directory
        backup_dir = f"{persist_dir}_backup_{int(time.time())}"
        shutil.copytree(persist_dir, backup_dir)
        logger.info(f"Vector store backup created: {backup_dir}")
        return True
    except Exception as e:
        logger.error(f"Error creating vector store backup: {e}")
        return False

def rebuild_vector_store(data_dir: str, persist_dir: str, embedding_model: str) -> bool:
    """
    Rebuild the vector store from scratch
    
    Args:
        data_dir: Path to the document directory
        persist_dir: Path to the vector store directory
        embedding_model: Name of the embedding model
        
    Returns:
        True if rebuild successful, False otherwise
    """
    logger.info("Beginning vector store rebuild process...")
    
    # Scan document directory
    documents = scan_document_directory(data_dir)
    
    if not documents:
        logger.error("No documents found to process")
        return False
    
    # Backup existing vector store
    if os.path.exists(persist_dir):
        logger.info("Creating backup of existing vector store")
        backup_successful = backup_vector_store(persist_dir)
        
        if backup_successful:
            # Remove existing vector store
            logger.info(f"Removing existing vector store: {persist_dir}")
            try:
                shutil.rmtree(persist_dir)
                os.makedirs(persist_dir, exist_ok=True)
            except Exception as e:
                logger.error(f"Error removing existing vector store: {e}")
                return False
    
    try:
        logger.info("Initializing document processor...")
        document_processor = DocumentProcessor()
        
        logger.info("Initializing new vector store...")
        vector_store = ChromaVectorStore(
            model_name=embedding_model,
            persist_directory=persist_dir
        )
        
        # Process and add documents
        logger.info(f"Processing {len(documents)} documents...")
        
        total_chunks = 0
        processed_files = 0
        failed_files = 0
        
        for i, doc_path in enumerate(documents):
            try:
                logger.info(f"Processing document [{i+1}/{len(documents)}]: {os.path.basename(doc_path)}")
                
                # Process the document
                chunks = document_processor.process_file(doc_path)
                
                if chunks:
                    # Add chunks to vector store
                    logger.info(f"Adding {len(chunks)} chunks to vector store...")
                    vector_store.add_documents(chunks)
                    
                    total_chunks += len(chunks)
                    processed_files += 1
                else:
                    logger.warning(f"No chunks generated for: {doc_path}")
                    failed_files += 1
            
            except Exception as e:
                logger.error(f"Error processing document {doc_path}: {e}")
                failed_files += 1
        
        # Log summary
        logger.info("=" * 50)
        logger.info("Vector store rebuild summary:")
        logger.info(f"  Total files processed: {processed_files}")
        logger.info(f"  Total files failed: {failed_files}")
        logger.info(f"  Total chunks added: {total_chunks}")
        logger.info("=" * 50)
        
        if processed_files > 0:
            return True
        else:
            logger.error("No files were successfully processed")
            return False
    
    except Exception as e:
        logger.error(f"Error rebuilding vector store: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_vector_store_queries(vector_store: ChromaVectorStore) -> None:
    """
    Test vector store with various queries
    
    Args:
        vector_store: Vector store instance
    """
    logger.info("Testing vector store with sample queries...")
    
    test_queries = [
        "홍익인간",  # Hongik Ingan
        "정법",      # Jungbub
        "천공",      # Cheongong
        "어려움",    # Hardship
        "해맞이"     # Sunrise greeting
    ]
    
    for query in test_queries:
        logger.info(f"Testing query: '{query}'")
        
        try:
            # Get documents
            docs = vector_store.similarity_search(query, k=2)
            
            if docs:
                logger.info(f"Query '{query}' returned {len(docs)} documents")
                
                # Log first document
                content = docs[0].page_content if hasattr(docs[0], 'page_content') else str(docs[0])
                metadata = docs[0].metadata if hasattr(docs[0], 'metadata') else {}
                
                logger.info(f"  Top document source: {metadata.get('filename', 'Unknown')}")
                preview = content[:100] + "..." if len(content) > 100 else content
                logger.info(f"  Content preview: {preview}")
            else:
                logger.warning(f"Query '{query}' returned no documents")
        
        except Exception as e:
            logger.error(f"Error testing query '{query}': {e}")

def main():
    """Main function"""
    import time
    
    logger.info("=" * 80)
    logger.info("Hongik-Jiki Document Repair Tool")
    logger.info("=" * 80)
    
    # Check environment
    env_vars = check_environment()
    data_dir = env_vars['DATA_DIR']
    persist_dir = env_vars['PERSIST_DIRECTORY']
    embedding_model = env_vars['EMBEDDING_MODEL']
    
    # Check if OpenAI API key is set
    if not env_vars['OPENAI_API_KEY']:
        logger.warning("OPENAI_API_KEY is not set. Some functions may not work properly.")
    
    # Check vector store
    vector_store, doc_count = check_vector_store(persist_dir, embedding_model)
    
    # Determine action based on vector store state
    if vector_store is None:
        logger.warning("Vector store could not be initialized. Will attempt to rebuild.")
        rebuild_required = True
    elif doc_count == 0:
        logger.warning("Vector store is empty. Will rebuild with available documents.")
        rebuild_required = True
    else:
        # Ask for confirmation
        print("\nVector store appears to be functioning with", doc_count, "documents.")
        choice = input("Do you want to rebuild it anyway? (y/N): ").lower()
        rebuild_required = choice == 'y'
    
    # Rebuild vector store if required
    if rebuild_required:
        logger.info("Starting vector store rebuild process...")
        
        rebuild_success = rebuild_vector_store(data_dir, persist_dir, embedding_model)
        
        if rebuild_success:
            logger.info("Vector store rebuild completed successfully!")
            
            # Initialize vector store again for testing
            vector_store, doc_count = check_vector_store(persist_dir, embedding_model)
            
            if vector_store and doc_count > 0:
                # Test queries
                test_vector_store_queries(vector_store)
        else:
            logger.error("Vector store rebuild failed!")
    else:
        logger.info("Skipping vector store rebuild.")
        
        # Test existing vector store
        if vector_store:
            test_vector_store_queries(vector_store)
    
    logger.info("Document repair process completed.")

if __name__ == "__main__":
    import time
    main()
