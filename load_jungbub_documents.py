# 메타데이터에서 None 값을 제거하는 함수
def sanitize_metadata(metadata):
     """메타데이터에서 None 값 제거"""
     return {k: (v if v is not None else "") for k, v in metadata.items()}
def split_into_smaller_chunks(text, max_chunk_size=800):
    """
    텍스트를 최대 청크 크기로 분할

    Args:
        text: 분할할 텍스트
        max_chunk_size: 최대 청크 크기 (문자 수)

    Returns:
        List[str]: 분할된 텍스트 청크 리스트
    """
    # 문장 단위로 분할
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # 현재 문장이 최대 청크 크기를 초과하면 그 자체로 청크로 처리
        if len(sentence) > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""

            # 긴 문장을 더 작은 부분으로 분할
            for i in range(0, len(sentence), max_chunk_size):
                sub_chunk = sentence[i:i+max_chunk_size]
                chunks.append(sub_chunk)

        # 현재 청크에 문장을 추가했을 때 최대 크기를 초과하는 경우
        elif len(current_chunk) + len(sentence) > max_chunk_size:
            chunks.append(current_chunk)
            current_chunk = sentence

        # 그렇지 않으면 현재 청크에 문장 추가
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence

    # 마지막 청크 추가
    if current_chunk:
        chunks.append(current_chunk)

    return chunks
import os
import json
import hashlib
import logging
import re
from typing import List, Dict, Any
from dotenv import load_dotenv
import time

# Import custom modules
from hongikjiki.text_processing.document_processor import DocumentProcessor
from hongikjiki.vector_store.chroma_store import ChromaVectorStore 
from hongikjiki.vector_store.embeddings import get_embeddings

from langchain.text_splitter import CharacterTextSplitter

# Load environment variables
load_dotenv()

def setup_logging() -> logging.Logger:
    """Set up logging configuration"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger("JungbubDocLoader")
    logger.setLevel(logging.INFO)
    
    # Create file handler
    file_handler = logging.FileHandler(os.path.join(log_dir, "document_loading.log"))
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def load_processed_files_record() -> Dict[str, str]:
    """Load record of processed files with their hash values"""
    record_file = "data/processed_files.json"
    
    if os.path.exists(record_file):
        try:
            with open(record_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading processed files record: {e}")
            return {}
    else:
        return {}

def update_processed_files_record(processed_files: Dict[str, str]) -> None:
    """Update record of processed files"""
    record_file = "data/processed_files.json"
    os.makedirs(os.path.dirname(record_file), exist_ok=True)
    
    try:
        with open(record_file, 'w', encoding='utf-8') as f:
            json.dump(processed_files, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Error updating processed files record: {e}")

def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA-256 hash of file contents"""
    try:
        hash_obj = hashlib.sha256()
        with open(file_path, 'rb') as f:
            # Read file in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b''):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating hash for {file_path}: {e}")
        return ""

def main() -> None:
    """Main function to load Jungbub documents"""
    global logger
    logger = setup_logging()
    
    try:
        # Display start message
        logger.info("=== Starting Jungbub Documents Loading Process ===")
        
        # Check for OpenAI API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY environment variable not set!")
            return
        
        # Initialize embeddings
        logger.info("Initializing HuggingFace embeddings...")
        embeddings = get_embeddings("huggingface", model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        
        # Initialize vector store
        logger.info("Initializing vector store...")
        vector_store = ChromaVectorStore(
            collection_name="hongikjiki_jungbub",
            persist_directory="./data/vector_store",
            embeddings=embeddings
        )
        
        # Load record of processed files
        processed_files = load_processed_files_record()
        logger.info(f"Loaded record of {len(processed_files)} previously processed files")
        
        # Initialize document processor
        processor = DocumentProcessor()
        
        # Track processed files and added documents
        new_files_dict = {}
        total_processed_files = 0
        total_added_chunks = 0
        
        # Process documents recursively
        data_dir = "data/jungbub_teachings"
        logger.info(f"Scanning directory: {data_dir}")
        
        for root, dirs, files in os.walk(data_dir):
            logger.info(f"Processing directory: {root} - Found {len(files)} files")
            for file in files:
                file_path = os.path.join(root, file)
                
                # Skip non-text files based on extension
                if not file.lower().endswith(('.txt', '.rtf', '.pdf', '.docx')):
                    logger.debug(f"Skipping non-text file: {file_path}")
                    continue
                
                # Calculate file hash
                file_hash = calculate_file_hash(file_path)
                if not file_hash:
                    logger.warning(f"Skipping {file_path} due to hash calculation error")
                    continue
                
                # Check if file is already processed and hash hasn't changed
                if file_path in processed_files and processed_files[file_path] == file_hash:
                    logger.debug(f"Skipping already processed file: {file_path}")
                    continue
                
                # Process new file
                logger.info(f"Processing document: {file_path}*********")
                try:
                    start_time = time.time()
                    # Direct chunk splitting and robust handling
                    if file_path.lower().endswith(".txt"):
                        with open(file_path, "r", encoding="utf-8") as f:
                            text = f.read()
                        chunks = split_into_smaller_chunks(text, max_chunk_size=800)
                        logger.info(f"Split .txt file into {len(chunks)} chunks using split_into_smaller_chunks")
                        chunk_items = [{"content": c, "metadata": sanitize_metadata({"source": file_path})} for c in chunks]
                    else:
                        processed_chunks = processor.process_file(file_path)
                        logger.info(f"Processed file with DocumentProcessor, got {len(processed_chunks)} chunks")
                        chunk_items = []
                        for chunk in processed_chunks:
                            if isinstance(chunk, dict) and "content" in chunk:
                                chunk["metadata"] = sanitize_metadata(chunk.get("metadata", {"source": file_path}))
                                chunk_items.append(chunk)
                            elif isinstance(chunk, str):
                                chunk_items.append({"content": chunk, "metadata": sanitize_metadata({"source": file_path})})
                            else:
                                logger.warning(f"Unknown chunk format in {file_path}: {type(chunk)}")
                    logger.info(f"Total chunk items to embed: {len(chunk_items)}")

                    if chunk_items:
                        logger.info(f"Embedding {len(chunk_items)} chunks for '{file_path}'")
                        file_added = 0
                        batch_size = 2
                        for batch_start in range(0, len(chunk_items), batch_size):
                            batch = chunk_items[batch_start:batch_start+batch_size]
                            texts = []
                            metadatas = []
                            for chunk in batch:
                                content = chunk.get("content", "")
                                metadata = chunk.get("metadata", {}) or {}
                                # Recursively split if content is too long
                                max_chunk_size = 800
                                if len(content) > max_chunk_size:
                                    sub_chunks = split_into_smaller_chunks(content, max_chunk_size=max_chunk_size)
                                    for idx, sub_chunk in enumerate(sub_chunks):
                                        sub_metadata = {"source": file_path, "sub_chunk_index": idx}
                                        for k, v in metadata.items():
                                            if v is not None and isinstance(v, (str, int, float, bool)):
                                                sub_metadata[k] = v
                                        texts.append(sub_chunk.strip())
                                        metadatas.append(sub_metadata)
                                else:
                                    clean_metadata = {"source": file_path}
                                    for k, v in metadata.items():
                                        if v is not None and isinstance(v, (str, int, float, bool)):
                                            clean_metadata[k] = v
                                    texts.append(content)
                                    metadatas.append(clean_metadata)
                            # Now check if any chunk is still too large; recursively split down to 400 chars if needed
                            final_texts = []
                            final_metadatas = []
                            for text, meta in zip(texts, metadatas):
                                if len(text) > 1200:
                                    logger.warning(f"Chunk over 1200 chars, recursively splitting to <=400 in {file_path}")
                                    subchunks = split_into_smaller_chunks(text, max_chunk_size=400)
                                    for i, sub in enumerate(subchunks):
                                        m = dict(meta)
                                        m["recursive_split"] = True
                                        m["recursive_index"] = i
                                        final_texts.append(sub)
                                        final_metadatas.append(m)
                                else:
                                    final_texts.append(text)
                                    final_metadatas.append(meta)
                            # Add in vector store in microbatches
                            for i in range(0, len(final_texts), 2):
                                batch_texts = final_texts[i:i+2]
                                batch_metadatas = final_metadatas[i:i+2]
                                try:
                                    vector_store.add_texts(
                                        [t for t in batch_texts],
                                        metadatas=[sanitize_metadata(m) for m in batch_metadatas]
                                    )
                                    logger.info(f"Added batch {i//2+1} of {(len(final_texts)+1)//2} (size: {len(batch_texts)})")
                                except Exception as e:
                                    logger.error(f"Error adding batch {i//2+1}: {e}")
                                    # Try adding individually
                                    for j, (t, m) in enumerate(zip(batch_texts, batch_metadatas)):
                                        try:
                                            vector_store.add_texts([t], metadatas=[m])
                                            logger.info(f"Added individual chunk {i+j+1} successfully")
                                        except Exception as e2:
                                            logger.error(f"Error adding individual chunk {i+j+1}: {e2}")
                                            # Try splitting further if too big
                                            if len(t) > 500:
                                                logger.warning(f"Trying to split problematic chunk down to <=400 chars")
                                                mini_chunks = split_into_smaller_chunks(t, max_chunk_size=400)
                                                for k, mini in enumerate(mini_chunks):
                                                    mini_meta = dict(m)
                                                    mini_meta["mini_chunk"] = True
                                                    mini_meta["mini_index"] = k
                                                    try:
                                                        vector_store.add_texts([mini], metadatas=[mini_meta])
                                                        logger.info(f"Added mini chunk (size: {len(mini)})")
                                                    except Exception as e3:
                                                        logger.error(f"Failed to add mini chunk (size: {len(mini)}): {e3}")
                        file_added = len(chunk_items)
                        total_added_chunks += file_added
                        logger.info(f"Added {file_added} chunks to vector store for '{file_path}'")
                        new_files_dict[file_path] = file_hash
                        total_processed_files += 1
                        elapsed = time.time() - start_time
                        logger.info(f"Finished '{file_path}' in {elapsed:.2f}s")
                    else:
                        logger.warning(f"No chunks generated from {file_path}")
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
        
        # Update record of processed files
        processed_files.update(new_files_dict)
        update_processed_files_record(processed_files)
        
        # Report summary
        logger.info("=== Document Loading Summary ===")
        logger.info(f"Total processed files: {total_processed_files}")
        logger.info(f"Total added chunks: {total_added_chunks}")
        logger.info(f"Total documents in vector store: {vector_store.count()}")
        
        logger.info("=== Document Loading Process Completed ===")
    
    except Exception as e:
        logger.error(f"Error in document loading process: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()