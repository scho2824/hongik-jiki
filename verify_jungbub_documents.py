# verify_jungbub_documents.py - revised version
import os
import logging
from dotenv import load_dotenv
from hongikjiki.vector_store.chroma_store import ChromaVectorStore
from hongikjiki.vector_store.embeddings import get_embeddings

# Load environment variables
load_dotenv()

def setup_logging():
    """Set up basic logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("JungbubVerifier")

def verify_documents():
    """Verify documents in the vector store with test queries"""
    logger = setup_logging()
    
    try:
        # Check for OpenAI API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY environment variable not set!")
            return False
        
        # Initialize embeddings and vector store with correct parameters
        logger.info("Initializing OpenAI embeddings...")
        embeddings = get_embeddings("openai", model="text-embedding-3-small")
        
        # Initialize vector store
        logger.info("Connecting to vector store...")
        vector_store = ChromaVectorStore(
            collection_name="hongikjiki_jungbub",
            persist_directory="./data/vector_store",
            embeddings=embeddings
        )
        
        # Get document count
        doc_count = vector_store.count()
        logger.info(f"Total documents in vector store: {doc_count}")
        
        if doc_count == 0:
            logger.warning("No documents found in vector store!")
            return False
        
        # Run test queries
        test_queries = [
            "홍익인간이란 무엇인가요?",
            "천공 스승님은 누구신가요?",
            "정법의 핵심 가르침은 무엇인가요?",
            "용서란 무엇인가요?",
            "자연의 법칙을 따르는 것이 왜 중요한가요?"
        ]
        
        print("\n=== Running Test Queries ===")
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            results = vector_store.search(query, k=2)
            
            if not results:
                print("No results found!")
                continue
            
            for i, result in enumerate(results):
                print(f"\nResult {i+1} (similarity score: {result['score']:.4f})")
                
                # Print metadata in a safe way that handles missing keys
                metadata = result.get('metadata', {})
                lecture_num = metadata.get('lecture_number', 'N/A')
                title = metadata.get('title', 'Untitled')
                content_type = metadata.get('content_type', 'Unknown')
                
                print(f"Source: 정법 {lecture_num}강 - {title} ({content_type})")
                
                # Print content preview safely
                content = result.get('content', '')
                content_preview = content[:300] + "..." if len(content) > 300 else content
                print(f"Content: {content_preview}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during verification: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = verify_documents()
    print(f"\nVerification result: {'Success' if success else 'Failed'}")