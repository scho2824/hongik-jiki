# fixed_app.py
import os
import gradio as gr
from hongikjiki.utils import load_dotenv, setup_logging
from hongikjiki.langchain_integration.llm import get_llm
from hongikjiki.vector_store.embeddings import get_embeddings
from hongikjiki.vector_store.chroma_store import ChromaVectorStore

# Set up logging
logger = setup_logging()
logger.info("Starting fixed app")

# Load environment variables
load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    logger.error("OPENAI_API_KEY not found in environment variables")
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Try to initialize the vector store
vector_store = None
try:
    logger.info("Initializing vector store...")
    embeddings = get_embeddings("openai", model="text-embedding-ada-002", api_key=api_key)
    vector_store = ChromaVectorStore(
        collection_name="hongikjiki_jungbub",
        persist_directory="data/vector_store",
        embeddings=embeddings
    )
    doc_count = vector_store.count()
    logger.info(f"Vector store initialized with {doc_count} documents")
    print(f"Vector store contains {doc_count} documents")
    
    # Test query to verify it's working
    if doc_count > 0:
        test_results = vector_store.search("정법이란 무엇인가요?", k=1)
        if test_results:
            logger.info("Vector store query test successful")
            print("Vector store query test successful")
        else:
            logger.warning("Vector store query returned no results")
            print("Vector store query returned no results")
    
except Exception as e:
    logger.error(f"Failed to initialize vector store: {e}")
    print(f"Failed to initialize vector store: {e}")
    vector_store = None

# Initialize LLM
try:
    logger.info("Initializing LLM...")
    llm = get_llm(llm_type="openai", model="gpt-4o", api_key=api_key)
    logger.info("LLM initialized successfully")
    print("LLM initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize LLM: {e}")
    raise ValueError(f"Failed to initialize LLM: {e}")

def answer_question(message):
    """Answer user questions using vector store if available, otherwise fallback to direct LLM"""
    logger.info(f"Received question: {message}")
    
    try:
        if vector_store and vector_store.count() > 0:
            logger.info("Using vector store to find relevant documents")
            # Get relevant documents
            results = vector_store.search(message, k=3)
            
            if results:
                # Prepare context from search results
                context = "\n\n".join([f"문서 {i+1}:\n{r.get('content', '')}" for i, r in enumerate(results)])
                
                # Create prompt with retrieved context
                prompt = f"""
                당신은 정법 지식을 제공하는 홍익지기 챗봇입니다.
                아래 제공된 정법 문서를 참고하여 질문에 답변해주세요.
                
                ### 관련 정법 문서:
                {context}
                
                ### 사용자 질문:
                {message}
                
                ### 답변:
                """
                
                logger.info("Generating answer with context from vector store")
                response = llm.generate(prompt)
                return f"💬 {response}"
            else:
                logger.warning("Vector store returned no results, falling back to direct LLM")
        else:
            logger.info("Vector store not available, using direct LLM")
        
        # Fallback to direct LLM if vector store is not available or returned no results
        prompt = f"""
        당신은 정법 지식을 제공하는 홍익지기 챗봇입니다.
        다음 질문에 대해 홍익인간과 정법의 관점에서 답변해주세요:
        
        질문: {message}
        
        답변:
        """
        response = llm.generate(prompt)
        return f"💬 {response}"
        
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        return f"오류가 발생했습니다: {str(e)}"

# Create simple interface
iface = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(placeholder="질문을 입력하세요...", label="질문"),
    outputs="text",
    title="홍익지기 챗봇",
    description="정법 강의를 기반으로 질문에 답하는 챗봇입니다."
)

# Add example questions
gr.Examples(
    examples=[
        "정법이란 무엇인가요?",
        "감정이 자꾸 요동치는 이유가 뭘까요?",
        "왜 인간관계가 이렇게 어려운 걸까요?",
        "무기력함을 어떻게 이겨낼 수 있죠?"
    ],
    inputs=iface.input_components[0]
)

if __name__ == "__main__":
    logger.info("Launching Gradio interface")
    print("Launching Hongik-Jiki chatbot...")
    iface.launch()