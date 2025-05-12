import os
import sys
import importlib
from hongikjiki.utils import load_dotenv, setup_logging

# Set up logging
logger = setup_logging()

# Load environment variables
load_dotenv()
logger.info("환경 변수 로드 완료")

# Check for API key
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    logger.error("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다")
    print("오류: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다. 먼저 다음 명령을 실행하세요:")
    print("export OPENAI_API_KEY=your_api_key_here")
    sys.exit(1)

print(f"OpenAI API 키 확인: {api_key[:5]}...{api_key[-5:]}")
logger.info(f"OpenAI API 키 확인: {api_key[:5]}...{api_key[-5:]}")

try:
    # Try loading the embeddings module to test API key
    from hongikjiki.vector_store.embeddings import get_embeddings
    embeddings = get_embeddings("openai", api_key=api_key, model_name="text-embedding-3-small")
    logger.info("임베딩 모듈 로드 성공")
    print("임베딩 모듈 로드 성공")
    
    # Try loading the LLM
    from hongikjiki.langchain_integration.llm import get_llm
    llm = get_llm(llm_type="openai", api_key=api_key, model="gpt-4o")
    logger.info("LLM 모듈 로드 성공")
    print("LLM 모듈 로드 성공")
    
    # Import and run the Gradio app
    print("Gradio 앱 시작 중...")
    import gradio as gr
    
    # Define a simplified chatbot
    def simple_chat(message):
        response = llm.generate(f"다음 질문에 대해 홍익인간과 정법의 관점에서 답변해주세요: {message}")
        return f"💬 {response}"
    
    # Create a simple Gradio interface
    demo = gr.Interface(
        fn=simple_chat,
        inputs=gr.Textbox(placeholder="질문을 입력하세요...", label="질문"),
        outputs="text",
        title="홍익지기 챗봇 (간단 버전)",
        description="정법 강의 기반 챗봇의 간단 버전입니다."
    )
    
    # Launch the app
    print("앱이 시작되면 웹 브라우저에서 접속할 수 있는 URL이 표시됩니다.")
    demo.launch()
    
except Exception as e:
    import traceback
    error_details = traceback.format_exc()
    logger.error(f"오류 발생: {e}\n{error_details}")
    print(f"오류 발생: {e}")
    print("\n상세 오류 정보:")
    print(error_details)
    sys.exit(1)