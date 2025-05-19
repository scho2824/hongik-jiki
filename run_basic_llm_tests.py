import os
import logging
from dotenv import load_dotenv

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file in the current directory
# Ensure this script is run from the project root where .env is located.
load_dotenv()

# Attempt to import the LLM classes
try:
    from hongikjiki.langchain_integration.llm import OpenAILLM, NaverClovaLLM
except ImportError as e:
    logger.error(f"Failed to import LLM modules: {e}")
    logger.error("Please ensure that this script is run from the project root directory ('/Users/swthehongik/Documents/Hongik-Jiki/')")
    logger.error("and that the 'hongikjiki' package is correctly structured and accessible.")
    exit(1)

def test_openai_llm():
    logger.info("Attempting to test OpenAILLM...")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not found in environment variables. Skipping OpenAI LLM test.")
        return

    try:
        # You can specify model, temperature, max_tokens if needed, or rely on defaults
        llm = OpenAILLM(api_key=api_key)
        prompt = "Hello, OpenAI! Can you tell me a fun fact?"
        logger.info(f"Sending prompt to OpenAI: '{prompt}'")
        
        # OpenAILLM has a test_completion method
        response = llm.test_completion(prompt)
        
        logger.info(f"OpenAI Response: {response}")
        if not response or "오류 발생" in response or "[오류 발생" in response:
            logger.error("OpenAI LLM test might have failed or returned an error/empty response.")
        else:
            logger.info("OpenAI LLM test completed.")
    except Exception as e:
        logger.error(f"Error during OpenAI LLM test: {e}", exc_info=True)

def test_clova_llm():
    logger.info("Attempting to test NaverClovaLLM...")
    # api_key_studio = os.getenv("CLOVA_API_KEY") # Clova Studio 자체 API 키가 있다면 이 변수를 사용
    naver_client_id = os.getenv("NAVER_CLIENT_ID")
    naver_client_secret = os.getenv("NAVER_CLIENT_SECRET")
    api_gateway = os.getenv("CLOVA_API_GATEWAY") # Optional, uses default if not set

    if not naver_client_id or not naver_client_secret:
        logger.warning("NAVER_CLIENT_ID or NAVER_CLIENT_SECRET not found in .env. Skipping Naver Clova LLM test.")
        return

    try:
        # NaverClovaLLM 클래스는 app_id, api_key, api_key_primary_val 등을 파라미터로 받습니다.
        # Client ID는 app_id로, Client Secret은 api_key_primary_val로 매핑하는 것을 시도해볼 수 있습니다.
        # NaverClovaLLM 클래스가 X-NCP-CLOVASTUDIO-API-KEY (api_key 파라미터)를 필수로 요구하고,
        # 이 값이 Client Secret과 동일하게 사용될 수 있다면 api_key에도 naver_client_secret을 전달합니다.
        llm_params = {
            "ncp_client_id": naver_client_id,
            "ncp_client_secret": naver_client_secret,
            "temperature": 0.1,  # Explicitly set temperature to a float value
            # "clova_studio_api_key": os.getenv("CLOVA_API_KEY"), # 만약 별도의 Clova Studio 키가 있다면 사용
        }
        if api_gateway:
            llm_params["api_gateway"] = api_gateway
            
        llm = NaverClovaLLM(**llm_params)
        prompt = "안녕하세요, 클로바! 재미있는 사실 하나 알려줄 수 있나요?" # 프롬프트를 한국어로 변경해볼 수 있습니다.
        logger.info(f"Sending prompt to Naver Clova with params {llm_params}: '{prompt}'")
        response = llm.generate(prompt)
        logger.info(f"Naver Clova Response: {response}")
        if not response or "[API 오류" in response or "[오류 발생" in response:
            logger.error("Naver Clova LLM test might have failed or returned an error/empty response.")
        else:
            logger.info("Naver Clova LLM test completed.")
    except Exception as e:
        logger.error(f"Error during Naver Clova LLM test: {e}", exc_info=True)
if __name__ == "__main__":
    logger.info("Starting basic LLM functionality tests...")
    test_openai_llm()
    logger.info("-" * 50)
    test_clova_llm() # Re-enable Clova LLM test
    logger.info("Basic LLM tests finished.")
    logger.info("For more comprehensive testing, consider setting up a test suite using a framework like pytest or unittest.")