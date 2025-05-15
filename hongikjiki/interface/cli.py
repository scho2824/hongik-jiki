"""
hongikjiki.interface.cli

홍익지기 챗봇의 명령줄 인터페이스(CLI)를 제공합니다.

기능:
- 환경 설정(.env) 로드
- 텍스트 전처리 및 벡터 스토어 초기화
- 데이터베이스 비어 있을 경우 문서 로드 및 청크 생성
- 사용자의 입력을 받아 대화 생성 및 응답 제공
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Define ROOT_DIR
ROOT_DIR = Path(__file__).resolve().parents[2]

# 환경변수 로드
load_dotenv()

# 로깅 설정 - module-level logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from hongikjiki.utils.file_utils import setup_logging
from hongikjiki.interface.utils import print_welcome_message
from hongikjiki.modules.text_processing.document_processor import DocumentProcessor
from hongikjiki.modules.vector_store.chroma_store import ChromaVectorStore
from hongikjiki.modules.vector_store.embeddings import get_embeddings
from hongikjiki.core.chatbot import HongikJikiChatbot

# Updated import path for get_llm
from hongikjiki.langchain_integration.llm import get_llm

# Optional: Import tag extractor if you want to use it
try:
    from hongikjiki.modules.tagging.tag_schema import TagSchema
    from hongikjiki.modules.tagging.tag_extractor import TagExtractor
    USE_TAG_EXTRACTOR = True
except ImportError:
    USE_TAG_EXTRACTOR = False
    logger.warning("Tag extractor modules not found, continuing without tag extraction")

def initialize_chatbot():
    """챗봇 초기화 및 데이터 로드"""
    # 환경변수 로드
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'openai')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    DATA_DIR = Path(os.getenv('DATA_DIR', ROOT_DIR / "data" / "jungbub_teachings"))
    CHATBOT_NAME = os.getenv('CHATBOT_NAME', 'Hongik-Jiki')
    DEVELOPER_NAME = os.getenv('DEVELOPER_NAME', '개발연구원 조성우')
    
    # Check if API key is set
    if not OPENAI_API_KEY:
        print("오류: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("OpenAI API 키를 설정한 후 다시 시도하세요.")
        sys.exit(1)
    
    # 로깅 설정
    setup_logging()
    logger.info(f"{CHATBOT_NAME} 초기화 시작 (개발자: {DEVELOPER_NAME})")
    
    # 텍스트 프로세서 생성
    text_processor = DocumentProcessor()
    
    # LLM 초기화
    llm = get_llm("openai", api_key=OPENAI_API_KEY)
    
    # 벡터 스토어 생성
    embeddings = get_embeddings(model_name=EMBEDDING_MODEL)
    vector_store = ChromaVectorStore(
        embeddings=embeddings,
        persist_directory="./chroma_db"
    )
    
    # 태그 추출기 초기화 (선택적)
    tag_extractor = None
    if USE_TAG_EXTRACTOR:
        try:
            tag_schema_path = ROOT_DIR / "data" / "config" / "tag_schema.yaml"
            tag_schema = TagSchema(str(tag_schema_path))
            tag_extractor = TagExtractor(tag_schema)
            logger.info("태그 추출기 초기화 완료")
        except Exception as e:
            logger.warning(f"태그 추출기 초기화 실패: {e}")
    
    # 데이터베이스에 문서 수 확인
    collection_info = vector_store.count()
    logger.info(f"현재 데이터베이스 문서 수: {collection_info}")
    
    # 문서가 없으면 문서 로드 및 처리
    if collection_info == 0:
        logger.info("데이터베이스가 비어 있습니다. 문서 로드 및 처리를 시작합니다.")
        
        # Check if DATA_DIR exists
        if not DATA_DIR.is_dir():
            logger.warning(f"{DATA_DIR} 폴더가 존재하지 않습니다.")
            print(f"오류: {DATA_DIR} 폴더를 찾을 수 없습니다.")
            sys.exit(1)
        
        # 문서 로드
        documents = text_processor.process_directory(str(DATA_DIR))
        
        if not documents:
            logger.warning(f"{DATA_DIR} 폴더에 문서가 없습니다.")
            print(f"오류: {DATA_DIR} 폴더에 정법 문서를 찾을 수 없습니다.")
            print("정법 문서를 data/jungbub_teachings 폴더에 추가한 후 다시 시도하세요.")
            sys.exit(1)
        
        # 벡터 데이터베이스에 추가
        vector_store.add_documents(documents)
    
    # 챗봇 생성 - 필요한 모든 파라미터 전달
    chatbot_params = {
        "llm": llm,
        "vector_store": vector_store
    }
    
    # 태그 추출기가 있으면 추가
    if tag_extractor:
        chatbot_params["tag_extractor"] = tag_extractor
    
    chatbot = HongikJikiChatbot(**chatbot_params)
    logger.info("챗봇 초기화 완료")
    
    return chatbot

def run_cli():
    """명령줄 인터페이스로 챗봇 실행"""
    print_welcome_message()
    
    # 챗봇 초기화
    chatbot = initialize_chatbot()
    
    print("\nHongik-Jiki가 준비되었습니다.")
    print("대화를 종료하려면 'q', 'quit', 또는 'exit'를 입력하세요.")
    print("관련 질문을 보려면 'related'를 입력하세요.")
    print("-" * 60)
    
    # 대화 기록 유지
    conversation_history = []
    
    while True:
        user_input = input("\n질문: ")
        
        if not user_input.strip():
            continue
            
        if user_input.lower() in ['q', 'quit', 'exit', '종료']:
            print("\n대화를 종료합니다. 감사합니다.")
            break
            
        if user_input.lower() == 'related':
            related_questions = chatbot.get_related_questions()
            if related_questions:
                print("\n관련 질문:")
                for i, q in enumerate(related_questions, 1):
                    print(f"{i}. {q.get('question', '')}")
                print("-" * 60)
            else:
                print("\n관련 질문이 없습니다.")
            continue
        
        # 응답 생성
        try:
            # answer_question returns a dict with 'answer' and 'file' keys
            response_data = chatbot.answer_question(user_input, conversation_history)
            
            # Extract the answer from the response data
            if isinstance(response_data, dict):
                response = response_data.get("answer", "응답을 찾을 수 없습니다.")
            else:
                response = str(response_data)
            
            # 대화 기록 업데이트 (수동으로 추적)
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": response})
            
            # 대화 기록을 적절한 크기로 유지 (너무 길면 오래된 메시지 제거)
            if len(conversation_history) > 10:
                conversation_history = conversation_history[-10:]
                
        except Exception as e:
            logger.error(f"응답 생성 오류: {e}")
            response = f"죄송합니다. 응답 생성 중 오류가 발생했습니다: {str(e)}"
        
        # 응답 출력
        print("\n답변:")
        print(response)
        print("-" * 60)
        
        # 관련 질문 존재 시 안내
        if hasattr(chatbot, 'get_related_questions') and chatbot.get_related_questions():
            print("\n관련 질문을 보려면 'related'를 입력하세요.")

def main():
    """CLI 모드 메인 함수"""
    run_cli()

if __name__ == "__main__":
    main()