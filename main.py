import os
import sys
import logging
from dotenv import load_dotenv

# 모듈 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 유틸리티 모듈 임포트
from hongikjiki.utils.file_utils import ensure_dir
from hongikjiki.core.chatbot import HongikJikiChatbot  # 수정된 임포트
from hongikjiki.modules.vector_store.chroma_store import ChromaVectorStore  # JungbubVectorStore 대신 ChromaVectorStore 사용

# 환경변수 로드
load_dotenv()
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')
DATA_DIR = os.getenv('DATA_DIR')
CHATBOT_NAME = os.getenv('CHATBOT_NAME', 'Hongik-Jiki')
DEVELOPER_NAME = os.getenv('DEVELOPER_NAME', '조성우')

# 로깅 설정
def setup_logging():
    """로깅 설정"""
    logger = logging.getLogger("HongikJikiChatBot")
    logger.setLevel(logging.INFO)
    
    # 기존 핸들러 제거
    if logger.handlers:
        logger.handlers.clear()
    
    # 파일 핸들러 추가
    log_dir = "logs"
    ensure_dir(log_dir)
    file_handler = logging.FileHandler(os.path.join(log_dir, "chatbot.log"))
    file_handler.setLevel(logging.INFO)
    
    # 콘솔 핸들러 추가
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 포맷터 설정
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 핸들러 추가
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def print_welcome_message():
    """시작 메시지 출력"""
    print("=" * 60)
    print(f"{CHATBOT_NAME} - 개발자: {DEVELOPER_NAME}")
    print("천공 스승님의 가르침에 기반한 인공지능 비서")
    print("정법은 통찰로 자신과 세상의 본질을 깨닫고, 역설로 우리의 상식을 뒤집어")
    print("홍익인간의 삶을 실현하는 데 목적이 있습니다.")
    print("=" * 60)

def initialize_chatbot():
    """챗봇 초기화 및 데이터 로드"""
    # 로깅 설정
    logger = setup_logging()
    logger.info(f"{CHATBOT_NAME} 초기화 시작 (개발자: {DEVELOPER_NAME})")
    
    # 벡터 스토어 생성
    vector_store = ChromaVectorStore(
        collection_name="hongikjiki_documents",
        persist_directory="./chroma_db"
    )
    
    # 데이터베이스에 문서 수 확인
    collection_count = vector_store.count()
    logger.info(f"현재 데이터베이스 문서 수: {collection_count}")
    
    # 문서가 없으면 오류 출력 및 종료
    if collection_count == 0:
        logger.warning("데이터베이스가 비어 있습니다. 문서를 먼저 전처리하세요.")
        print("데이터베이스가 비어 있습니다. 먼저 문서를 처리해 주세요.")
        sys.exit(1)
    
    # LLM 생성
    from hongikjiki.langchain_integration.llm import get_llm
    llm = get_llm("openai")
    
    # 챗봇 생성
    chatbot = HongikJikiChatbot(llm, vector_store)
    
    return chatbot

def run_cli():
    """명령줄 인터페이스로 챗봇 실행"""
    print_welcome_message()
    
    # 챗봇 초기화
    chatbot = initialize_chatbot()
    
    print(f"\n{CHATBOT_NAME}이 준비되었습니다.")
    print("대화를 종료하려면 'q', 'quit', 또는 'exit'를 입력하세요.")
    print("-" * 60)
    
    history = []
    
    while True:
        user_input = input("\n질문: ")
        
        if user_input.lower() in ['q', 'quit', 'exit', '종료']:
            print("\n대화를 종료합니다. 감사합니다.")
            break
        
        if not user_input.strip():
            continue
        
        # 응답 생성
        response = chatbot.answer_question(user_input, history)
        
        # 응답 출력 (응답 형식에 따라 조정)
        if isinstance(response, dict):
            print("\n답변:")
            print(response.get("answer", "응답을 생성할 수 없습니다."))
        else:
            print("\n답변:")
            print(response)
        print("-" * 60)
        
        # 대화 기록 업데이트
        history.append({"role": "user", "content": user_input})
        if isinstance(response, dict):
            history.append({"role": "assistant", "content": response.get("answer", "")})
        else:
            history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    run_cli()