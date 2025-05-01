"""
Hongik-Jiki 명령줄 인터페이스
"""

import os
import sys
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# 패키지 내 모듈 임포트
from .utils import setup_logging, print_welcome_message
from .text_processor import JungbubTextProcessor
from .vector_store import JungbubVectorStore
from .chatbot import HongikJikiBot

def initialize_chatbot():
    """챗봇 초기화 및 데이터 로드"""
    # 환경변수 로드
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')
    DATA_DIR = os.getenv('DATA_DIR')
    CHATBOT_NAME = os.getenv('CHATBOT_NAME', 'Hongik-Jiki')
    DEVELOPER_NAME = os.getenv('DEVELOPER_NAME', '조성우')
    
    # 로깅 설정
    logger = setup_logging()
    logger.info(f"{CHATBOT_NAME} 초기화 시작 (개발자: {DEVELOPER_NAME})")
    
    # 텍스트 프로세서 생성
    text_processor = JungbubTextProcessor()
    
    # 벡터 스토어 생성
    vector_store = JungbubVectorStore(
        embedding_model_name=EMBEDDING_MODEL,
        persist_directory="./chroma_db"
    )
    
    # 데이터베이스에 문서 수 확인
    collection_info = vector_store.collection.count()
    logger.info(f"현재 데이터베이스 문서 수: {collection_info}")
    
    # 문서가 없으면 문서 로드 및 처리
    if collection_info == 0:
        logger.info("데이터베이스가 비어 있습니다. 문서 로드 및 처리를 시작합니다.")
        
        # 문서 로드
        documents = text_processor.load_documents(DATA_DIR)
        
        if not documents:
            logger.warning(f"{DATA_DIR} 폴더에 문서가 없습니다.")
            print(f"오류: {DATA_DIR} 폴더에 정법 문서를 찾을 수 없습니다.")
            print("정법 문서를 data/jungbub_teachings 폴더에 추가한 후 다시 시도하세요.")
            sys.exit(1)
        
        # 문서 분할
        chunks = text_processor.split_documents(documents)
        
        # 벡터 데이터베이스에 추가
        vector_store.add_documents(chunks)
    
    # 챗봇 생성
    chatbot = HongikJikiBot(vector_store)
    
    return chatbot

def run_cli():
    """명령줄 인터페이스로 챗봇 실행"""
    print_welcome_message()
    
    # 챗봇 초기화
    chatbot = initialize_chatbot()
    
    print("\nHongik-Jiki가 준비되었습니다.")
    print("대화를 종료하려면 'q', 'quit', 또는 'exit'를 입력하세요.")
    print("-" * 60)
    
    while True:
        user_input = input("\n질문: ")
        
        if user_input.lower() in ['q', 'quit', 'exit', '종료']:
            print("\n대화를 종료합니다. 감사합니다.")
            # 대화 이력 저장
            chatbot.save_conversation()
            break
        
        if not user_input.strip():
            continue
        
        # 응답 생성
        response = chatbot.get_response(user_input)
        
        # 응답 출력
        print("\n답변:")
        print(response)
        print("-" * 60)

def main():
    """CLI 모드 메인 함수"""
    run_cli()

if __name__ == "__main__":
    main()