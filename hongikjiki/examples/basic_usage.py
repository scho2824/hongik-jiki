"""
홍익지기 챗봇 기본 사용 예제
"""

import os
import logging
import sys
from dotenv import load_dotenv

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("HongikJikiChatBot")

# 현재 디렉토리를 모듈 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# .env 파일 로드 (API 키 등)
load_dotenv()

from hongikjiki.chatbot import HongikJikiChatBot

def main():
    """메인 함수"""
    # 환경 변수에서 API 키 가져오기
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        logger.error("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        logger.error("API 키를 .env 파일에 설정하거나 환경 변수로 설정하세요.")
        return
    
    # 챗봇 생성
    chatbot = HongikJikiChatBot(
        persist_directory="./data",
        embedding_type="openai",   # 'huggingface' 또는 'openai'
        llm_type="openai",         # 'openai' 또는 'clova'
        collection_name="hongikjiki_documents",
        max_history=10,
        chunk_size=1000,
        chunk_overlap=200,
        embedding_kwargs={
            "model": "text-embedding-3-small",
            "api_key": openai_api_key
        },
        llm_kwargs={
            "model": "gpt-3.5-turbo",
            "api_key": openai_api_key,
            "temperature": 0.7
        }
    )
    
    # 벡터 저장소에 문서 수 확인
    doc_count = chatbot.get_document_count()
    logger.info(f"현재 벡터 저장소에 {doc_count}개의 문서가 있습니다.")
    
    # 문서가 없으면 샘플 문서 로드
    if doc_count == 0:
        sample_dir = "./sample_docs"
        if os.path.exists(sample_dir):
            logger.info(f"{sample_dir} 디렉토리에서 샘플 문서 로드 중...")
            loaded_count = chatbot.load_documents(sample_dir)
            logger.info(f"총 {loaded_count}개의 문서 청크를 로드했습니다.")
        else:
            logger.warning(f"{sample_dir} 디렉토리가 존재하지 않습니다. 샘플 문서를 로드할 수 없습니다.")
    
    # 대화 루프
    print("\n=== 홍익지기 챗봇과 대화를 시작합니다 ===")
    print("종료하려면 'exit' 또는 'quit'를 입력하세요.")
    print("대화 기록을 초기화하려면 'clear'를 입력하세요.")
    
    while True:
        user_input = input("\n질문: ").strip()
        
        if user_input.lower() in ['exit', 'quit']:
            print("대화를 종료합니다.")
            break
        
        if user_input.lower() == 'clear':
            chatbot.clear_history()
            print("대화 기록이 초기화되었습니다.")
            continue
        
        if not user_input:
            continue
        
        # 챗봇 응답 생성
        response = chatbot.chat(user_input)
        print(f"\n답변: {response}")

if __name__ == "__main__":
    main()