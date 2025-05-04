"""
홍익지기 챗봇 고급 사용 예제
문서 관리, 검색, 컨텍스트 추가 등의 고급 기능 데모
"""

import os
import logging
import sys
import json
from typing import List, Dict, Any
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
from hongikjiki.text_processing import DocumentProcessor

def print_menu():
    """메뉴 출력"""
    print("\n=== 홍익지기 챗봇 고급 기능 메뉴 ===")
    print("1. 대화 모드")
    print("2. 문서 로드")
    print("3. 문서 검색")
    print("4. 문서 통계")
    print("5. 벡터 저장소 초기화")
    print("6. 종료")
    
    return input("\n메뉴 선택 (1-6): ").strip()

def chat_mode(chatbot: HongikJikiChatBot):
    """대화 모드"""
    print("\n=== 대화 모드 ===")
    print("대화를 종료하려면 'back'을 입력하세요.")
    print("대화 기록을 초기화하려면 'clear'를 입력하세요.")
    
    while True:
        user_input = input("\n질문: ").strip()
        
        if user_input.lower() == 'back':
            print("대화 모드를 종료합니다.")
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

def load_documents(chatbot: HongikJikiChatBot):
    """문서 로드 기능"""
    print("\n=== 문서 로드 ===")
    print("1. 디렉토리에서 로드")
    print("2. 단일 파일 로드")
    print("3. 뒤로 가기")
    
    choice = input("\n선택 (1-3): ").strip()
    
    if choice == '1':
        directory = input("로드할 문서 디렉토리 경로: ").strip()
        if os.path.exists(directory) and os.path.isdir(directory):
            loaded_count = chatbot.load_documents(directory)
            print(f"총 {loaded_count}개의 문서 청크를 로드했습니다.")
        else:
            print(f"오류: {directory} 디렉토리가 존재하지 않습니다.")
    
    elif choice == '2':
        file_path = input("로드할 문서 파일 경로: ").strip()
        if os.path.exists(file_path) and os.path.isfile(file_path):
            loaded_count = chatbot.load_document(file_path)
            print(f"총 {loaded_count}개의 문서 청크를 로드했습니다.")
        else:
            print(f"오류: {file_path} 파일이 존재하지 않습니다.")

def search_documents(chatbot: HongikJikiChatBot):
    """문서 검색 기능"""
    print("\n=== 문서 검색 ===")
    print("검색을 종료하려면 'back'을 입력하세요.")
    
    while True:
        query = input("\n검색어: ").strip()
        
        if query.lower() == 'back':
            print("검색을 종료합니다.")
            break
        
        if not query:
            continue
        
        # 검색 결과 수 설정
        try:
            k = int(input("검색 결과 수 (기본 4): ").strip() or "4")
        except ValueError:
            k = 4
        
        # 검색 실행
        results = chatbot.search_documents(query, k=k)
        
        if not results:
            print("검색 결과가 없습니다.")
            continue
        
        # 결과 출력
        print(f"\n총 {len(results)}개의 검색 결과:")
        for i, doc in enumerate(results):
            print(f"\n--- 결과 {i+1} (유사도: {doc['score']:.4f}) ---")
            
            # 메타데이터 출력
            metadata = doc.get('metadata', {})
            lecture_num = metadata.get('lecture_number', '?')
            title = metadata.get('title', '제목 없음')
            content_type = metadata.get('content_type', '알 수 없음')
            
            print(f"강의: {lecture_num}강, 제목: {title}, 유형: {content_type}")
            
            # 내용 미리보기 (100자 제한)
            content = doc.get('content', '')
            preview = content[:100] + "..." if len(content) > 100 else content
            print(f"내용 미리보기: {preview}")
        
        # 선택적 대화에 활용
        use_in_chat = input("\n이 검색 결과를 바탕으로 대화하시겠습니까? (y/n): ").strip().lower()
        
        if use_in_chat == 'y':
            question = input("\n질문: ").strip()
            
            if not question:
                continue
            
            # 검색 결과 컨텍스트 구성
            context = ""
            for i, doc in enumerate(results):
                metadata = doc.get('metadata', {})
                lecture_num = metadata.get('lecture_number', '?')
                title = metadata.get('title', '제목 없음')
                content = doc.get('content', '')
                
                context += f"[문서 {i+1}] 강의 {lecture_num}강 - {title}\n{content}\n\n"
            
            # 컨텍스트 기반 응답 생성
            response = chatbot.chat_with_context(question, context)
            print(f"\n답변: {response}")

def document_stats(chatbot: HongikJikiChatBot):
    """문서 통계 정보"""
    print("\n=== 문서 통계 ===")
    
    # 총 문서 수
    doc_count = chatbot.get_document_count()
    print(f"벡터 저장소에 총 {doc_count}개의 문서 청크가 있습니다.")
    
    # 추가 정보가 필요하면 여기에 구현

def reset_vector_store(chatbot: HongikJikiChatBot):
    """벡터 저장소 초기화"""
    print("\n=== 벡터 저장소 초기화 ===")
    confirm = input("정말로 벡터 저장소를 초기화하시겠습니까? 모든 문서가 삭제됩니다! (y/n): ").strip().lower()
    
    if confirm == 'y':
        success = chatbot.reset_vector_store()
        if success:
            print("벡터 저장소가 초기화되었습니다.")
        else:
            print("벡터 저장소 초기화 중 오류가 발생했습니다.")
    else:
        print("벡터 저장소 초기화가 취소되었습니다.")

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
    
    print("=== 홍익지기 챗봇 고급 기능 데모 ===")
    
    # 메인 메뉴 루프
    while True:
        choice = print_menu()
        
        if choice == '1':
            chat_mode(chatbot)
        elif choice == '2':
            load_documents(chatbot)
        elif choice == '3':
            search_documents(chatbot)
        elif choice == '4':
            document_stats(chatbot)
        elif choice == '5':
            reset_vector_store(chatbot)
        elif choice == '6':
            print("프로그램을 종료합니다.")
            break
        else:
            print("잘못된 선택입니다. 다시 시도하세요.")

if __name__ == "__main__":
    main()
    