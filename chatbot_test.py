# chatbot_test.py
import os
import logging
from dotenv import load_dotenv
from hongikjiki.chatbot import HongikJikiChatBot

# .env 파일 로드 (API 키 등)
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("HongikJikiChatBot")

def test_chatbot():
    """챗봇 기본 기능 테스트"""
    print("\n=== 챗봇 기본 기능 테스트 ===")
    
    try:
        # 챗봇 초기화
        chatbot = HongikJikiChatBot(
            persist_directory="./data",
            embedding_type="openai",  # OpenAI 임베딩 사용
            llm_type="openai",        # OpenAI LLM 사용
            collection_name="hongikjiki_jungbub",  # use the loaded collection name
            embedding_kwargs={
                "model": "text-embedding-3-small"
            },
            llm_kwargs={
                "model": "gpt-3.5-turbo",
                "temperature": 0.7
            }
        )
        
        # 문서 개수 확인
        doc_count = chatbot.get_document_count()
        print(f"현재 벡터 저장소에 {doc_count}개의 문서가 있습니다.")
        
        # 간단한 검색 테스트
        print("\n=== 검색 테스트 ===")
        results = chatbot.search_documents("홍익인간", k=2)
        print(f"검색 결과 수: {len(results)}")
        for i, result in enumerate(results):
            print(f"\n결과 {i+1} (유사도: {result['score']:.4f})")
            print(f"내용: {result['content']}")
        
        # 대화 테스트
        print("\n=== 대화 테스트 ===")
        questions = [
            "홍익인간은 무엇을 의미하나요?",
            "정법의 핵심 가르침은 무엇인가요?",
            "자연의 법칙을 따르는 것이 왜 중요한가요?"
        ]
        
        for question in questions:
            print(f"\n질문: {question}")
            response = chatbot.chat(question)
            print(f"답변: {response}")
        
        return True
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_chatbot()
    print(f"\n테스트 결과: {'성공' if success else '실패'}")