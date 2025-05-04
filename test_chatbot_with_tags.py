"""
태그 기능이 활성화된 챗봇 테스트
"""
import logging
logging.basicConfig(level=logging.INFO)

from hongikjiki.chatbot import HongikJikiChatBot

def main():
    """태그 기능 테스트"""
    print("태그 기능이 활성화된 챗봇 테스트 시작...")
    
    # 챗봇 초기화 (태그 기능 활성화)
    chatbot = HongikJikiChatBot(use_tags=True)
    print(f"태그 기능 활성화 여부: {chatbot.use_tags}")
    
    # 테스트 질문
    test_queries = [
        "홍익인간이란 무엇인가요?",
        "자유의지에 대해 설명해주세요"
    ]
    
    # 각 질문에 대한 응답 생성
    for query in test_queries:
        print(f"\n질문: {query}")
        response = chatbot.chat(query)
        print(f"응답:\n{response}")
    
    print("\n태그 기능 테스트 완료")

if __name__ == "__main__":
    main()
