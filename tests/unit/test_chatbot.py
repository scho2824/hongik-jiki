import pytest
from unittest.mock import MagicMock, patch
from hongikjiki.core.chatbot import HongikJikiChatbot

def test_chatbot_answer_question(mock_llm, mock_vector_store):
    """챗봇 질문 응답 테스트"""
    # 챗봇 인스턴스 생성
    chatbot = HongikJikiChatbot(mock_llm, mock_vector_store)
    
    # 질문 처리
    result = chatbot.answer_question("테스트 질문", [])
    
    # 검증
    assert isinstance(result, dict)
    assert "answer" in result
    assert "file" in result
    assert "모의 응답입니다." in result["answer"]
    
    # 메서드 호출 검증
    mock_vector_store.search.assert_called_once()
    mock_llm.generate.assert_called_once()

@patch('hongikjiki.core.chatbot.generate_related_questions')
def test_chatbot_related_questions(mock_generate, mock_llm, mock_vector_store):
    """관련 질문 생성 테스트"""
    # 관련 질문 모의 설정
    mock_generate.return_value = [
        {"question": "관련 질문 1", "insight": "테스트"}
    ]
    
    # 챗봇 인스턴스 생성
    chatbot = HongikJikiChatbot(mock_llm, mock_vector_store)

    # vector_store.count() 결과를 0으로 설정하여 간단 모드 실행 보장
    mock_vector_store.count.return_value = 0
    
    # 질문 처리
    chatbot.answer_question("테스트 질문", [])

    # Mock 호출 확인
    mock_generate.assert_called_once()
    
    # 관련 질문 확인
    related = chatbot.get_related_questions()
    assert len(related) == 1
    assert related[0]["question"] == "관련 질문 1"
