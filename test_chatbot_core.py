import pytest
from hongikjiki.core.chatbot import HongikJikiChatbot
from hongikjiki.langchain_integration.llm import get_llm

class MockVectorStore:
    def search(self, query, k=3):
        return [{"content": "테스트 내용", "metadata": {}}]

    def count(self):
        return 1

@pytest.fixture
def chatbot():
    llm = get_llm("openai")
    return HongikJikiChatbot(llm, MockVectorStore())

def test_answer_question(chatbot):
    response = chatbot.answer_question("테스트 질문", history=[])
    assert isinstance(response, (str, dict)), f"Expected str or dict, got {type(response)}"