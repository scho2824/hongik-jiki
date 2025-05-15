import pytest
from hongikjiki.core.chatbot import HongikJikiChatbot
from hongikjiki.modules.vector_store.chroma_store import ChromaVectorStore
from hongikjiki.langchain_integration.llm import get_llm

def test_chatbot_full_flow(tmp_path):
    # 1. 임베딩 및 벡터 저장소 초기화
    vector_store = ChromaVectorStore(persist_directory=str(tmp_path))
    texts = ["정법은 인간과 우주의 본질을 깨닫는 길입니다."]
    metadatas = [{"source": "test_doc", "source_id": "doc1"}]
    vector_store.add_texts(texts, metadatas)

    # 2. LLM 연결
    llm = get_llm("openai")  # OpenAI key 필요, 또는 mock 사용 가능

    # 3. 챗봇 인스턴스 생성
    chatbot = HongikJikiChatbot(llm=llm, vector_store=vector_store)

    # 4. 질문 → 답변 흐름 실행
    query = "우주의 본질은 무엇인가요?"
    response = chatbot.answer_question(query, history=[])

    # 5. 결과 검증
    assert isinstance(response, (str, dict))
    if isinstance(response, str):
        assert len(response) > 0
        