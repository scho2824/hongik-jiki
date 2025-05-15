# test_chain.py
import logging
from dotenv import load_dotenv
load_dotenv()  # API 키 불러오기

from hongikjiki.langchain_integration.llm import get_llm
from hongikjiki.modules.vector_store.chroma_store import ChromaVectorStore
from hongikjiki.modules.vector_store.embeddings import get_embeddings
from hongikjiki.langchain_integration.chain import get_chatbot_chain

# 로깅 설정
logging.basicConfig(level=logging.INFO)

def test_chain():
    # LLM 모델 로드
    llm = get_llm("openai")
    
    # 임베딩 및 벡터 저장소 초기화
    embeddings = get_embeddings("openai")
    vector_store = ChromaVectorStore(
        collection_name="hongikjiki_jungbub",
        persist_directory="./data/vector_store",
        embeddings=embeddings
    )
    
    # 체인 생성
    chain = get_chatbot_chain(llm, vector_store)
    
    # 테스트 질문
    result = chain.run("정법이란 무엇인가요?")
    
    print(f"답변 타입: {type(result)}")
    if isinstance(result, dict):
        print(f"답변: {result.get('answer', '')[:100]}...")
    else:
        print(f"답변: {result[:100]}...")
    
    return True

if __name__ == "__main__":
    test_chain()