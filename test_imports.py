# test_imports.py
print("핵심 모듈 임포트 테스트 시작...")

# 1. core 모듈 테스트
try:
    from hongikjiki.core.chatbot import HongikJikiChatbot
    print("✅ core.chatbot 임포트 성공")
except Exception as e:
    print(f"❌ core.chatbot 임포트 실패: {e}")

# 2. langchain_integration 모듈 테스트
try:
    from hongikjiki.langchain_integration.llm import get_llm
    print("✅ langchain_integration.llm 임포트 성공")
except Exception as e:
    print(f"❌ langchain_integration.llm 임포트 실패: {e}")

# 3. modules 주요 서브모듈 테스트
try:
    from hongikjiki.modules.text_processing.document_processor import DocumentProcessor
    print("✅ modules.text_processing.document_processor 임포트 성공")
except Exception as e:
    print(f"❌ modules.text_processing.document_processor 임포트 실패: {e}")

try:
    from hongikjiki.modules.vector_store.chroma_store import ChromaVectorStore
    print("✅ modules.vector_store.chroma_store 임포트 성공")
except Exception as e:
    print(f"❌ modules.vector_store.chroma_store 임포트 실패: {e}")

print("임포트 테스트 완료")