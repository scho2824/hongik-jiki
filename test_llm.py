# test_llm.py
from hongikjiki.langchain_integration.llm import get_llm

# LLM 초기화 테스트
try:
    llm = get_llm("openai")
    print("✅ LLM 초기화 성공")
    
    # 기본 생성 테스트 (API 키가 설정된 경우에만)
    try:
        response = llm.generate("안녕하세요, 테스트 메시지입니다.")
        print(f"✅ LLM 응답 생성 성공: {response[:50]}...")
    except Exception as e:
        print(f"⚠️ LLM 응답 생성 테스트 건너뜀 (API 키 필요): {e}")
    
except Exception as e:
    print(f"❌ LLM 테스트 실패: {e}")
    