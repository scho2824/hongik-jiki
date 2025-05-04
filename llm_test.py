# llm_test.py
import os
from dotenv import load_dotenv
import logging

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LLM_Test")

# OpenAI 패키지 명시적 임포트
try:
    import openai
    has_openai = True
except ImportError:
    logger.error("openai 패키지를 찾을 수 없습니다. pip install openai로 설치하세요.")
    has_openai = False

# hongikjiki 패키지 임포트
try:
    from hongikjiki.langchain_integration.llm import OpenAILLM, get_llm
    has_hongikjiki = True
except ImportError:
    logger.error("hongikjiki.langchain_integration.llm을 임포트할 수 없습니다.")
    has_hongikjiki = False

def test_openai_direct():
    """직접 OpenAI API 호출 테스트"""
    if not has_openai:
        logger.error("openai 패키지가 설치되지 않아 테스트를 건너뜁니다.")
        return None
    
    logger.info("직접 OpenAI API 호출 테스트 시작")
    
    try:
        # OpenAI 클라이언트 초기화
        client = openai.OpenAI()
        
        # 테스트 프롬프트
        test_prompt = "홍익인간이란 무엇인가요? 간단히 한 문장으로 설명해주세요."
        
        # API 호출
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": test_prompt}],
            temperature=0.7,
            max_tokens=100
        )
        
        # 응답 출력
        result = response.choices[0].message.content
        logger.info(f"API 직접 호출 결과: {result}")
        return result
        
    except Exception as e:
        logger.error(f"API 직접 호출 오류: {e}")
        return None

def examine_llm_class():
    """OpenAILLM 클래스의 메서드 검사"""
    if not has_hongikjiki:
        logger.error("hongikjiki 패키지가 임포트되지 않아 테스트를 건너뜁니다.")
        return
    
    try:
        # OpenAILLM 인스턴스 생성
        llm = OpenAILLM(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=100
        )
        
        # 클래스의 메서드 출력
        methods = [method for method in dir(llm) if not method.startswith('_')]
        logger.info(f"OpenAILLM 클래스의 메서드: {methods}")
        
        # 생성 관련 메서드 찾기
        generation_methods = [m for m in methods if 'generate' in m or 'create' in m or 'complete' in m]
        if generation_methods:
            logger.info(f"생성 관련 메서드: {generation_methods}")
        else:
            logger.warning("생성 관련 메서드를 찾을 수 없습니다.")
            
        return methods
    except Exception as e:
        logger.error(f"클래스 검사 오류: {e}")
        return None

def test_llm_class_with_detected_method():
    """감지된 메서드로 OpenAILLM 테스트"""
    if not has_hongikjiki:
        logger.error("hongikjiki 패키지가 임포트되지 않아 테스트를 건너뜁니다.")
        return None
    
    try:
        # OpenAILLM 인스턴스 생성
        llm = OpenAILLM(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=100
        )
        
        # 메서드 리스트 가져오기
        methods = [method for method in dir(llm) if not method.startswith('_')]
        
        # 가능한 생성 메서드 시도
        possible_methods = ['generate_text', 'create_completion', 'complete', 'generate', '__call__']
        
        for method_name in possible_methods:
            if method_name in methods:
                logger.info(f"메서드 '{method_name}' 테스트 중...")
                method = getattr(llm, method_name)
                
                try:
                    result = method("정법이란 무엇인가요?")
                    logger.info(f"메서드 '{method_name}' 테스트 성공: {result}")
                    return result
                except Exception as e:
                    logger.error(f"메서드 '{method_name}' 테스트 오류: {e}")
        
        logger.warning("사용 가능한 생성 메서드를 찾을 수 없습니다.")
        return None
    except Exception as e:
        logger.error(f"감지된 메서드 테스트 오류: {e}")
        return None

if __name__ == "__main__":
    print("LLM 응답 생성 테스트 시작...\n")
    
    # API 키 확인
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        exit(1)
    
    # OpenAI 직접 호출 테스트
    print("\n===== 직접 OpenAI API 호출 테스트 =====")
    direct_result = test_openai_direct()
    if direct_result:
        print(f"✅ 결과: {direct_result}")
    else:
        print("❌ 테스트 실패")
    
    # OpenAILLM 클래스 검사
    print("\n===== OpenAILLM 클래스 검사 =====")
    methods = examine_llm_class()
    if methods:
        print(f"✅ 메서드 목록: {', '.join(methods)}")
    else:
        print("❌ 클래스 검사 실패")
    
    # 감지된 메서드로 테스트
    print("\n===== 감지된 메서드로 OpenAILLM 테스트 =====")
    test_result = test_llm_class_with_detected_method()
    if test_result:
        print(f"✅ 결과: {test_result}")
    else:
        print("❌ 테스트 실패")
    
    print("\n모든 테스트 완료")