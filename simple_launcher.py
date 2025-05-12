#!/usr/bin/env python3
"""
홍익지기 챗봇 간단 런처
"""
import os
import subprocess
import sys
from hongikjiki.utils import load_dotenv, setup_logging

def main():
    # 로깅 설정
    logger = setup_logging()
    logger.info("홍익지기 챗봇 시작")
    
    # 환경 변수 로드
    load_dotenv()
    logger.info("환경 변수 로드 완료")
    
    # API 키 확인
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("오류: OPENAI_API_KEY가 설정되지 않았습니다.")
        print("다음 방법 중 하나로 API 키를 설정하세요:")
        print("1. export OPENAI_API_KEY=your_api_key")
        print("2. .env 파일에 OPENAI_API_KEY=your_api_key 추가")
        logger.error("API 키가 설정되지 않았습니다")
        return 1
    
    logger.info(f"API 키 확인: {api_key[:5]}...{api_key[-5:]}")
    print(f"API 키 확인: {api_key[:5]}...{api_key[-5:]}")
    
    # 서브프로세스로 Gradio 앱 실행
    print("Gradio 앱을 시작합니다...")
    print("웹 브라우저에서 URL을 열어 앱에 접속하세요.")
    print("앱을 종료하려면 Ctrl+C를 누르세요.")
    
    # 실행할 Python 코드 만들기
    launcher_code = """
import os
import gradio as gr
from hongikjiki.utils import load_dotenv
from hongikjiki.vector_store import load_vector_store
from hongikjiki.vector_store.chroma_store import ChromaVectorStore
from hongikjiki.langchain_integration.llm import get_llm
from hongikjiki.langchain_integration.chain import get_chatbot_chain

# 환경 변수 로드
load_dotenv()

# 간단 인터페이스 생성
def answer_question(question):
    try:
        # LLM 초기화
        llm = get_llm(llm_type="openai", model="gpt-4o")
        
        # 벡터 스토어 로드 시도
        try:
            collection, embeddings = load_vector_store(
                persist_directory="data/vector_store",
                collection_name="hongikjiki_documents",
                embedding_type="openai",
                reset_if_error=True
            )
            vector_store = ChromaVectorStore(collection)
            chatbot = get_chatbot_chain(llm=llm, vector_store=vector_store)
            
            # 벡터 스토어 기반 응답
            response = chatbot.run(question)
            if isinstance(response, str):
                return f"💬 {response}"
            answer = f"💬 {response.get('answer', '')}"
            lecture_id = response.get("lecture_id", "")
            lecture_title = response.get("lecture_title", "")
            
            if lecture_id or lecture_title:
                source = f"\\n\\n🔗 출처:"
                if lecture_title:
                    source += f" 「{lecture_title}」"
                if lecture_id:
                    source += f" (강의 번호: {lecture_id})"
                answer += source
                
            source_summary = response.get("source_summary", "")
            if source_summary:
                answer += f"\\n\\n📝 강의 요약:\\n{source_summary}"
                
            return answer
            
        except Exception as e:
            print(f"벡터 스토어 오류, 간단 모드 사용: {e}")
            # 벡터 스토어 없이 간단 응답
            prompt = f"다음 질문에 대해 홍익인간과 정법의 관점에서 답변해주세요. 가능한 상세하게 답변하고, 답변은 한국어로 작성해주세요: {question}"
            answer = llm.generate(prompt)
            return f"💬 {answer}\\n\\n(간단 모드: 벡터 스토어 없음)"
    except Exception as e:
        return f"오류 발생: {str(e)}"

# Gradio 인터페이스 생성
demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(placeholder="질문을 입력하세요...", label="질문"),
    outputs=gr.Textbox(label="답변"),
    title="홍익지기 챗봇",
    description="정법 강의 기반 챗봇입니다. 삶의 방향, 감정, 사회, 영성 등에 대한 통찰을 얻어보세요.",
    examples=[
        "영혼과 육신의 관계는?",
        "현대 사회가 무너지는 이유는 무엇인가요?",
        "청년이 사회에서 가져야 할 태도는?",
        "운명은 정해져 있나요?",
        "수행이란 정확히 무엇인가요?",
        "정법은 불교나 유교와 무엇이 다른가요?"
    ]
)

# 앱 시작
demo.launch()
"""
    
    # 임시 파일에 코드 저장
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp:
        temp_path = temp.name
        temp.write(launcher_code.encode('utf-8'))
    
    try:
        # 서브프로세스로 앱 실행
        subprocess.run([sys.executable, temp_path], check=True)
        return 0
    except KeyboardInterrupt:
        print("\n사용자에 의해 앱이 종료되었습니다.")
        return 0
    except Exception as e:
        print(f"앱 실행 중 오류 발생: {e}")
        return 1
    finally:
        # 임시 파일 삭제
        try:
            os.unlink(temp_path)
        except:
            pass

if __name__ == "__main__":
    sys.exit(main())