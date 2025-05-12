import os
import gradio as gr
import tempfile
from hongikjiki.utils import load_dotenv
from hongikjiki.langchain_integration.llm import get_llm
from hongikjiki.vector_store.embeddings import get_embeddings

# 환경 변수 로드
load_dotenv()

# API 키 확인
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다. .env 파일에 설정하세요.")
print(f"API 키 확인: {api_key[:5]}...{api_key[-5:]}")

# 간단한 데모 인터페이스 생성
def answer_question(message):
    # 실제 OpenAI API 호출
    try:
        llm = get_llm(llm_type="openai", model="gpt-4o")
        response = llm.generate(f"다음 질문에 대해 홍익인간과 정법의 관점에서 답변해주세요: {message}")
        return f"💬 {response}"
    except Exception as e:
        return f"오류 발생: {str(e)}"

iface = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(placeholder="질문을 입력하세요...", label="질문"),
    outputs="text",
    title="홍익지기 챗봇 (간단 버전)",
    description="정법 강의 기반 챗봇의 간단 버전입니다. 벡터 저장소를 사용하지 않고 OpenAI API만 사용합니다."
)

if __name__ == "__main__":
    print("Gradio 앱 시작 중...")
    print("앱이 시작되면 웹 브라우저에서 접속할 수 있는 URL이 표시됩니다.")
    try:
        # Gradio 앱 실행
        iface.launch(show_error=True)
    except KeyboardInterrupt:
        print("\n사용자에 의해 앱이 종료되었습니다.")
    except Exception as e:
        print(f"\n오류 발생: {e}")