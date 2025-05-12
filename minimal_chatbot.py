#!/usr/bin/env python3
"""
홍익지기 챗봇 - 최소 버전
벡터 스토어와 파일 처리 없이 단순히 LLM만 사용
"""
import os
import sys
import gradio as gr
from hongikjiki.utils import load_dotenv, setup_logging

# 로깅 설정
logger = setup_logging()
logger.info("간단 챗봇 시작")

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
    sys.exit(1)

print(f"API 키 확인: {api_key[:5]}...{api_key[-5:]}")

# OpenAI 클라이언트 초기화
try:
    import openai
    client = openai.OpenAI(api_key=api_key)
    print("✅ OpenAI 클라이언트 초기화 성공")
except Exception as e:
    print(f"❌ OpenAI 클라이언트 초기화 실패: {e}")
    sys.exit(1)

# 채팅 함수
def chat(message, history):
    try:
        prompt = f"""다음 질문에 대해 홍익인간과 정법의 관점에서 답변해주세요.
        가능한 상세하게 답변하고, 답변은 한국어로 작성해주세요.
        홍익인간은 '널리 인간을 이롭게 한다'는 의미를 가지며, 정법은 자연의 법칙과 조화로운 삶을 추구합니다.
        
        질문: {message}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=800
        )
        
        answer = response.choices[0].message.content.strip()
        return f"💬 {answer}"
    except Exception as e:
        print(f"❌ 답변 생성 오류: {e}")
        return f"오류가 발생했습니다: {str(e)}"

# 인터페이스 생성
demo = gr.ChatInterface(
    fn=chat,
    title="홍익지기 챗봇 (최소 버전)",
    description="정법 강의에 기반한 철학적 통찰을 제공하는 챗봇입니다.",
    examples=[
        "영혼과 육신의 관계는?",
        "현대 사회가 무너지는 이유는 무엇인가요?",
        "청년이 사회에서 가져야 할 태도는?",
        "운명은 정해져 있나요?",
        "수행이란 정확히 무엇인가요?",
        "정법은 불교나 유교와 무엇이 다른가요?"
    ]
)

# 앱 실행
if __name__ == "__main__":
    print("간단 홍익지기 챗봇을 시작합니다...")
    demo.launch()