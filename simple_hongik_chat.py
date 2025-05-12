#!/usr/bin/env python3
"""
홍익지기 챗봇 - 간단 버전
벡터 스토어 없이 OpenAI API만 사용하는 간단한 챗봇
"""
import os
import sys
import gradio as gr
from hongikjiki.utils import load_dotenv, setup_logging

# 기본 설정
logger = setup_logging()
logger.info("홍익지기 간단 챗봇 시작")

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
    sys.exit(1)

logger.info(f"API 키 확인: {api_key[:5]}...{api_key[-5:]}")
print(f"API 키 확인: {api_key[:5]}...{api_key[-5:]}")

try:
    # OpenAI 설정
    import openai
    client = openai.OpenAI(api_key=api_key)
    logger.info("OpenAI 클라이언트 초기화 완료")
    
    def answer_question(question: str) -> str:
        """OpenAI API를 사용하여 질문에 답변"""
        try:
            # API 호출
            prompt = f"""다음 질문에 대해 홍익인간과 정법의 관점에서 답변해주세요. 
            가능한 철학적이고 깊이 있는 답변을 작성해주세요.
            답변은 한국어로 작성해주세요.
            
            질문: {question}
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
            logger.error(f"API 호출 오류: {e}")
            return f"오류가 발생했습니다: {str(e)}"
    
    # Gradio 인터페이스 구성
    demo = gr.Interface(
        fn=answer_question,
        inputs=gr.Textbox(placeholder="질문을 입력하세요...", label="질문"),
        outputs=gr.Textbox(label="답변"),
        title="홍익지기 챗봇 (간단 버전)",
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
    
    # 앱 실행
    print("홍익지기 챗봇을 시작합니다...")
    demo.launch(share=False)  # share=True로 설정하면 공개 URL 생성
    
except Exception as e:
    import traceback
    logger.error(f"오류 발생: {e}\n{traceback.format_exc()}")
    print(f"오류 발생: {e}")
    print("\n상세 오류:")
    traceback.print_exc()
    sys.exit(1)