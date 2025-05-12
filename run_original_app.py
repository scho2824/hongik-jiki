#!/usr/bin/env python3
"""
홍익지기 챗봇 - 원본 앱 실행 스크립트
개선된 오류 처리 및 이슈 수정 포함
"""
import os
import sys
import traceback
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
    
    try:
        print("Gradio 앱 시작을 위한 모듈 임포트 중...")
        # 필요한 모듈 임포트
        import gradio as gr
        import tempfile
        import time
        import json
        from hongikjiki.langchain_integration.chain import get_chatbot_chain
        from hongikjiki.langchain_integration.llm import get_llm
        from hongikjiki.vector_store import load_vector_store
        from hongikjiki.vector_store.chroma_store import ChromaVectorStore
        
        print("벡터 스토어 초기화 중...")
        # 벡터 스토어 초기화 - 향상된 오류 처리
        vector_store = None
        try:
            # 벡터 스토어 로드 시도
            collection, embeddings = load_vector_store(
                persist_directory="data/vector_store",
                collection_name="hongikjiki_documents",
                embedding_type="openai",
                embedding_kwargs={"model_name": "text-embedding-3-small", "api_key": api_key},
                reset_if_error=True,
                fallback_to_temp=True
            )
            
            # 벡터 스토어 객체 생성
            vector_store = ChromaVectorStore(collection, embeddings=embeddings)
            print("✅ 벡터 스토어 로드 성공")
        except Exception as e:
            print(f"⚠️ 벡터 스토어 로드 실패: {e}")
            print("간단 모드로 전환합니다")
            vector_store = None
        
        # LLM 초기화
        print("LLM 초기화 중...")
        llm = get_llm(llm_type="openai", model="gpt-4o", api_key=api_key)
        print("✅ LLM 초기화 성공")
        
        # 관련 질문 데이터 로드
        related_question_buttons = []
        try:
            qa_file_path = "data/qa/high_insight_qa_dataset_formatted_related.json"
            if os.path.exists(qa_file_path):
                with open(qa_file_path, "r", encoding="utf-8") as f:
                    related_questions_map = json.load(f)
                
                # 관련 질문 맵 로드
                with open(qa_file_path, "r", encoding="utf-8") as f:
                    related_map = json.load(f)
                print("✅ 관련 질문 데이터 로드 성공")
            else:
                print(f"⚠️ 관련 질문 파일이 없습니다: {qa_file_path}")
                related_questions_map = {}
                related_map = {}
        except Exception as e:
            print(f"⚠️ 관련 질문 데이터 로드 오류: {e}")
            related_questions_map = {}
            related_map = {}
        
        # 챗봇 초기화
        if vector_store:
            # 벡터 스토어 기반 챗봇
            chatbot = get_chatbot_chain(llm=llm, vector_store=vector_store)
            print("✅ 벡터 스토어 기반 챗봇 초기화 성공")
        else:
            # 간단 모드 챗봇
            class SimpleChatbot:
                def __init__(self, llm):
                    self.llm = llm
                    
                def run(self, query):
                    prompt = f"""다음 질문에 대해 홍익인간과 정법의 관점에서 답변해주세요. 
                    가능한 상세하게 답변하고, 답변은 한국어로 작성해주세요.
                    홍익인간은 '널리 인간을 이롭게 한다'는 의미를 가지며, 정법은 자연의 법칙과 조화로운 삶을 추구합니다.
                    
                    질문: {query}
                    """
                    answer = self.llm.generate(prompt)
                    return {"answer": answer}
            
            chatbot = SimpleChatbot(llm)
            print("✅ 간단 모드 챗봇 초기화 성공")
        
        # 텍스트 입력 박스 생성
        input_box = gr.Textbox(placeholder="질문을 입력하세요...", label="질문")
        
        # 질문 응답 함수
        def answer_question(message, history):
            try:
                # 벡터 스토어가 없으면 간단 모드 실행
                if not vector_store:
                    response = chatbot.run(message)
                    if isinstance(response, str):
                        answer = f"💬 {response}"
                    else:
                        answer = f"💬 {response.get('answer', '')}"
                    return answer, ""
                
                # 정상 모드: 벡터 스토어 기반 챗봇 실행
                response = chatbot.run(message)
                if isinstance(response, str):
                    return f"💬 {response}", ""
                
                answer = f"💬 {response.get('answer', '')}"
                lecture_id = response.get("lecture_id", "")
                lecture_title = response.get("lecture_title", "")
                qa_id = response.get("qa_id", "")
                related = related_questions_map.get(qa_id, [])
                source_summary = response.get("source_summary", "")
                
                # 출처 정보 추가
                if lecture_id or lecture_title:
                    source = f"\n\n🔗 출처:"
                    if lecture_title:
                        source += f" 「{lecture_title}」"
                    if lecture_id:
                        source += f" (강의 번호: {lecture_id})"
                    answer += source
                
                # 소스 요약 추가
                if source_summary:
                    answer += f"\n\n📝 강의 요약:\n{source_summary}"
                
                # 관련 질문 업데이트
                global related_question_buttons
                related_question_buttons = related
                
                # 다운로드용 임시 파일 생성 - 파일 경로 안전하게 생성
                try:
                    temp = tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".txt")
                    temp.write(answer)
                    temp.close()
                    return answer, temp.name
                except Exception as temp_error:
                    logger.error(f"임시 파일 생성 오류: {temp_error}")
                    # 파일 생성에 실패하면 빈 문자열 반환
                    return answer, ""
            except Exception as e:
                error_msg = f"답변 생성 중 오류 발생: {str(e)}"
                print(f"⚠️ {error_msg}")
                return f"💬 죄송합니다. {error_msg}", ""
        
        # Gradio 인터페이스 생성
        iface = gr.ChatInterface(
            fn=answer_question,
            textbox=input_box,
            title="홍익지기 챗봇",
            description=f"정법 강의를 기반으로 질문에 답하는 GPT 챗봇입니다.\n삶의 방향, 감정, 사회, 영성 등에 대한 통찰을 얻어보세요.\n{'(벡터 스토어 사용 중)' if vector_store else '(간단 모드 - 벡터 스토어 없음)'}",
            additional_outputs=[gr.File(label="답변 다운로드")]
        )
        
        # 예시 질문 추가
        gr.Examples(
            examples=[
                "영혼과 육신의 관계는?",
                "현대 사회가 무너지는 이유는 무엇인가요?",
                "청년이 사회에서 가져야 할 태도는?",
                "운명은 정해져 있나요?",
                "수행이란 정확히 무엇인가요?",
                "정법은 불교나 유교와 무엇이 다른가요?"
            ],
            inputs=input_box,
            label="💡 예시 질문을 선택해보세요"
        )
        
        # 관련 질문 삽입 함수
        def insert_related_question(q):
            return q
        
        # 관련 질문 그룹 (초기는 비어있음)
        related_group = gr.Group(
            [gr.Markdown("📎 관련 질문을 눌러보세요:")] +
            [gr.Button(q["question"], tooltip=q["insight"]).click(fn=insert_related_question, inputs=[], outputs=iface.input_textbox)
             for q in related_question_buttons]
        )
        
        # API 서버 설정
        from flask import Flask, request, jsonify
        app = Flask(__name__)
        
        @app.route("/recommendations", methods=["GET"])
        def get_recommendations():
            qa_id = request.args.get("qa_id", "")
            related = related_map.get(qa_id, [])
            return jsonify({"qa_id": qa_id, "related_questions": related})
        
        # 앱 실행
        print("\n홍익지기 챗봇을 시작합니다...")
        related_group.render()
        
        # 앱 실행 - prevent_thread_lock=True로 설정
        iface.launch(prevent_thread_lock=True)
        print("✅ 앱이 실행 중입니다. 웹 브라우저에서 접속하세요.")
        print("종료하려면 Ctrl+C를 누르세요.")
        
        # 메인 스레드 유지
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n사용자에 의해 앱이 종료되었습니다.")
            return 0
        
    except Exception as e:
        print(f"오류 발생: {e}")
        print("\n상세 오류:")
        traceback.print_exc()
        logger.error(f"앱 실행 오류: {e}\n{traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    sys.exit(main())