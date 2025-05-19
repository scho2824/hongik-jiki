# hongikjiki/app/ui.py
import gradio as gr
import logging
from hongikjiki.app.config import USE_MESSAGE_FORMAT
from hongikjiki.app.handlers import (
    initialize_handlers, 
    answer_question, 
    get_related_questions, 
    update_question_buttons,
    click_related_question
)

logger = logging.getLogger("HongikJikiChatBot")

def create_ui(chatbot_instance):
    """Gradio UI 생성"""
    # 핸들러 초기화
    initialize_handlers(chatbot_instance)
    
    with gr.Blocks(title="홍익지기 챗봇") as demo:
        gr.Markdown("# 🌕 홍익지기 챗봇")
        gr.Markdown("정법 강의를 기반으로 질문에 답하는 GPT 챗봇입니다.")
        gr.Markdown("삶의 방향, 감정, 사회, 영성 등에 대한 통찰을 얻어보세요.")
        
        # 버전에 따른 챗봇 인터페이스 선택
        chatbot = gr.Chatbot(height=500, type='messages')
        
        # 입력 및 버튼 영역
        with gr.Row():
            with gr.Column(scale=8):
                msg = gr.Textbox(
                    placeholder="질문을 입력하세요...",
                    label="질문",
                    show_label=False
                )
            
            with gr.Column(scale=1):
                submit_btn = gr.Button("질문하기")
        
        # 파일 다운로드 영역
        download_file = gr.File(label="답변 다운로드", visible=False)
        
        # 예시 질문 (동적 업데이트)
        example_questions = chatbot_instance.generate_related_questions([], "")
        example_questions = [q["question"] for q in example_questions]
        example_dropdown = gr.Dropdown(
            choices=example_questions,
            label="💡 예시 질문을 선택해보세요",
            interactive=True
        )
        
        # 예시 질문 선택 시 입력창에 반영
        example_dropdown.change(
            fn=lambda x: x,
            inputs=example_dropdown,
            outputs=msg
        )
        
        # 관련 질문 영역 
        with gr.Accordion("📎 관련 질문", open=True) as related_accordion:
            related_questions_component = gr.JSON(get_related_questions, visible=False)
            
            # 관련 질문 버튼
            question1_btn = gr.Button("관련 질문 1", visible=False)
            question2_btn = gr.Button("관련 질문 2", visible=False)
            question3_btn = gr.Button("관련 질문 3", visible=False)
        
        # 질문 응답 후 질문 버튼 업데이트 함수
        def process_after_answer(chatbot_output, file_output):
            """질문-응답 후 관련 질문 버튼 업데이트"""
            button_updates = update_question_buttons()
            return chatbot_output, file_output, button_updates[0], button_updates[1], button_updates[2]
        
        # 이벤트 핸들러 설정
        submit_response = submit_btn.click(
            answer_question,
            inputs=[msg, chatbot],
            outputs=[chatbot, download_file]
        ).then(
            process_after_answer,
            inputs=[chatbot, download_file],
            outputs=[chatbot, download_file, question1_btn, question2_btn, question3_btn]
        )
        
        msg_response = msg.submit(
            answer_question,
            inputs=[msg, chatbot],
            outputs=[chatbot, download_file]
        ).then(
            process_after_answer,
            inputs=[chatbot, download_file],
            outputs=[chatbot, download_file, question1_btn, question2_btn, question3_btn]
        )
        
        # 관련 질문 버튼 클릭 이벤트
        question1_btn.click(
            click_related_question,
            inputs=[question1_btn],
            outputs=[msg]
        )
        
        question2_btn.click(
            click_related_question,
            inputs=[question2_btn],
            outputs=[msg]
        )
        
        question3_btn.click(
            click_related_question,
            inputs=[question3_btn],
            outputs=[msg]
        )
        
    return demo