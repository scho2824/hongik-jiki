# hongikjiki/app/handlers.py
import gradio as gr
import logging
from hongikjiki.app.config import USE_MESSAGE_FORMAT

logger = logging.getLogger("HongikJikiChatBot")

# 전역 변수
chatbot_instance = None

def initialize_handlers(chatbot):
    """전역 챗봇 인스턴스 초기화"""
    global chatbot_instance
    chatbot_instance = chatbot

def answer_question(message, history):
    """사용자 질문에 답변 생성 핸들러"""
    global chatbot_instance
    
    if not chatbot_instance:
        logger.error("챗봇 인스턴스가 초기화되지 않았습니다.")
        return history, None
        
    try:
        result = chatbot_instance.answer_question(message, history)
        formatted_answer = result["answer"]
        file_output = result["file"]
        
        # Gradio 버전에 따른 반환 형식 조정
        if USE_MESSAGE_FORMAT:
            history.append((message, formatted_answer))
            return history, file_output
        else:
            return history + [(message, formatted_answer)], file_output
            
    except Exception as e:
        logger.error(f"답변 생성 중 오류 발생: {e}")
        error_msg = f"오류가 발생했습니다: {str(e)}"
        
        if USE_MESSAGE_FORMAT:
            history.append((message, error_msg))
            return history, None
        else:
            return history + [(message, error_msg)], None

def get_related_questions():
    """관련 질문 목록 가져오기"""
    global chatbot_instance
    if chatbot_instance:
        return chatbot_instance.get_related_questions()
    return []

def update_question_buttons():
    """관련 질문 버튼 업데이트"""
    questions = get_related_questions()
    if not questions or len(questions) == 0:
        return (
            gr.update(visible=False, value=""),
            gr.update(visible=False, value=""),
            gr.update(visible=False, value="")
        )
    
    updates = []
    # 각 버튼 업데이트
    for i in range(3):
        if i < len(questions):
            q = questions[i]
            question = q.get("question", "관련 질문")
            updates.append(gr.update(visible=True, value=question))
        else:
            updates.append(gr.update(visible=False, value=""))
    
    return tuple(updates)

def click_related_question(question_text):
    """관련 질문 클릭 핸들러"""
    return question_text, gr.update(value=question_text)