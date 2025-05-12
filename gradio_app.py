import os
import gradio as gr
from hongikjiki.chatbot import HongikJikiChatBot

# 챗봇 초기화
chatbot = HongikJikiChatBot(
    persist_directory="data/vector_store",
    embedding_type="openai",  # 또는 "huggingface"
    llm_type="openai",
    collection_name="hongikjiki_documents",
    embedding_kwargs={"model_name": "text-embedding-3-small"},
    llm_kwargs={"model": "gpt-4o"},
    tag_patterns_path=os.path.abspath("data/config/tag_patterns.json")
)

# 질문 처리 함수
def answer_question(user_input):
    response = chatbot.chat(user_input)
    result = f"💬 **답변:**\n{response['answer']}\n\n"
    if response.get("tags"):
        result += f"🏷 **태그:** {', '.join(response['tags'])}\n\n"
    if response.get("related_questions"):
        result += "**📎 관련 질문:**\n" + "\n".join(f"- {q}" for q in response["related_questions"])
    return result

# Gradio 인터페이스 설정
iface = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(lines=2, placeholder="정법이란 무엇인가요?"),
    outputs="markdown",
    title="홍익지기 챗봇",
    description="정법 강의를 기반으로 질문에 답하는 GPT 챗봇입니다. 삶의 방향, 감정, 사회, 영성 등에 대한 통찰을 얻어보세요.",
    flagging_mode="never"
)

if __name__ == "__main__":
    iface.launch()