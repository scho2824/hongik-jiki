import gradio as gr
from hongikjiki.core.chatbot import HongikJikiChatbot  # type: ignore[attr-defined]
from hongikjiki.modules.vector_store import load_vector_store
from hongikjiki.langchain_integration.llm import get_llm

# Initialize LLM and vector store, then instantiate chatbot with the correct signature
llm = get_llm("openai", model="gpt-4", temperature=0.7)
vector_store = load_vector_store(
    persist_directory="data/vector_store",
    collection_name="hongikjiki_jungbub",
    embedding_type="openai",
    embedding_kwargs={"model": "text-embedding-3-small"}
)
chatbot = HongikJikiChatbot(llm, vector_store)

def respond_to_query(user_input, chat_history):
    result = chatbot.answer_question(user_input)
    response = result.get('answer', '')
    followups = result.get('related_questions', [])[:3]
    message_block = (
        f"{response}\n\n**이와 관련된 질문들:**\n"
        + "\n".join(f"- {q}" for q in followups)
    )
    chat_history.append((user_input, message_block))
    return "", chat_history

with gr.Blocks(theme=gr.themes.Soft()) as demo:  # type: ignore[attr-defined]
    gr.Markdown("""
    # 🧘 홍익지기 챗봇
    정법에 기반한 통찰형 질문 안내자입니다. 지금 떠오른 질문을 자유롭게 입력해보세요.
    """)

    chatbot_ui = gr.Chatbot(label="홍익지기 응답기록", bubble_full_width=False)
    user_input = gr.Textbox(placeholder="예: '가족갈등이 있을 땐 어떻게 해야 하나요?'", show_label=False)
    with gr.Row():
        submit_btn = gr.Button("질문하기", variant="primary")
        clear_btn = gr.Button("초기화")

    submit_btn.click(fn=respond_to_query, inputs=[user_input, chatbot_ui], outputs=[user_input, chatbot_ui])
    clear_btn.click(lambda: ("", []), None, [user_input, chatbot_ui])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
