import os
import gradio as gr
import tempfile
import time
import json
from hongikjiki.langchain_integration.chain import get_chatbot_chain
from hongikjiki.langchain_integration.llm import get_llm
from hongikjiki.vector_store import load_vector_store
from gradio.components import Button, Markdown
from flask import request, jsonify

related_question_buttons = []

with open("data/qa/high_insight_qa_dataset_formatted_related.json", "r", encoding="utf-8") as f:
    related_questions_map = json.load(f)

# Load related question map
with open("data/qa/high_insight_qa_dataset_formatted_related.json", "r", encoding="utf-8") as f:
    related_map = json.load(f)

# 챗봇 초기화
vector_store = load_vector_store(
    persist_directory="data/vector_store",
    embedding_type="openai",
    collection_name="hongikjiki_documents",
    embedding_kwargs={"model_name": "text-embedding-3-small"}
)
llm = get_llm(llm_type="openai", model="gpt-4o")
chatbot = get_chatbot_chain(llm=llm, vector_store=vector_store)

def answer_question(message, history):
    response = chatbot.run(message)
    answer = f"💬 {response.get('answer', '')}"
    lecture_id = response.get("lecture_id", "")
    lecture_title = response.get("lecture_title", "")
    qa_id = response.get("qa_id", "")
    related = related_questions_map.get(qa_id, [])
    source_summary = response.get("source_summary", "")

    if lecture_id or lecture_title:
        source = f"\n\n🔗 출처:"
        if lecture_title:
            source += f" 「{lecture_title}」"
        if lecture_id:
            source += f" (강의 번호: {lecture_id})"
        answer += source

    if source_summary:
        answer += f"\n\n📝 강의 요약:\n{source_summary}"

    global related_question_buttons
    related_question_buttons = related

    # Create temporary file for download
    temp = tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".txt")
    temp.write(answer)
    temp.close()
    return answer, temp.name

iface = gr.ChatInterface(
    fn=answer_question,
    title="홍익지기 챗봇",
    description="정법 강의를 기반으로 질문에 답하는 GPT 챗봇입니다.\n삶의 방향, 감정, 사회, 영성 등에 대한 통찰을 얻어보세요.",
    additional_outputs=[gr.File(label="답변 다운로드")]
)

gr.Examples(
    examples=[
        "영혼과 육신의 관계는?",
        "현대 사회가 무너지는 이유는 무엇인가요?",
        "청년이 사회에서 가져야 할 태도는?",
        "운명은 정해져 있나요?",
        "수행이란 정확히 무엇인가요?",
        "정법은 불교나 유교와 무엇이 다른가요?"
    ],
    inputs=iface.input_textbox,
    label="💡 예시 질문을 선택해보세요"
)

def insert_related_question(q):
    return q

related_group = gr.Group(
    [gr.Markdown("📎 관련 질문을 눌러보세요:")] +
    [gr.Button(q["question"], tooltip=q["insight"]).click(fn=insert_related_question, inputs=[], outputs=iface.input_textbox)
     for q in related_question_buttons]
)

from flask import Flask
app = Flask(__name__)

@app.route("/recommendations", methods=["GET"])
def get_recommendations():
    qa_id = request.args.get("qa_id", "")
    related = related_map.get(qa_id, [])
    return jsonify({"qa_id": qa_id, "related_questions": related})

if __name__ == "__main__":
    related_group.render()
    iface.launch()