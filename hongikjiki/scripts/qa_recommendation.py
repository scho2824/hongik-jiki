import json
import argparse
from collections import defaultdict, Counter
from pathlib import Path
from hongikjiki.langchain_integration.llm import get_llm
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_qa_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_related_map(related_map, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(related_map, f, ensure_ascii=False, indent=2)

def build_tag_index(qa_data):
    tag_index = defaultdict(list)
    for qa in qa_data:
        for tag in qa.get("tags", []):
            tag_index[tag].append(qa)
    return tag_index

def recommend_questions(qa_data, tag_index, llm, top_k=5):
    id_to_question = {qa["qa_id"]: qa["question"] for qa in qa_data}
    recommendations = {}

    for qa in qa_data:
        target_id = qa["qa_id"]
        target_tags = qa.get("tags", [])
        counter = Counter()

        for tag in target_tags:
            for related in tag_index.get(tag, []):
                rid = related["qa_id"]
                if rid != target_id:
                    counter[rid] += 1

        related_ids = [rid for rid, _ in counter.most_common(top_k)]
        related_questions = []
        for rid in related_ids:
            question = id_to_question.get(rid)
            if question:
                try:
                    prompt = f"다음 질문은 정법적 관점에서 어떤 통찰을 주는지 1문장으로 설명해주세요:\n\n\"{question}\""
                    explanation = llm.generate(prompt).strip()
                except Exception as e:
                    logger.warning(f"Insight generation failed for question: {question[:50]}... → {e}")
                    explanation = "관련 질문입니다."

                related_questions.append({"question": question, "insight": explanation})

        recommendations[target_id] = related_questions

    return recommendations

def main():
    parser = argparse.ArgumentParser(description="Recommend related questions from QA dataset based on tags.")
    parser.add_argument("--input", required=True, help="Path to formatted QA JSON")
    parser.add_argument("--output", required=True, help="Path to save related questions JSON")
    parser.add_argument("--top_k", type=int, default=5, help="Number of related questions to return per QA")
    args = parser.parse_args()

    qa_data = load_qa_dataset(args.input)
    llm = get_llm()
    tag_index = build_tag_index(qa_data)
    related_map = recommend_questions(qa_data, tag_index, llm, top_k=args.top_k)
    save_related_map(related_map, args.output)

    logger.info(f"Generated related question map for {len(qa_data)} QA items → {args.output}")

if __name__ == "__main__":
    main()
