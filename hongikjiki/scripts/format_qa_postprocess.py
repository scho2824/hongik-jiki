
import uuid
import re

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


import json
import argparse
from pathlib import Path

def load_qa_data(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_qa_data(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def extract_tags(text):
    tag_keywords = ["수행", "정법", "마음", "청년", "정치", "사회", "영성", "감정", "죽음", "책임", "공부", "소통"]
    return [tag for tag in tag_keywords if tag in text]

def summarize_answer(answer):
    sentences = re.split(r"[.?!]\s+", answer.strip())
    return sentences[0] if sentences else answer.strip()

def format_qa_items(raw_qa):
    formatted = []
    for item in raw_qa:
        qa_id = str(uuid.uuid4())
        question = item.get("question", "")
        answer = item.get("answer", "")
        source_id = item.get("lecture_id") or item.get("source_id") or ""
        combined_text = f"{question} {answer}"
        tags = extract_tags(combined_text)
        insight_summary = summarize_answer(answer)

        formatted.append({
            "qa_id": qa_id,
            "source_id": source_id,
            "question": question,
            "answer": answer,
            "tags": tags,
            "insight_summary": insight_summary
        })
    return formatted

def main():
    parser = argparse.ArgumentParser(description="Format QA JSON to standard structure.")
    parser.add_argument("--input", required=True, help="Path to input QA JSON file")
    parser.add_argument("--output", required=True, help="Path to save formatted QA JSON file")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    raw_data = load_qa_data(input_path)
    formatted_data = format_qa_items(raw_data)
    save_qa_data(formatted_data, output_path)

    logger.info(f"Formatted {len(formatted_data)} QA items to: {output_path}")

    # Optional: generate related question mapping
    generate_related = True
    if generate_related:
        from collections import defaultdict, Counter

        def build_tag_index(qa_data):
            tag_index = defaultdict(list)
            for qa in qa_data:
                for tag in qa.get("tags", []):
                    tag_index[tag].append(qa)
            return tag_index

        def recommend_questions(qa_data, tag_index, top_k=5):
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
                related_questions = [id_to_question[rid] for rid in related_ids if rid in id_to_question]
                recommendations[target_id] = related_questions

            return recommendations

        related_map = recommend_questions(formatted_data, build_tag_index(formatted_data), top_k=5)
        related_output_path = output_path.with_name(output_path.stem + "_related.json")
        save_qa_data(related_map, related_output_path)
        logger.info(f"Generated related questions file: {related_output_path}")

if __name__ == "__main__":
    main()