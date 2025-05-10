import json
import os

from collections import defaultdict

from openai import OpenAI  # Ensure proper OpenAI client import
from dotenv import load_dotenv

import re

def summarize_answer(answer: str, max_sentences: int = 2) -> str:
    """
    Return the first up to max_sentences sentences of the answer.
    """
    # split into sentences by punctuation
    sentences = re.split(r'(?<=[.?!])\s+', answer.strip())
    # return up to max_sentences joined back with spaces
    return " ".join(sentences[:max_sentences]).strip()

def enhance_tags(answer: str, original_tags: list) -> list:
    # 태그 보강 로직 (현재는 원본 태그만 사용)
    return list(dict.fromkeys(original_tags))

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def refine_qa_pair(qa):
    """
    Refine a QA pair with summarized answer, quoted insight, explanation, and enhanced tags.
    """
    original_question = qa.get("question", "").strip()
    original_answer = qa.get("answer", "").strip()

    # 1) 간결 요약
    summary = summarize_answer(original_answer, max_sentences=2)
    # 2) 근거 인용 스니펫 (최대 80자)
    snippet = original_answer[:80].rstrip() + "…" if len(original_answer) > 80 else original_answer
    # 3) 태그 보강
    tags = enhance_tags(original_answer, qa.get("tags", []))

    insight_explanation = (
        f"이 인용은 ‘{snippet}’라는 문장을 근거로 삼아, {summary}라고 요약할 수 있습니다."
    )

    return {
        **qa,
        "answer": summary,
        "full_answer": original_answer,
        "insight_summary": summary,
        "quoted_insight": snippet,
        "insight_explanation": insight_explanation,
        "tags": tags
    }

def process_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as infile:
        qa_data = json.load(infile)

    refined_data = [refine_qa_pair(qa) for qa in qa_data]

    # Group by quoted_insight
    grouped = defaultdict(list)
    for qa in refined_data:
        quoted = qa["quoted_insight"]
        grouped[quoted].append(qa)

    # Keep only 1~2 representative questions per insight
    deduped_data = []
    for group in grouped.values():
        sorted_group = sorted(group, key=lambda x: x["question"])
        deduped_data.extend(sorted_group[:2])  # Keep top 2 per quoted_insight

    # Remove exact duplicate questions across all insights
    unique_data = []
    seen_questions = set()
    for qa in deduped_data:
        question_text = qa.get("cleaned_question", qa.get("question", "")).strip()
        if question_text not in seen_questions:
            seen_questions.add(question_text)
            unique_data.append(qa)
    deduped_data = unique_data

    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(deduped_data, outfile, ensure_ascii=False, indent=2)

    print(f"✅ Refined QA saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to original QA JSON")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save refined QA")
    args = parser.parse_args()

    process_file(args.input_file, args.output_file)