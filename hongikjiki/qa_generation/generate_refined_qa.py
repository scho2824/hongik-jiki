

import json
import os

from collections import defaultdict

from openai import OpenAI  # Ensure proper OpenAI client import
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def refine_qa_pair(qa):
    """
    Refine a QA pair with cleaned question, quoted insight, and explanation.
    """
    original_question = qa.get("question", "")
    original_answer = qa.get("answer", "")

    # Placeholder logic; to be replaced with actual model-inference or regex-based methods
    cleaned_question = f"정제된 질문: {original_question.strip().replace('이 내용은', '').strip()}"
    quoted_insight = original_answer[:150] + "..." if len(original_answer) > 150 else original_answer
    insight_explanation = "이 인용은 핵심 의미를 요약한 것입니다. (자동 생성 예시)"

    return {
        **qa,
        "cleaned_question": cleaned_question,
        "quoted_insight": quoted_insight,
        "insight_explanation": insight_explanation
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