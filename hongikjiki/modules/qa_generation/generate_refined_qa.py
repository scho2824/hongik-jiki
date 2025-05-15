import json
import os

from collections import defaultdict

from openai import OpenAI  # Ensure proper OpenAI client import
from dotenv import load_dotenv

import re

from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[3]

from hongikjiki.modules.tagging.tag_schema import TagSchema
from hongikjiki.modules.tagging.tag_extractor import TagExtractor

tag_schema_path = ROOT_DIR / "data" / "converted_tag_schema.yaml"
tag_schema = TagSchema.load_from_yaml(str(tag_schema_path))
tag_extractor = TagExtractor(tag_schema)

def summarize_answer(answer: str, max_sentences: int = 2) -> str:
    """
    Return the first up to max_sentences sentences of the answer.
    """
    # split into sentences by punctuation
    sentences = re.split(r'(?<=[.?!])\s+', answer.strip())
    # return up to max_sentences joined back with spaces
    return " ".join(sentences[:max_sentences]).strip()

def enhance_tags(answer: str, original_tags: list) -> list:
    tags = tag_extractor.extract_tags(answer)
    # Handle extract_tags returning either dict or tuple
    if isinstance(tags, tuple):
        main_tags, near_tags = tags
        extracted = list(main_tags.keys())
    else:
        extracted = list(tags.keys()) if isinstance(tags, dict) else []
    all_tags = list(dict.fromkeys(original_tags + extracted))
    return all_tags

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def refine_single_qa(qa_item):
    """
    Refine a QA pair with summarized answer, quoted insight, explanation, and enhanced tags.
    """
    original_question = qa_item.get("question", "").strip()
    original_answer = qa_item.get("answer", "").strip()

    summary = summarize_answer(original_answer, max_sentences=2)
    snippet = original_answer[:80].rstrip() + "…" if len(original_answer) > 80 else original_answer
    tags = enhance_tags(original_answer, qa_item.get("tags", []))

    explanation = (
        f"이 인용은 ‘{snippet}’라는 문장을 근거로 삼아, {summary}라고 요약할 수 있습니다."
        if summary else f"이 인용은 '{snippet}'과 관련된 간단한 설명입니다."
    )

    tag_descriptions = {}
    for tag in tags:
        tag_obj = tag_schema.tags.get(tag)
        tag_descriptions[tag] = tag_obj.description if tag_obj and hasattr(tag_obj, "description") else "이 개념"

    return {
        **qa_item,
        "answer": summary,
        "full_answer": original_answer,
        "insight_summary": summary,
        "quoted_insight": snippet,
        "insight_explanation": explanation,
        "tags": tags,
        "tag_descriptions": tag_descriptions
    }

def load_qa_data(input_path):
    """Load QA data from a JSON file."""
    with input_path.open("r", encoding="utf-8") as infile:
        return json.load(infile)

def save_qa_data(output_path, data):
    """Save QA data to a JSON file."""
    with output_path.open("w", encoding="utf-8") as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=2)

def process_file(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)
    # Load original QA data
    qa_data = load_qa_data(input_path)
    # Refine each QA item
    refined_data = [refine_single_qa(qa) for qa in qa_data]

    # Group by quoted_insight to cluster similar insights
    grouped = defaultdict(list)
    for qa_item in refined_data:
        quoted = qa_item["quoted_insight"]
        grouped[quoted].append(qa_item)

    # Keep only 1~2 representative questions per insight group
    deduped_data = []
    for group in grouped.values():
        sorted_group = sorted(group, key=lambda x: x["question"])
        deduped_data.extend(sorted_group[:2])  # Keep top 2 per quoted_insight

    # Remove exact duplicate questions across all insights
    unique_data = []
    seen_questions = set()
    for qa_item in deduped_data:
        question_text = qa_item.get("cleaned_question", qa_item.get("question", "")).strip()
        if question_text not in seen_questions:
            seen_questions.add(question_text)
            unique_data.append(qa_item)
    deduped_data = unique_data

    # Save refined QA data
    save_qa_data(output_path, deduped_data)

    print(f"✅ Refined QA saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=Path, required=True, help="Path to original QA JSON")
    parser.add_argument("--output_file", type=Path, required=True, help="Path to save refined QA")
    args = parser.parse_args()

    process_file(args.input_file, args.output_file)