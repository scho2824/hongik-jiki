

"""
refined_high_insight_qa_dataset.jsonl에 대해 자동 태그를 부여하는 스크립트
"""

import json
from pathlib import Path
from typing import List, Dict

TAG_PATTERN_PATH = "data/config/tag_patterns.json"
INPUT_FILE = "data/qa/refined_high_insight_qa_dataset.jsonl"
OUTPUT_FILE = "data/qa/refined_high_insight_qa_dataset_tagged.jsonl"

def load_tag_patterns(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["tags"]

def extract_tags(text: str, tag_patterns: Dict[str, List[str]]) -> List[str]:
    tags = []
    for tag, keywords in tag_patterns.items():
        if any(keyword in text for keyword in keywords):
            tags.append(tag)
    return list(set(tags))

def tag_qa_dataset(input_path: str, output_path: str, tag_patterns: Dict[str, List[str]]):
    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:
        
        for line in infile:
            qa = json.loads(line)

            # 태깅 대상 텍스트 결합
            base_text = qa.get("question", "") + " " + qa.get("answer", "")
            base_text += " " + qa.get("insight_summary", "")
            base_text += " " + qa.get("quoted_insight", "")
            base_text = base_text.strip()

            # 태그 추출
            tags = extract_tags(base_text, tag_patterns)
            qa["tags"] = tags

            # 저장
            outfile.write(json.dumps(qa, ensure_ascii=False) + "\n")

    print(f"✅ 태깅 완료: {output_path}")

if __name__ == "__main__":
    if not Path(INPUT_FILE).exists():
        raise FileNotFoundError(f"입력 파일이 존재하지 않습니다: {INPUT_FILE}")
    tag_patterns = load_tag_patterns(TAG_PATTERN_PATH)
    tag_qa_dataset(INPUT_FILE, OUTPUT_FILE, tag_patterns)