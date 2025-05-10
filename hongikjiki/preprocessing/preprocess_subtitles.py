import os
import json
from typing import List, Dict
from pathlib import Path

def normalize_text(text: str) -> str:
    """Apply basic normalization: strip, unify quotes, fix dashes, etc."""
    return (
        text.replace("“", "\"").replace("”", "\"")
            .replace("‘", "'").replace("’", "'")
            .replace("–", "-").replace("—", "-")
            .strip()
    )

def extract_tags(text: str) -> List[str]:
    """Tag simple themes based on keywords."""
    tags = []
    if "사랑" in text or "관계" in text:
        tags.append("관계")
    if "죽음" in text or "삶" in text:
        tags.append("삶과 죽음")
    if "자유" in text or "선택" in text:
        tags.append("자유")
    if "에너지" in text:
        tags.append("에너지")
    return tags

def read_subtitle_files(input_dir: str) -> List[Dict[str, str]]:
    """
    Recursively load .json or .txt subtitle files from input_dir and sub‑directories.
    Each item in the returned list is {"content": "<text>"} with length > 20.
    """
    data: List[Dict[str, str]] = []
    ALLOWED_SUFFIXES = {".json", ".txt"}

    for path in Path(input_dir).rglob("*"):
        if not path.is_file() or path.suffix.lower() not in ALLOWED_SUFFIXES:
            continue

        try:
            if path.suffix.lower() == ".json":
                with path.open("r", encoding="utf-8") as f:
                    items = json.load(f)
                    for item in items:
                        text = item.get("text", "").strip()
                        if text and len(text) > 20:
                            data.append({
                                "content": normalize_text(text),
                                "source": str(path),
                                "tags": extract_tags(text)
                            })
            else:  # .txt
                with path.open("r", encoding="utf-8") as f:
                    for line in f:
                        text = line.strip()
                        if text and len(text) > 20:
                            data.append({
                                "content": normalize_text(text),
                                "source": str(path),
                                "tags": extract_tags(text)
                            })
        except Exception as e:
            print(f"⚠️ Failed to load {path}: {e}")

    return data

def save_dataset(output_file: str, dataset: List[Dict[str, str]]):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    print(f"📁 자막 디렉토리: {args.input_dir}")
    print(f"💾 출력 파일: {args.output_file}")

    dataset = read_subtitle_files(args.input_dir)
    save_dataset(args.output_file, dataset)

    print(f"✅ 총 {len(dataset)}개의 문장이 저장되었습니다.")