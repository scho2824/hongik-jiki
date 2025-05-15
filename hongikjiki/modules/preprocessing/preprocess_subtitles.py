import os
import json
from typing import List, Dict, Any
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

def normalize_text(text: str) -> str:
    """Apply basic normalization: strip, unify quotes, fix dashes, etc."""
    return (
        text.replace("“", "\"").replace("”", "\"")
            .replace("‘", "'").replace("’", "'")
            .replace("–", "-").replace("—", "-")
            .strip()
    )

def extract_tags(text: str) -> List[str]:
    """Delegate tag extraction to the tagging module."""
    try:
        from hongikjiki.modules.tagging.tag_extractor import TagExtractor
        # Load tag schema from JSON file
        from hongikjiki.modules.tagging.tag_schema import TagSchema
        tag_schema = TagSchema.load_from_file("data/tag_keywords.json")
        extractor = TagExtractor(tag_schema)
        raw_tags = extractor.extract_tags(text)
        # Process dictionary or tuple output into a flat list of tag strings
        if isinstance(raw_tags, tuple) and len(raw_tags) == 2:
            main_tags, near_tags = raw_tags
            # Prefer main tags if available, else take top near tags
            if main_tags:
                return list(main_tags.keys())
            return list(dict(near_tags).keys())  # assume near_tags is list of (tag, score)
        elif isinstance(raw_tags, dict):
            # Filter tags by confidence threshold (e.g., >= 0.6)
            return [tag for tag, score in raw_tags.items() if score >= 0.6]
        else:
            return []
    except Exception as e:
        print(f"⚠️ 태그 추출 오류: {e}")
        return []

def read_subtitle_files(input_dir: str) -> List[Dict[str, Any]]:
    """
    Recursively load .json or .txt subtitle files from input_dir and sub‑directories.
    Each item is a dict containing content (str), source (str), and tags (List[str]).
    """
    data: List[Dict[str, Any]] = []
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
            logger.warning(f"⚠️ Failed to load {path}: {e}")

    return data

def save_dataset(output_file: str, dataset: List[Dict[str, Any]]):
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