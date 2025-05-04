import os
import json
from typing import List, Dict

def read_subtitle_files(input_dir: str) -> List[Dict[str, str]]:
    data = []
    for filename in os.listdir(input_dir):
        if not (filename.endswith(".json") or filename.endswith(".txt")):
            continue
        path = os.path.join(input_dir, filename)
        try:
            if filename.endswith(".json"):
                with open(path, "r", encoding="utf-8") as f:
                    items = json.load(f)
                    for item in items:
                        text = item.get("text", "").strip()
                        if text and len(text) > 20:
                            data.append({"content": text})
            elif filename.endswith(".txt"):
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        text = line.strip()
                        if text and len(text) > 20:
                            data.append({"content": text})
        except Exception as e:
            print(f"⚠️ Failed to load {filename}: {e}")
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