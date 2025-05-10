# scripts/aggregate_qa.py
import json
from pathlib import Path

def load_raw_documents(data_dir):
    # 모든 하위 폴더 내의 .txt 파일들을 재귀적으로 탐색
    qa_pairs = []
    for path in Path(data_dir).rglob("*.txt"):
        try:
            content = path.read_text(encoding="utf-8").strip()
            qa_pairs.append({
                "question": f"What is in {path.name}?",
                "answer": content
            })
        except Exception as e:
            print(f"⚠️ Error reading {path}: {e}")
    return qa_pairs

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()

    qa_data = load_raw_documents("data/jungbub_teachings")
    with open(args.out, "w", encoding="utf-8") as f:
        for item in qa_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")