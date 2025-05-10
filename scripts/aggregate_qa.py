# scripts/aggregate_qa.py
import json
import os

def load_raw_documents(data_dir):
    # 예시: data/raw 내의 텍스트 파일을 QA 쌍으로 만들어봄
    qa_pairs = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
                content = f.read().strip()
                qa_pairs.append({"question": f"What is in {filename}?", "answer": content})
    return qa_pairs

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()

    qa_data = load_raw_documents("data/raw")
    with open(args.out, "w", encoding="utf-8") as f:
        for item in qa_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")