

import json
from typing import List, Dict

def load_dataset(input_path: str) -> List[Dict[str, str]]:
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)

def simple_qa_generator(dataset: List[Dict[str, str]]) -> List[Dict[str, str]]:
    qa_pairs = []
    for item in dataset:
        text = item.get("content", "")
        if len(text.strip()) < 30:
            continue
        question = f"이 문장은 무엇을 말하고 있나요?"
        answer = text.strip()
        qa_pairs.append({"question": question, "answer": answer})
    return qa_pairs

def save_qa_dataset(output_path: str, qa_pairs: List[Dict[str, str]]):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    print(f"📥 입력 파일: {args.input_file}")
    print(f"📤 출력 파일: {args.output_file}")

    dataset = load_dataset(args.input_file)
    qa_pairs = simple_qa_generator(dataset)
    save_qa_dataset(args.output_file, qa_pairs)

    print(f"✅ 총 {len(qa_pairs)}개의 QA 쌍이 생성되어 저장되었습니다.")