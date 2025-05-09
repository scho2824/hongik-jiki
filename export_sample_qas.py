import json
import random
import argparse

def is_valid_qa(item):
    return bool(item.get("question")) and bool(item.get("answer"))

def main(input_file, sample_size, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = [item for item in json.load(f) if is_valid_qa(item)]

    # Prioritize samples with tags
    data_with_tags = [item for item in data if item.get("tags")]
    data_without_tags = [item for item in data if not item.get("tags")]

    # Combine and sample
    combined_data = data_with_tags + data_without_tags
    samples = random.sample(combined_data, min(sample_size, len(combined_data)))

    with open(output_file, "w", encoding="utf-8") as f_out:
        for i, item in enumerate(samples, 1):
            f_out.write(f"[Q{i}] {item['question']}\n")
            f_out.write(f"[A{i}] {item['answer']}\n")
            f_out.write(f"Tags: {item.get('tags', [])}\n\n")

    print(f"✅ {len(samples)} QA pairs exported to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export random QA samples from dataset")
    parser.add_argument("--input_file", type=str, default="data/qa/jungbub_qa_dataset.json", help="Path to input QA dataset")
    parser.add_argument("--sample_size", type=int, default=15, help="Number of samples to export")
    parser.add_argument("--output_file", type=str, default="data/qa/sample_qas.txt", help="Output file path")
    args = parser.parse_args()

    main(args.input_file, args.sample_size, args.output_file)