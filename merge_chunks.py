# hongikjiki/utils/merge_chunks.py

import os
import json
import argparse

def merge_chunks(input_dir, output_file):
    merged = []

    # 디렉토리 내 모든 파일 탐색
    for idx, filename in enumerate(os.listdir(input_dir), start=1):
        if not filename.endswith(".json"):
            continue
        print(f"Processing file {idx}: {filename}")  # debug progress
        path = os.path.join(input_dir, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"⚠️ Skipping {filename} due to load error: {e}")
            continue

        # 각각의 청크가 dict 구조일 경우
        if isinstance(data, dict):
            content = data.get("content") or data.get("page_content", "")
            if not content.strip():
                print(f"⚠️ Skipping empty content in file: {filename} (single object)")
                continue
            merged.append({
                "metadata": data.get("metadata", {}),
                "content": content,
                "tags": data.get("tags", [])
            })
        elif isinstance(data, list):
            for idx, item in enumerate(data):
                if isinstance(item, dict):
                    content = item.get("content") or item.get("page_content", "")
                    if not content.strip():
                        print(f"⚠️ Skipping empty content in file: {filename}, item index: {idx}")
                        continue
                    merged.append({
                        "metadata": item.get("metadata", {}),
                        "content": content,
                        "tags": item.get("tags", [])
                    })
    
    # 병합된 리스트를 하나의 JSON 파일로 저장
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 병합 완료: {len(merged)}개의 청크가 {output_file}에 저장되었습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="청크들이 위치한 디렉토리 경로")
    parser.add_argument("--output-file", required=True, help="병합된 JSON 출력 경로")
    args = parser.parse_args()

    merge_chunks(args.input_dir, args.output_file)