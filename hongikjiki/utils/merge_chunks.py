# hongikjiki/utils/merge_chunks.py

import os
import json
import argparse

def merge_chunks(input_dir, output_file):
    merged = []

    # 디렉토리 내 모든 파일 탐색
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            path = os.path.join(input_dir, filename)
            with open(path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    # 각각의 청크가 dict 구조일 경우
                    if isinstance(data, dict):
                        merged.append(data)
                    # 혹시 리스트 형태라면 펼쳐서 추가
                    elif isinstance(data, list):
                        merged.extend(data)
                except json.JSONDecodeError:
                    print(f"⚠️ JSON 디코딩 실패: {filename}")
    
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