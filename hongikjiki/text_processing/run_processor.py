#!/usr/bin/env python3
"""
CLI 진입점: DocumentProcessor를 이용해 단일 파일 또는 디렉토리의 문서를 전처리하고 결과를 JSON으로 저장합니다.
"""

import argparse
import json
import os
from hongikjiki.text_processing.document_processor import DocumentProcessor

def main():
    parser = argparse.ArgumentParser(description="DocumentProcessor 기반 문서 전처리")
    parser.add_argument("--input-file", type=str, help="단일 텍스트 파일 경로")
    parser.add_argument("--input-dir", type=str, help="텍스트 파일이 포함된 디렉토리 경로")
    parser.add_argument("--output-file", type=str, required=True, help="전처리 결과를 저장할 JSON 파일 경로")
    
    args = parser.parse_args()
    processor = DocumentProcessor()

    if args.input_file:
        chunks = processor.process_file(args.input_file)
    elif args.input_dir:
        chunks = processor.process_directory(args.input_dir)
    else:
        print("에러: --input-file 또는 --input-dir 중 하나는 반드시 지정해야 합니다.")
        return

    # JSON으로 결과 저장
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        print(f"📄 처리된 청크 수: {len(chunks)}")
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"✅ 전처리 완료: {len(chunks)}개 청크가 {args.output_file}에 저장됨")

if __name__ == "__main__":
    main()
