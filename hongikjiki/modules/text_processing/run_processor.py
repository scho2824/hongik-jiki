#!/usr/bin/env python3
"""
CLI 진입점: DocumentProcessor를 이용해 단일 파일 또는 디렉토리의 문서를 전처리하고 결과를 JSON으로 저장합니다.
"""

import argparse
import json
import os
import logging
from pathlib import Path
from hongikjiki.modules.text_processing.document_processor import DocumentProcessor

ROOT_DIR = Path(__file__).resolve().parents[3]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="DocumentProcessor 기반 문서 전처리")
    parser.add_argument("--input-file", type=Path, help="단일 텍스트 파일 경로")
    parser.add_argument("--input-dir", type=Path, help="텍스트 파일이 포함된 디렉토리 경로")
    parser.add_argument("--output-file", type=Path, required=True, help="전처리 결과를 저장할 JSON 파일 경로")
    parser.add_argument("--chunk-size", type=int, default=1000, help="청크 최대 크기 (기본값: 1000)")
    parser.add_argument("--overlap", type=int, default=200, help="청크 간 중첩 길이 (기본값: 200)")
    
    args = parser.parse_args()
    processor = DocumentProcessor(chunk_size=args.chunk_size, overlap=args.overlap)

    if args.input_file:
        chunks = processor.process_file(args.input_file)
    elif args.input_dir:
        chunks = processor.process_directory(args.input_dir)
    else:
        print("에러: --input-file 또는 --input-dir 중 하나는 반드시 지정해야 합니다.")
        return

    # JSON으로 결과 저장
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as f:
        logger.info(f"📄 처리된 청크 수: {len(chunks)}")
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ 전처리 완료: {len(chunks)}개 청크가 {args.output_file}에 저장됨")

if __name__ == "__main__":
    main()
