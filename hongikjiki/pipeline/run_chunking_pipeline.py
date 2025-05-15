#!/usr/bin/env python3
"""
Run Chunking Pipeline for Hongik-Jiki Chatbot

This script loads documents, normalizes and chunks them, then saves the chunks for tagging.
"""

import os
import sys
import json
import logging
from uuid import uuid4
from pathlib import Path
from typing import Dict, Any

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hongikjiki.modules.text_processing.document_loader import DocumentLoader
from hongikjiki.modules.text_processing.text_normalizer import TextNormalizer
from hongikjiki.modules.text_processing.document_chunker import DocumentChunker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ChunkingPipeline")

INPUT_DIR = ROOT_DIR / "data/jungbub_teachings"
OUTPUT_DIR = ROOT_DIR / "data/tag_data/input_chunks"

def run_chunking_pipeline(input_dir: str = str(INPUT_DIR), output_dir: str = str(OUTPUT_DIR)) -> None:
    output_path = Path(output_dir)
    os.makedirs(output_path, exist_ok=True)

    loader = DocumentLoader()
    normalizer = TextNormalizer()
    chunker = DocumentChunker()

    documents = loader.load_documents_from_dir(input_dir)
    logger.info(f"{len(documents)}개의 문서를 로드했습니다.")

    chunk_count = 0

    for doc in documents:
        doc_id = str(uuid4())
        normalized = normalizer.normalize(doc["content"])
        chunks = chunker.chunk_text(normalized)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_{i}"
            out_path = output_path / f"{chunk_id}.json"
            chunk_data: Dict[str, Any] = {
                "id": chunk_id,
                "content": chunk,
                "metadata": {
                    "source": doc.get("path", ""),
                    "filename": doc.get("filename", ""),
                    "label": doc.get("label", ""),
                    "chunk_index": i
                }
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(chunk_data, f, ensure_ascii=False, indent=2)
            chunk_count += 1

    logger.info(f"{chunk_count}개의 청크를 {output_path}에 저장했습니다.")

if __name__ == "__main__":
    run_chunking_pipeline()