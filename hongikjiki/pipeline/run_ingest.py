#!/usr/bin/env python3
"""
Run Ingestion Pipeline

문서를 로드하고 벡터 저장소에 추가하는 스크립트
"""

import os
import sys
import logging
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from hongikjiki.modules.text_processing.document_processor import DocumentProcessor
from hongikjiki.modules.vector_store.chroma_store import ChromaVectorStore
from hongikjiki.modules.vector_store.embeddings import get_embeddings

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IngestPipeline")

def run_ingest_pipeline(data_dir: str | Path | None = None) -> None:
    """
    문서 로드 및 벡터 저장소 추가 파이프라인
    
    Args:
        data_dir: 문서 디렉토리 (기본값: ROOT_DIR/data/jungbub_teachings)
    """
    # 데이터 디렉토리 설정
    if data_dir is None:
        data_dir = ROOT_DIR / "data" / "jungbub_teachings"
    data_dir = Path(data_dir)
    
    logger.info(f"데이터 디렉토리: {data_dir}")
    
    # 임베딩 모델 초기화
    embeddings = get_embeddings("openai", model="text-embedding-ada-002")
    
    # 벡터 저장소 초기화
    vector_store = ChromaVectorStore(
        collection_name="hongikjiki_jungbub",
        persist_directory=str(ROOT_DIR / "data" / "vector_store"),
        embeddings=embeddings
    )
    
    # 초기 벡터 저장소 문서 수 확인
    initial_count = vector_store.count()
    logger.info(f"초기 벡터 저장소 문서 수: {initial_count}")
    
    # 문서 프로세서 초기화
    processor = DocumentProcessor()
    
    # 문서 로드 및 처리
    logger.info(f"문서 로드 및 처리 시작: {data_dir}")
    document_chunks = processor.process_directory(str(data_dir))
    logger.info(f"총 {len(document_chunks)}개 청크 생성")
    
    # 벡터 저장소에 추가
    if document_chunks:
        logger.info("벡터 저장소에 문서 추가 중...")
        vector_ids = vector_store.add_documents(document_chunks)
        logger.info(f"총 {len(vector_ids) if vector_ids else 0}개 문서 추가됨")
    else:
        logger.warning("추가할 문서 청크가 없습니다.")
    
    # 최종 벡터 저장소 문서 수 확인
    final_count = vector_store.count()
    logger.info(f"최종 벡터 저장소 문서 수: {final_count}")
    logger.info(f"추가된 문서 수: {final_count - initial_count}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="문서 로드 및 벡터 저장소 추가 파이프라인")
    parser.add_argument("--data-dir", type=str, help="문서 디렉토리 경로 (기본값: data/jungbub_teachings)")
    
    args = parser.parse_args()
    run_ingest_pipeline(args.data_dir)