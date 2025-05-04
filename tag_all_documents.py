"""
Hongik-Jiki 문서 태깅 스크립트

이 스크립트는 정법 가르침 문서들을 읽어서 자동으로 태그를 할당하고, 
벡터 저장소에 태그된 문서를 추가합니다.
"""
import os
import sys
import json
import logging
import argparse
from typing import List, Dict, Any

# 로깅 설정
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("HongikJikiTagger")

# 프로젝트 루트 경로 추가
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# 필요한 모듈 임포트
from hongikjiki.tagging.tag_schema import TagSchema
from hongikjiki.tagging.tag_extractor import TagExtractor
from hongikjiki.tagging.tag_analyzer import TagAnalyzer
from hongikjiki.tagging.tagging_tools import TaggingBatch
from hongikjiki.text_processing.document_processor import DocumentProcessor
from hongikjiki.vector_store.chroma_store import ChromaVectorStore
from hongikjiki.vector_store.embeddings import get_embeddings

def setup_components(args):
    """필요한 컴포넌트 초기화"""
    logger.info("컴포넌트 초기화 중...")
    
    # 태그 스키마 초기화
    tag_schema = TagSchema(args.tag_schema)
    logger.info(f"태그 스키마 로드 완료: {len(tag_schema.tags)} 개의 태그")
    
    # 태그 추출기 초기화
    tag_extractor = TagExtractor(tag_schema, args.tag_patterns)
    logger.info("태그 추출기 초기화 완료")
    
    # 태그 분석기 초기화
    tag_analyzer = TagAnalyzer(tag_schema)
    if os.path.exists(args.stats_file):
        tag_analyzer.load_tag_statistics(args.stats_file)
    logger.info("태그 분석기 초기화 완료")
    
    # 문서 처리기 초기화
    document_processor = DocumentProcessor()
    logger.info("문서 처리기 초기화 완료")
    
    # 태깅 배치 초기화
    batch_tagger = TaggingBatch(tag_schema, tag_extractor, args.output_dir)
    logger.info("태깅 배치 프로세서 초기화 완료")
    
    # 임베딩 모델 초기화
    embeddings = get_embeddings(args.embedding_type, model_name=args.embedding_model)
    logger.info(f"임베딩 모델 초기화 완료: {args.embedding_type}")
    
    # 벡터 저장소 초기화
    vector_store = ChromaVectorStore(
        collection_name=args.collection_name,
        persist_directory=args.persist_directory,
        embeddings=embeddings
    )
    logger.info(f"벡터 저장소 초기화 완료: {args.collection_name}")
    
    return {
        "tag_schema": tag_schema,
        "tag_extractor": tag_extractor,
        "tag_analyzer": tag_analyzer,
        "document_processor": document_processor,
        "batch_tagger": batch_tagger,
        "vector_store": vector_store
    }

def process_document_files(components, docs_dir: str, min_confidence: float = 0.6):
    """문서 파일 처리 및 태깅"""
    logger.info(f"{docs_dir} 디렉토리에서 문서 처리 시작...")
    
    document_processor = components["document_processor"]
    batch_tagger = components["batch_tagger"]
    
    # 결과 저장용 데이터
    processed_files = 0
    tagged_documents = []
    
    # 디렉토리 내 모든 파일 처리
    for root, _, files in os.walk(docs_dir):
        for file in files:
            # 숨김 파일 및 처리 불가능한 파일 건너뛰기
            if file.startswith(".") or not file.endswith((".txt", ".rtf", ".pdf", ".docx")):
                continue
                
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, docs_dir)
            
            try:
                logger.info(f"문서 처리 중: {relative_path}")
                
                # 문서 처리
                chunks = document_processor.process_file(file_path)
                
                if not chunks:
                    logger.warning(f"문서에서 청크를 추출하지 못했습니다: {relative_path}")
                    continue
                
                # 각 청크 처리
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{relative_path}_chunk_{i}"
                    chunk_content = chunk["content"]
                    chunk_metadata = chunk["metadata"].copy()
                    
                    # 청크 ID 설정
                    chunk_metadata["chunk_id"] = chunk_id
                    chunk_metadata["source_file"] = relative_path
                    chunk_metadata["chunk_index"] = i
                    
                    # 문서 객체 생성
                    doc_item = {
                        "id": chunk_id,
                        "content": chunk_content,
                        "metadata": chunk_metadata
                    }
                    
                    # 태깅을 위해 문서 저장
                    tagged_documents.append(doc_item)
                
                processed_files += 1
                logger.info(f"문서 처리 완료: {relative_path}, {len(chunks)} 청크 생성")
                
            except Exception as e:
                logger.error(f"문서 처리 오류: {relative_path}, {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    # 태깅 배치 처리
    if tagged_documents:
        logger.info(f"총 {len(tagged_documents)} 청크 태깅 시작...")
        stats = batch_tagger.process_batch(tagged_documents, min_confidence)
        logger.info(f"태깅 완료: {stats['successfully_tagged']} 청크 태깅됨")
        
        # 배치 결과 내보내기
        batch_tagger.export_batch_results("data/tag_data/batch_report.json")
    else:
        logger.warning("처리된 문서가 없습니다.")
    
    return {
        "processed_files": processed_files,
        "total_chunks": len(tagged_documents),
        "tagged_documents": tagged_documents
    }

def add_tagged_documents_to_vectorstore(components, tagged_documents: List[Dict[str, Any]]):
    """태그된 문서를 벡터 저장소에 추가"""
    logger.info("태그된 문서를 벡터 저장소에 추가하는 중...")
    
    vector_store = components["vector_store"]
    tag_dir = os.path.join("data", "tag_data", "auto_tagged")
    
    # 태그 정보 수집
    document_tags = {}
    for file in os.listdir(tag_dir):
        if file.endswith('_tags.json'):
            try:
                with open(os.path.join(tag_dir, file), 'r') as f:
                    tag_data = json.load(f)
                    document_id = tag_data.get('document_id')
                    tags = tag_data.get('tags', {})
                    if document_id and tags:
                        document_tags[document_id] = tags
            except Exception as e:
                logger.error(f"태그 파일 처리 오류: {file}, {e}")
    
    # 각 문서에 태그 추가하고 벡터 저장소에 추가
    vector_docs = []
    for doc in tagged_documents:
        doc_id = doc["id"]
        
        # 태그 정보가 있으면 메타데이터에 추가
        if doc_id in document_tags:
            doc["metadata"]["tags"] = document_tags[doc_id]
            logger.info(f"문서에 태그 추가: {doc_id}, 태그: {list(document_tags[doc_id].keys())}")
        
        # 벡터 저장소용 문서 객체 생성
        vector_docs.append({
            "content": doc["content"],
            "metadata": doc["metadata"]
        })
    
    # 벡터 저장소에 문서 추가
    if vector_docs:
        try:
            ids = vector_store.add_documents(vector_docs)
            logger.info(f"벡터 저장소에 {len(ids)} 개의 문서 추가 완료")
        except Exception as e:
            logger.error(f"벡터 저장소 추가 오류: {e}")
            import traceback
            logger.error(traceback.format_exc())
    else:
        logger.warning("태그된 문서가 없습니다.")
    
    # 벡터 저장소 문서 수 확인
    doc_count = vector_store.count()
    logger.info(f"벡터 저장소 문서 수: {doc_count}")

def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(description="Hongik-Jiki 문서 태깅 및 벡터 저장소 추가")
    
    parser.add_argument("--docs-dir", type=str, default="data/jungbub_teachings",
                        help="정법 가르침 문서 디렉토리")
    parser.add_argument("--output-dir", type=str, default="data/tag_data/auto_tagged",
                        help="태그된 문서 저장 디렉토리")
    parser.add_argument("--tag-schema", type=str, default="data/config/tag_schema.yaml",
                        help="태그 스키마 파일 경로")
    parser.add_argument("--tag-patterns", type=str, default="data/config/tag_patterns.json",
                        help="태그 패턴 파일 경로")
    parser.add_argument("--stats-file", type=str, default="data/tag_data/tag_statistics.json",
                        help="태그 통계 파일 경로")
    parser.add_argument("--persist-directory", type=str, default="./data/vector_store",
                        help="벡터 저장소 디렉토리")
    parser.add_argument("--collection-name", type=str, default="hongikjiki_documents",
                        help="벡터 저장소 컬렉션 이름")
    parser.add_argument("--embedding-type", type=str, default="huggingface",
                        help="임베딩 모델 타입 (huggingface 또는 openai)")
    parser.add_argument("--embedding-model", type=str, 
                        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                        help="임베딩 모델 이름")
    parser.add_argument("--min-confidence", type=float, default=0.6,
                        help="태그 최소 신뢰도 임계값")
    parser.add_argument("--skip-vectorstore", action="store_true",
                        help="벡터 저장소 추가 단계 건너뛰기")
    
    return parser.parse_args()

def main():
    """메인 함수"""
    # 인수 파싱
    args = parse_arguments()
    
    logger.info("=== Hongik-Jiki 문서 태깅 및 벡터 저장소 추가 시작 ===")
    
    # 필요한 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.persist_directory, exist_ok=True)
    os.makedirs(os.path.dirname(args.stats_file), exist_ok=True)
    
    # 컴포넌트 초기화
    components = setup_components(args)
    
    # 문서 처리 및 태깅
    process_results = process_document_files(components, args.docs_dir, args.min_confidence)
    
    # 태그 통계 저장
    components["tag_analyzer"].save_tag_statistics(args.stats_file)
    
    # 벡터 저장소에 문서 추가
    if not args.skip_vectorstore and process_results["tagged_documents"]:
        add_tagged_documents_to_vectorstore(components, process_results["tagged_documents"])
    
    logger.info("=== Hongik-Jiki 문서 태깅 및 벡터 저장소 추가 완료 ===")
    logger.info(f"처리된 파일: {process_results['processed_files']}")
    logger.info(f"총 청크: {process_results['total_chunks']}")
    
    print("\n문서 태깅 및 벡터 저장소 추가가 완료되었습니다.")
    print(f"처리된 파일: {process_results['processed_files']}")
    print(f"생성된 청크: {process_results['total_chunks']}")
    print(f"태그된 문서는 {args.output_dir} 디렉토리에 저장되었습니다.")
    print(f"벡터 저장소는 {args.persist_directory} 디렉토리에 있습니다.")

if __name__ == "__main__":
    main()
    