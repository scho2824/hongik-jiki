"""
문서 관리 모듈
정법 교육 자료의 추가, 삭제, 업데이트 및 변경 감지 기능 제공
"""
import os
import json
import time
import hashlib
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Set
from datetime import datetime

from hongikjiki.text_processor import TextProcessor
from hongikjiki.utils import find_documents, ensure_dir

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 문서 메타데이터 파일 경로
META_FILE = Path("./data/document_metadata.json")

def calculate_file_hash(file_path: Union[str, Path]) -> str:
    """
    파일의 MD5 해시값 계산
    
    Args:
        file_path: 파일 경로
        
    Returns:
        MD5 해시값
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return ""
    
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def load_document_metadata() -> Dict[str, Dict[str, Any]]:
    """
    문서 메타데이터 로드
    
    Returns:
        메타데이터 딕셔너리
    """
    if not META_FILE.exists():
        return {}
    
    try:
        with open(META_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"메타데이터 로드 오류: {e}")
        return {}

def save_document_metadata(metadata: Dict[str, Dict[str, Any]]) -> None:
    """
    문서 메타데이터 저장
    
    Args:
        metadata: 메타데이터 딕셔너리
    """
    try:
        os.makedirs(META_FILE.parent, exist_ok=True)
        with open(META_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"메타데이터 저장 오류: {e}")

def detect_changed_files(data_dir: Union[str, Path]) -> Dict[str, List[Path]]:
    """
    변경된 파일 감지
    
    Args:
        data_dir: 문서 디렉토리
        
    Returns:
        변경된 파일 목록 (추가/수정/삭제)
    """
    data_dir = Path(data_dir)
    
    # 기존 메타데이터 로드
    metadata = load_document_metadata()
    
    # 현재 디렉토리의 파일 목록
    current_files = set(str(f) for f in find_documents(data_dir))
    
    # 기존 파일 목록
    old_files = set(metadata.keys())
    
    # 변경 파일 계산
    added_files = current_files - old_files
    deleted_files = old_files - current_files
    
    # 수정된 파일 찾기
    modified_files = []
    for file_path_str in current_files.intersection(old_files):
        file_path = Path(file_path_str)
        current_hash = calculate_file_hash(file_path)
        
        if metadata[file_path_str].get("hash") != current_hash:
            modified_files.append(file_path)
    
    return {
        "added": [Path(f) for f in added_files],
        "deleted": [Path(f) for f in deleted_files],
        "modified": modified_files
    }

def update_document_metadata(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    파일의 메타데이터 생성 또는 업데이트
    
    Args:
        file_path: 파일 경로
        
    Returns:
        업데이트된 메타데이터
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return {}
    
    # 파일 정보
    stats = file_path.stat()
    
    # 메타데이터 작성
    metadata = {
        "filename": file_path.name,
        "path": str(file_path),
        "file_type": file_path.suffix,
        "size_bytes": stats.st_size,
        "created_time": time.ctime(stats.st_ctime),
        "modified_time": time.ctime(stats.st_mtime),
        "hash": calculate_file_hash(file_path),
        "last_indexed": time.strftime("%Y-%m-%d %H:%M:%S"),
        "indexed_chunks": 0  # 청크 수는 처리 후 업데이트됨
    }
    
    return metadata

def update_documents(data_dir: Union[str, Path], vector_store=None, force_reindex: bool = False) -> Dict[str, Any]:
    """
    문서 변경 감지 및 업데이트
    
    Args:
        data_dir: 문서 디렉토리
        vector_store: 벡터 저장소 객체 (None이면 새로 생성)
        force_reindex: 강제 재색인화 여부
        
    Returns:
        업데이트 결과 정보
    """
    data_dir = Path(data_dir)
    
    if not data_dir.exists() or not data_dir.is_dir():
        logger.error(f"유효하지 않은 디렉토리: {data_dir}")
        return {"success": False, "error": "유효하지 않은 디렉토리입니다."}
    
    # 벡터 저장소가 제공되지 않으면 로드
    if vector_store is None:
        from hongikjiki.vector_store import VectorStore
        vector_store = VectorStore()
    
    # 텍스트 처리기 초기화
    processor = TextProcessor()
    
    # 변경된 파일 감지
    changed_files = detect_changed_files(data_dir) if not force_reindex else {
        "added": find_documents(data_dir),
        "deleted": [],
        "modified": []
    }
    
    # 변경된 파일이 없으면 종료
    if not changed_files["added"] and not changed_files["deleted"] and not changed_files["modified"]:
        logger.info("변경된 문서가 없습니다.")
        return {"success": True, "message": "변경된 문서가 없습니다.", "changes": 0}
    
    # 메타데이터 로드
    metadata = load_document_metadata()
    
    # 삭제된 문서의 벡터 제거
    for file_path in changed_files["deleted"]:
        file_path_str = str(file_path)
        if file_path_str in metadata:
            # 파일명으로 벡터 식별 및 제거
            file_prefix = file_path.name
            try:
                vector_store.collection.delete(where={"file_name": {"$eq": file_path.name}})
                logger.info(f"문서 벡터 제거됨: {file_path}")
            except Exception as e:
                logger.error(f"벡터 제거 오류 ({file_path}): {e}")
            
            # 메타데이터에서 제거
            del metadata[file_path_str]
    
    # 추가/수정된 문서 처리 및 색인화
    processed_count = 0
    total_chunks = 0
    
    for file_path in changed_files["added"] + changed_files["modified"]:
        file_path_str = str(file_path)
        
        try:
            # 문서 청크 생성
            document_chunks = processor.prepare_document_chunks(file_path)
            
            if not document_chunks:
                logger.warning(f"문서에서 추출된 청크가 없습니다: {file_path}")
                continue
            
            # 기존 벡터 제거 (수정된 문서의 경우)
            if file_path in changed_files["modified"]:
                try:
                    vector_store.collection.delete(where={"file_name": {"$eq": file_path.name}})
                except Exception as e:
                    logger.error(f"기존 벡터 제거 오류 ({file_path}): {e}")
            
            # 벡터 저장소에 추가
            vector_store.add_documents(document_chunks)
            
            # 메타데이터 업데이트
            metadata[file_path_str] = update_document_metadata(file_path)
            metadata[file_path_str]["indexed_chunks"] = len(document_chunks)
            
            processed_count += 1
            total_chunks += len(document_chunks)
            
            logger.info(f"문서 처리됨: {file_path} - {len(document_chunks)} 청크")
            
        except Exception as e:
            logger.error(f"문서 처리 오류 ({file_path}): {e}")
    
    # 메타데이터 저장
    save_document_metadata(metadata)
    
    logger.info(f"문서 업데이트 완료: "
               f"추가 {len(changed_files['added'])}, "
               f"수정 {len(changed_files['modified'])}, "
               f"삭제 {len(changed_files['deleted'])}")
    
    return {
        "success": True,
        "message": "문서 업데이트가 완료되었습니다.",
        "added": len(changed_files["added"]),
        "modified": len(changed_files["modified"]),
        "deleted": len(changed_files["deleted"]),
        "processed": processed_count,
        "total_chunks": total_chunks
    }

def list_documents(data_dir: Union[str, Path] = None) -> List[Dict[str, Any]]:
    """
    문서 목록 및 상태 반환
    
    Args:
        data_dir: 문서 디렉토리 (None이면 메타데이터 사용)
        
    Returns:
        문서 정보 목록
    """
    # 메타데이터 로드
    metadata = load_document_metadata()
    
    if data_dir is None:
        # 메타데이터만 사용
        return [
            {
                "filename": info["filename"],
                "path": info["path"],
                "file_type": info["file_type"],
                "size_bytes": info["size_bytes"],
                "last_indexed": info.get("last_indexed", ""),
                "indexed_chunks": info.get("indexed_chunks", 0)
            }
            for info in metadata.values()
        ]
    else:
        # 디렉토리에서 파일 목록 가져오기
        data_dir = Path(data_dir)
        documents = []
        
        for file_path in find_documents(data_dir):
            file_path_str = str(file_path)
            
            if file_path_str in metadata:
                # 기존 메타데이터 사용
                info = metadata[file_path_str]
                documents.append({
                    "filename": info["filename"],
                    "path": info["path"],
                    "file_type": info["file_type"],
                    "size_bytes": info["size_bytes"],
                    "last_indexed": info.get("last_indexed", ""),
                    "indexed_chunks": info.get("indexed_chunks", 0)
                })
            else:
                # 기본 정보만 표시
                stats = file_path.stat()
                documents.append({
                    "filename": file_path.name,
                    "path": file_path_str,
                    "file_type": file_path.suffix,
                    "size_bytes": stats.st_size,
                    "last_indexed": "",
                    "indexed_chunks": 0
                })
        
        return documents