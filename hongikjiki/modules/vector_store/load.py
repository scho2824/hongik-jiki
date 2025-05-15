import os
import logging
import shutil
import time
import traceback
import chromadb
from chromadb.config import Settings
from hongikjiki.modules.vector_store.embeddings import get_embeddings
from typing import Optional, Dict, Any

logger = logging.getLogger("HongikJikiChatBot")

def create_collection(path: str, name: str, metadata: Optional[dict] = None):
    client = chromadb.PersistentClient(
        path=path,
        settings=Settings(anonymized_telemetry=False, allow_reset=True)
    )
    collection = client.get_or_create_collection(
        name=name,
        metadata=metadata or {}
    )
    return client, collection

def ensure_openai_key(kwargs: dict):
    if "api_key" not in kwargs:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables")
            raise ValueError("OpenAI API 키가 환경 변수에 설정되지 않았습니다.")
        kwargs["api_key"] = api_key
    return kwargs

def load_vector_store(
    persist_directory: str,
    collection_name: str,
    embedding_type: str = "openai",
    embedding_kwargs: Optional[Dict[str, Any]] = None,
    reset_if_error: bool = False,
    fallback_to_temp: bool = True
):
    """
    Load or create a Chroma collection and initialize embeddings.

    Args:
        persist_directory (str): Chroma DB storage path.
        collection_name (str): Chroma collection name.
        embedding_type (str): Embedding backend (e.g., "openai", "huggingface").
        embedding_kwargs (dict): Parameters to pass to the embedding class.
        reset_if_error (bool): If True, will try to reset the store if error occurs.
        fallback_to_temp (bool): If True, will create a temporary store if all else fails.

    Returns:
        tuple: (Chroma collection, embedding object)
    """
    embedding_kwargs = embedding_kwargs or {}
    
    # Ensure API key is passed if using OpenAI
    if embedding_type.lower() == "openai":
        embedding_kwargs = ensure_openai_key(embedding_kwargs)
    
    # Get embeddings
    try:
        embeddings = get_embeddings(embedding_type, **embedding_kwargs)
        logger.info(f"임베딩 객체 초기화 성공: {embedding_type}")
    except Exception as emb_error:
        logger.error(f"임베딩 초기화 오류: {emb_error}")
        raise ValueError(f"임베딩 초기화 실패: {emb_error}")
    
    # Ensure directory exists
    os.makedirs(persist_directory, exist_ok=True)
    
    # 여러 단계의 시도를 통해 벡터 스토어 로드
    # 1. 기본 설정으로 시도
    try:
        logger.info(f"벡터 스토어 로드 시도 1: {persist_directory}, 컬렉션: {collection_name}")
        client, collection = create_collection(persist_directory, collection_name, metadata={"hnsw:space": "cosine"})
        
        logger.info(f"벡터 스토어 로드 성공: {collection_name}")
        return collection, embeddings
        
    except Exception as e:
        logger.warning(f"벡터 스토어 로드 첫 번째 시도 실패: {e}")
        
        # 2. 단순 설정으로 시도
        try:
            logger.info("벡터 스토어 로드 시도 2: 단순 설정")
            client, collection = create_collection(persist_directory, collection_name)
            logger.info(f"단순 설정으로 벡터 스토어 로드 성공: {collection_name}")
            return collection, embeddings
        except Exception as e2:
            logger.warning(f"벡터 스토어 로드 두 번째 시도 실패: {e2}")
            
            # 3. 리셋 후 다시 시도
            if reset_if_error:
                logger.warning("벡터 스토어 재설정 시도...")
                
                try:
                    # 기존 저장소 백업
                    backup_dir = f"{persist_directory}_backup_{int(time.time())}"
                    if os.path.exists(persist_directory):
                        shutil.copytree(persist_directory, backup_dir)
                        logger.info(f"벡터 스토어 백업 완료: {backup_dir}")
                        
                        # 디렉토리 삭제 및 재생성
                        shutil.rmtree(persist_directory)
                        os.makedirs(persist_directory, exist_ok=True)
                    
                    # 새 클라이언트 및 컬렉션 생성
                    client, collection = create_collection(persist_directory, collection_name)
                    
                    logger.info(f"벡터 스토어 재설정 및 새 컬렉션 생성 성공: {collection_name}")
                    return collection, embeddings
                    
                except Exception as reset_error:
                    logger.error(f"벡터 스토어 재설정 실패: {reset_error}")
                    # 실패 시 다음 단계로 진행
            
            # 4. 임시 디렉토리에 생성 시도
            if fallback_to_temp:
                logger.warning("임시 디렉토리에 벡터 스토어 생성 시도...")
                try:
                    import tempfile
                    temp_dir = tempfile.mkdtemp()
                    logger.info(f"임시 디렉토리 생성: {temp_dir}")
                    
                    client, collection = create_collection(temp_dir, collection_name)
                    
                    logger.info(f"임시 디렉토리에 벡터 스토어 생성 성공: {collection_name}")
                    return collection, embeddings
                    
                except Exception as temp_error:
                    logger.error(f"임시 벡터 스토어 생성 실패: {temp_error}")
                    # 실패 시 최종 예외 발생
            
            # 5. 모든 시도 실패
            error_msg = f"모든 벡터 스토어 로드 시도 실패: {e2}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise ValueError(error_msg)