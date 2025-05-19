#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import logging
from dotenv import load_dotenv
import json
import pickle
from datetime import datetime

# 프로젝트 루트 디렉토리를 Python 경로에 추가
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("HongikJiki")

# 진행 상황 저장 파일 경로
PROGRESS_FILE = ROOT_DIR / "data" / "progress.json"
VECTOR_STORE_FILE = ROOT_DIR / "data" / "vector_store.pkl"

def save_progress(stage: str, status: str = "completed"):
    """진행 상황 저장"""
    progress = {}
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f:
            progress = json.load(f)
    
    progress[stage] = {
        "status": status,
        "timestamp": str(datetime.now())
    }
    
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

def get_progress(stage: str) -> bool:
    """특정 단계의 진행 상황 확인"""
    if not PROGRESS_FILE.exists():
        return False
    
    with open(PROGRESS_FILE, 'r') as f:
        progress = json.load(f)
    
    return progress.get(stage, {}).get("status") == "completed"

def init_environment():
    """환경 설정 초기화"""
    if get_progress("environment"):
        logger.info("환경 설정이 이미 완료되었습니다.")
        return os.getenv("OPENAI_API_KEY")
    
    # API 키 확인
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
    
    save_progress("environment")
    logger.info("환경 설정 완료")
    return api_key

def process_documents():
    """문서 처리 및 벡터 스토어 생성"""
    from hongikjiki.modules.text_processing.document_processor import DocumentProcessor
    from hongikjiki.modules.vector_store.chroma_store import ChromaVectorStore
    from hongikjiki.modules.vector_store.embeddings import get_embeddings
    
    # 문서 처리기 초기화
    processor = DocumentProcessor()
    
    # 임베딩 모델 초기화
    embeddings = get_embeddings("openai", model="text-embedding-ada-002")
    
    # 벡터 스토어 초기화
    persist_directory = str(ROOT_DIR / "data" / "vector_store")
    vector_store = ChromaVectorStore(
        collection_name="hongikjiki_documents",
        persist_directory=persist_directory,
        embeddings=embeddings
    )
    
    # 이미 처리된 문서가 있는지 확인
    if get_progress("documents") and os.path.exists(persist_directory):
        logger.info("문서 처리가 이미 완료되었습니다. 저장된 벡터 스토어를 사용합니다.")
        return vector_store
    
    # 문서 처리
    data_dir = ROOT_DIR / "data" / "jungbub_teachings"
    logger.info(f"문서 처리 시작: {data_dir}")
    
    chunks = processor.process_directory(data_dir)
    if chunks:
        vector_store.add_documents(chunks)
        logger.info(f"벡터 스토어에 {len(chunks)}개 청크 추가 완료")
        save_progress("documents")
    else:
        logger.warning("처리된 문서 청크가 없습니다.")
    
    return vector_store

def init_chatbot(vector_store):
    """챗봇 초기화"""
    if get_progress("chatbot"):
        logger.info("챗봇이 이미 초기화되었습니다.")
    
    from hongikjiki.core.chatbot import HongikJikiChatbot
    from hongikjiki.langchain_integration.llm import get_llm
    
    # LLM 초기화 - max_tokens와 temperature 조정
    llm = get_llm(
        llm_type="openai",
        model="gpt-4",
        temperature=0.8,  # 창의성 증가
        max_tokens=2000   # 더 긴 답변 생성
    )
    
    # 챗봇 초기화
    chatbot = HongikJikiChatbot(llm, vector_store)
    logger.info("챗봇 초기화 완료")
    
    save_progress("chatbot")
    return chatbot

def reset_progress():
    """진행 상황 초기화"""
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
    
    # 벡터 스토어 디렉토리 삭제
    persist_directory = ROOT_DIR / "data" / "vector_store"
    if persist_directory.exists():
        import shutil
        shutil.rmtree(persist_directory)
    
    logger.info("진행 상황이 초기화되었습니다.")

def main():
    """메인 실행 함수"""
    try:
        # 명령행 인자 처리
        if len(sys.argv) > 1 and sys.argv[1] == "--reset":
            reset_progress()
        
        # 환경 설정
        api_key = init_environment()
        
        # 문서 처리 및 벡터 스토어 생성
        vector_store = process_documents()
        
        # 챗봇 초기화
        chatbot = init_chatbot(vector_store)
        
        # 웹 인터페이스 실행
        from hongikjiki.app.ui import create_ui
        demo = create_ui(chatbot)
        
        logger.info("홍익지기 챗봇 시작")
        demo.queue()  # 큐 활성화
        demo.launch(
            server_name="0.0.0.0",
            server_port=7861,
            share=True
        )
        
    except Exception as e:
        logger.error(f"오류 발생: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 