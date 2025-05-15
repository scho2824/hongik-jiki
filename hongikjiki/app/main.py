# hongikjiki/app/main.py
import os
import sys
import traceback
from pathlib import Path
from hongikjiki.utils.logging_setup import setup_logging
from hongikjiki.app.config import OPENAI_API_KEY
from hongikjiki.app.ui import create_ui
from hongikjiki.core.chatbot import HongikJikiChatbot

ROOT_DIR = Path(__file__).resolve().parents[2]

# 로깅 설정
logger = setup_logging()
logger.info("홍익지기 챗봇 시작")

# API 키 검증
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
    print("오류: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
    print("다음 방법 중 하나로 API 키를 설정하세요:")
    print("1. export OPENAI_API_KEY=your_api_key")
    print("2. .env 파일에 OPENAI_API_KEY=your_api_key 추가")
    sys.exit(1)

def init_modules():
    """필요한 모듈 초기화"""
    try:
        # 홍익지기 모듈 임포트
        from hongikjiki.langchain_integration.llm import get_llm
        from hongikjiki.modules.vector_store.embeddings import get_embeddings
        from hongikjiki.modules.vector_store.chroma_store import ChromaVectorStore
        
        # 벡터 스토어 초기화
        try:
            import chromadb
            from chromadb.config import Settings
            from hongikjiki.app.config import PERSIST_DIR, COLLECTION_NAME
            
            # 벡터 스토어 초기화 로직
            # ...
            print("벡터 스토어 로드 시작...")
            
            # 임베딩 모델 초기화
            embeddings = get_embeddings("openai", model="text-embedding-ada-002", api_key=OPENAI_API_KEY)
            
            # 벡터 스토어 초기화
            vector_store = ChromaVectorStore(
                collection_name=COLLECTION_NAME,
                persist_directory=str(PERSIST_DIR),
                embeddings=embeddings
            )
            
            # 문서 수 확인
            doc_count = vector_store.count()
            logger.info(f"벡터 스토어 로드 성공: {doc_count}개 문서")
            print(f"벡터 스토어 로드 성공: {doc_count}개 문서")
            
        except Exception as e:
            logger.error(f"벡터 스토어 로드 실패: {e}")
            logger.error(traceback.format_exc())
            print(f"벡터 스토어 로드 실패: {e}")
            print("간단 모드로 전환합니다")
            vector_store = None
        
        # LLM 초기화
        llm = get_llm(llm_type="openai", model="gpt-4o", api_key=OPENAI_API_KEY)
        logger.info("LLM 초기화 성공")
        print("LLM 초기화 성공")
        
        # 태그 시스템 로드
        try:
            from hongikjiki.modules.tagging.tag_schema import TagSchema
            from hongikjiki.modules.tagging.tag_extractor import TagExtractor
            
            tag_schema_path = ROOT_DIR / "data" / "config" / "tag_schema.yaml"
            tag_pattern_path = ROOT_DIR / "data" / "config" / "tag_patterns.json"
            
            if tag_schema_path.exists() and tag_pattern_path.exists():
                tag_schema = TagSchema(str(tag_schema_path))
                tag_extractor = TagExtractor(tag_schema, 0.5)
                logger.info("태그 시스템 로드 완료")
                print("태그 시스템 로드 완료")
            else:
                logger.warning(f"태그 파일이 없습니다: {tag_schema_path} 또는 {tag_pattern_path}")
                tag_schema = None
                tag_extractor = None
        except Exception as e:
            logger.error(f"태그 시스템 로드 실패: {e}")
            logger.error(traceback.format_exc())
            tag_schema = None
            tag_extractor = None
            
        # 챗봇 인스턴스 생성
        chatbot = HongikJikiChatbot(llm, vector_store, tag_extractor)
        
        return chatbot
            
    except Exception as e:
        logger.error(f"모듈 초기화 오류: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

def main():
    """애플리케이션 메인 함수"""
    # 모듈 초기화
    chatbot = init_modules()
    
    # UI 생성
    demo = create_ui(chatbot)
    
    # 앱 실행
    try:
        logger.info("애플리케이션 실행 시작")
        print("\n홍익지기 챗봇 실행 준비 완료! 웹 브라우저가 잠시 후 열립니다...")
        
        demo.launch(
            server_name="127.0.0.1",
            share=False,
            quiet=True
        )
    except KeyboardInterrupt:
        logger.info("사용자가 애플리케이션을 종료했습니다")
        print("\n애플리케이션이 종료되었습니다")
    except Exception as e:
        logger.error(f"애플리케이션 실행 중 오류 발생: {e}")
        logger.error(traceback.format_exc())
        print(f"오류: {e}")
        print("자세한 내용은 로그를 확인하세요")

if __name__ == "__main__":
    main()