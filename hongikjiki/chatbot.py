"""
hongikjiki.chatbot

홍익지기 챗봇 통합 모듈.
이 모듈은 핵심 챗봇 클래스를 가져와서 필요한 모든 의존성과 함께 구성합니다.
최종 사용자(CLI/웹 인터페이스)가 직접 사용하기 위한 래퍼 클래스입니다.
"""

import os
import logging
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logger = logging.getLogger(__name__)

# 필요한 컴포넌트 임포트
from hongikjiki.core.chatbot import HongikJikiChatbot
from hongikjiki.modules.vector_store.chroma_store import ChromaVectorStore
from hongikjiki.modules.vector_store.embeddings import get_embeddings
from hongikjiki.langchain_integration.llm import get_llm

# 선택적 태그 관련 모듈 임포트
try:
    from hongikjiki.modules.tagging.tag_schema import TagSchema
    from hongikjiki.modules.tagging.tag_extractor import TagExtractor
    HAS_TAG_MODULES = True
except ImportError:
    HAS_TAG_MODULES = False
    logger.warning("태그 추출기 모듈을 찾을 수 없습니다. 태그 추출 기능 없이 계속합니다.")


class HongikJikiBot:
    """
    홍익지기 챗봇의 완전히 구성된 래퍼 클래스.
    
    이 클래스는 코어 챗봇 엔진을 감싸고 필요한 모든 의존성을 구성합니다.
    CLI 또는 웹 인터페이스에서 직접 사용하기 위한 인터페이스를 제공합니다.
    """
    
    def __init__(self, 
                 vector_store: Optional[ChromaVectorStore] = None,
                 data_dir: Optional[str] = None,
                 embedding_model: Optional[str] = None,
                 llm_type: str = "openai",
                 api_key: Optional[str] = None):
        """
        HongikJikiBot 초기화
        
        Args:
            vector_store: 기존 벡터 저장소 (없으면 자동 생성)
            data_dir: 데이터 디렉토리 경로 (환경 변수에서 로드하지 않을 경우)
            embedding_model: 임베딩 모델 이름 (환경 변수에서 로드하지 않을 경우)
            llm_type: 언어 모델 타입 ("openai" 또는 "clova")
            api_key: API 키 (환경 변수에서 로드하지 않을 경우)
        """
        # 환경변수 로드
        self.embedding_model = embedding_model or os.getenv('EMBEDDING_MODEL', 'openai')
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.data_dir = data_dir or os.getenv('DATA_DIR', 'data/jungbub_teachings')
        
        # API 키 확인
        if not self.api_key and llm_type == "openai":
            logger.warning("OPENAI_API_KEY가 설정되지 않았습니다.")
        
        # 벡터 스토어 설정
        self.vector_store = vector_store
        if not self.vector_store:
            # 벡터 스토어 생성
            embeddings = get_embeddings(model_name=self.embedding_model)
            self.vector_store = ChromaVectorStore(
                embeddings=embeddings,
                persist_directory="./chroma_db"
            )
        
        # LLM 초기화
        self.llm = get_llm(llm_type, api_key=self.api_key)
        
        # 태그 추출기 초기화 (선택적)
        self.tag_extractor = None
        if HAS_TAG_MODULES:
            try:
                tag_schema_path = "data/config/tag_schema.yaml"
                tag_schema = TagSchema(tag_schema_path)
                self.tag_extractor = TagExtractor(tag_schema)
                logger.info("태그 추출기 초기화 완료")
            except Exception as e:
                logger.warning(f"태그 추출기 초기화 실패: {e}")
        
        # 코어 챗봇 초기화
        self.chatbot = HongikJikiChatbot(
            llm=self.llm,
            vector_store=self.vector_store,
            tag_extractor=self.tag_extractor
        )
        
        logger.info("홍익지기 챗봇 통합 시스템 초기화 완료")
    
    def get_response(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        사용자 질문에 대한 응답 생성 - 간단한 인터페이스
        
        Args:
            message: 사용자 질문
            history: 대화 이력 (선택적)
            
        Returns:
            str: 챗봇 응답 텍스트
        """
        response_data = self.chatbot.answer_question(message, history or [])
        
        # 응답 형식 확인 및 처리
        if isinstance(response_data, dict):
            return response_data.get("answer", "응답을 찾을 수 없습니다.")
        else:
            return str(response_data)
    
    def get_related_questions(self) -> List[Dict[str, str]]:
        """
        현재 관련 질문 목록 반환
        
        Returns:
            List[Dict]: 관련 질문 정보 리스트
        """
        return self.chatbot.get_related_questions()
    
    def answer_question(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        사용자 질문에 대한 응답 생성 - 상세한 인터페이스
        
        Args:
            message: 사용자 질문
            history: 대화 이력 (선택적)
            
        Returns:
            Dict: 응답 데이터 (답변, 파일 경로 등)
        """
        return self.chatbot.answer_question(message, history or [])


def create_chatbot(vector_store=None, data_dir=None, embedding_model=None, llm_type="openai", api_key=None) -> HongikJikiBot:
    """
    홍익지기 챗봇 인스턴스 생성 헬퍼 함수
    
    Args:
        vector_store: 기존 벡터 저장소 (없으면 자동 생성)
        data_dir: 데이터 디렉토리 경로 (환경 변수에서 로드하지 않을 경우)
        embedding_model: 임베딩 모델 이름 (환경 변수에서 로드하지 않을 경우)
        llm_type: 언어 모델 타입 ("openai" 또는 "clova")
        api_key: API 키 (환경 변수에서 로드하지 않을 경우)
        
    Returns:
        HongikJikiBot: 초기화된 챗봇 인스턴스
    """
    return HongikJikiBot(
        vector_store=vector_store,
        data_dir=data_dir,
        embedding_model=embedding_model,
        llm_type=llm_type,
        api_key=api_key
    )


# Backward compatibility alias for legacy imports
HongikJikiChatBot = HongikJikiBot