from flask import Flask, render_template, request, jsonify
import sys
import os
import logging

# 환경변수 로드
from dotenv import load_dotenv
load_dotenv()

# 환경변수에서 설정값 가져오기
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
DATA_DIR = os.getenv('DATA_DIR', './data/jungbub_teachings')
CHATBOT_NAME = os.getenv('CHATBOT_NAME', 'Hongik-Jiki')
DEVELOPER_NAME = os.getenv('DEVELOPER_NAME', '조성우')

# 모듈 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

# 챗봇 모듈 임포트
from hongikjiki.utils import setup_logging
from hongikjiki.text_processing.document_processor import DocumentProcessor
from hongikjiki.text_processing.document_loader import DocumentLoader
from hongikjiki.vector_store.chroma_store import ChromaVectorStore
from hongikjiki.chatbot import HongikJikiChatBot
from hongikjiki.langchain_integration.llm import get_llm

# Flask 앱 생성
app = Flask(__name__)

# 로깅 설정
logger = setup_logging()
logger.info(f"{CHATBOT_NAME} 웹 서버 초기화 시작")

def load_documents_recursive(directory):
    """
    디렉토리와 하위 디렉토리에서 문서를 재귀적으로 로드하는 함수
    
    Args:
        directory: 문서를 로드할 최상위 디렉토리
        
    Returns:
        list: 로드된 문서 목록
    """
    loader = DocumentLoader()
    loaded_docs = []
    
    # 디렉토리가 존재하는지 확인
    if not os.path.exists(directory):
        logger.warning(f"디렉토리가 존재하지 않습니다: {directory}")
        return loaded_docs
    
    # 디렉토리 내 모든 파일 처리
    for item in os.listdir(directory):
        # .으로 시작하는 숨김 파일 건너뛰기
        if item.startswith('.'):
            continue
            
        item_path = os.path.join(directory, item)
        
        if os.path.isfile(item_path):
            # 파일인 경우 문서 로드
            doc_data = loader.load_document(item_path)
            if doc_data:
                loaded_docs.append(doc_data)
                logger.info(f"문서 로드 완료: {item}")
        elif os.path.isdir(item_path):
            # 하위 디렉토리인 경우 재귀적으로 처리
            logger.info(f"하위 디렉토리 처리 중: {item}")
            subdirectory_docs = load_documents_recursive(item_path)
            loaded_docs.extend(subdirectory_docs)
    
    return loaded_docs

try:
    # 벡터 저장소 및 디렉토리 설정
    persist_directory = "./data/vector_store"
    os.makedirs(persist_directory, exist_ok=True)
    
    # 벡터 저장소 생성
    from hongikjiki.vector_store.embeddings import get_embeddings
    embeddings = get_embeddings("huggingface", model_name=EMBEDDING_MODEL)
    
    # 벡터 스토어 경로를 출력해 문제를 진단
    logger.info(f"디버그: 벡터 저장소 경로 = {os.path.abspath(persist_directory)}")
    
    vector_store = ChromaVectorStore(
        collection_name="hongikjiki_documents",
        persist_directory=persist_directory,
        embeddings=embeddings
    )
    
    # 데이터베이스에 문서 수 확인
    collection_info = vector_store.collection.count()
    logger.info(f"현재 데이터베이스 문서 수: {collection_info}")
    # Ingestion (chunking, tagging, QA, vector build) is handled by separate pipeline scripts.
    
    # LLM 인스턴스화 (HongikJikiChatBot 인스턴스화 전에)
    llm = get_llm("openai", model="gpt-4o", temperature=0.7)
    # 챗봇 초기화
    chatbot = HongikJikiChatBot(
        persist_directory=persist_directory,
        embedding_type="huggingface",  # 또는 "openai"
        llm_type="openai",  # 또는 다른 LLM 유형
        collection_name="hongikjiki_documents",
        embedding_kwargs={
            "model_name": EMBEDDING_MODEL
        },
        llm=llm  # Pass the pre-initialized llm
    )
    # Load and assign TagSchema so chatbot.tag_schema exists
    from hongikjiki.tagging.tag_schema import TagSchema
    tag_schema_path = "data/config/tag_schema.yaml"
    tag_schema = TagSchema(tag_schema_path)
    chatbot.tag_schema = tag_schema
    logger.info(f"TagSchema loaded for chatbot: {tag_schema_path}")
    # Ensure tag extractor is initialized with the correct patterns path
    from hongikjiki.tagging.tag_extractor import TagExtractor
    tag_patterns_path = "data/config/tag_patterns.json"
    chatbot.tag_extractor = TagExtractor(chatbot.tag_schema, tag_patterns_path)
    logger.info(f"TagExtractor re-initialized with patterns: {tag_patterns_path}")
    logger.info("챗봇 초기화 완료 (단일 호출)")
    
except Exception as e:
    logger.error(f"벡터 저장소 초기화 오류: {e}")
    logger.error("벡터 저장소 초기화 실패. 서버를 종료합니다.")
    import traceback
    logger.error(traceback.format_exc())  # 상세 오류 정보 기록
    sys.exit(1)

@app.route('/')
def index():
    return render_template('index.html', chatbot_name=CHATBOT_NAME, developer_name=DEVELOPER_NAME)

@app.route('/ask', methods=['POST'])
def ask():
    try:
        # 질문 및 추가 매개변수 가져오기
        data = request.json
        question = data.get('question', '')
        
        if not question:
            return jsonify({"error": "질문이 비어있습니다."}), 400
        
        # 선택된 태그 가져오기 (있는 경우)
        selected_tags = data.get('selected_tags', [])
        
        logger.info(f"사용자 질문: {question}")
        if selected_tags:
            logger.info(f"선택된 태그: {selected_tags}")
        
        # 질문에서 태그 추출
        extracted_tags = []
        if hasattr(chatbot, 'tag_extractor') and chatbot.tag_extractor:
            extracted_tags = chatbot.tag_extractor.extract_tags_from_query(question)
            logger.info(f"추출된 태그: {extracted_tags}")
        
        # 챗봇 응답 생성
        response = chatbot.chat(question)
        
        # 관련 질문 추천 가져오기 (해당 메서드가 있는 경우)
        suggested_questions = []
        if hasattr(chatbot, 'get_related_questions'):
            suggested_questions = chatbot.get_related_questions(question)
        
        # 추천 문서 가져오기 (해당 메서드가 있는 경우)
        recommended_documents = []
        if hasattr(chatbot, 'get_recommended_documents'):
            recommended_documents = chatbot.get_recommended_documents(question)

        # 응답 데이터 구성
        response_data = {
            "response": response,
            "extracted_tags": extracted_tags,
            "suggested_questions": suggested_questions,
            "recommended_documents": recommended_documents
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"응답 생성 오류: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": f"오류가 발생했습니다: {str(e)}"}), 500

if __name__ == '__main__':
    # 호스트와 포트 설정
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 8080))
    debug = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    logger.info(f"서버 시작: host={host}, port={port}, debug={debug}")
    app.run(debug=debug, host=host, port=port)