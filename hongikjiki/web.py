"""
Hongik-Jiki 웹 서버 인터페이스
"""

import os
import sys
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# 패키지 내 모듈 임포트
from .utils import setup_logging
from .text_processor import JungbubTextProcessor
from .vector_store import JungbubVectorStore
from .chatbot import HongikJikiBot

# 전역 변수로 챗봇 인스턴스 정의
logger = None
text_processor = None
vector_store = None
chatbot = None

def init_database():
    """데이터베이스 초기화 함수"""
    global vector_store, text_processor, logger
    
    # 환경변수 로드
    DATA_DIR = os.getenv('DATA_DIR')
    
    # 데이터베이스에 문서 수 확인
    collection_info = vector_store.collection.count()
    logger.info(f"현재 데이터베이스 문서 수: {collection_info}")
    
    # 문서가 없으면 문서 로드 및 처리
    if collection_info == 0:
        logger.info("데이터베이스가 비어 있습니다. 문서 로드 및 처리를 시작합니다.")
        
        # 문서 로드
        documents = text_processor.load_documents(DATA_DIR)
        
        if not documents:
            logger.warning(f"{DATA_DIR} 폴더에 문서가 없습니다.")
            print(f"오류: {DATA_DIR} 폴더에 정법 문서를 찾을 수 없습니다.")
            print("정법 문서를 data/jungbub_teachings 폴더에 추가한 후 다시 시도하세요.")
            return False
        
        # 문서 분할
        chunks = text_processor.split_documents(documents)
        
        # 벡터 데이터베이스에 추가
        vector_store.add_documents(chunks)
    
    return True

def create_app():
    """Flask 앱 생성 및 초기화 함수"""
    app = Flask(__name__, template_folder='templates')
    
    # 환경변수 로드
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')
    CHATBOT_NAME = os.getenv('CHATBOT_NAME', 'Hongik-Jiki')
    DEVELOPER_NAME = os.getenv('DEVELOPER_NAME', '조성우')
    
    # 챗봇 초기화
    global logger, text_processor, vector_store, chatbot
    
    logger = setup_logging()
    text_processor = JungbubTextProcessor()
    vector_store = JungbubVectorStore(
        embedding_model_name=EMBEDDING_MODEL,
        persist_directory="./chroma_db"
    )
    
    # 데이터베이스 초기화
    init_database()
    
    # 챗봇 초기화
    chatbot = HongikJikiBot(vector_store)
    
    # 라우트 설정
    @app.route('/')
    def index():
        return render_template('index.html', chatbot_name=CHATBOT_NAME, developer_name=DEVELOPER_NAME)
    
    @app.route('/ask', methods=['POST'])
    def ask():
        user_question = request.json.get('question', '')
        if not user_question:
            return jsonify({'error': '질문이 비어있습니다.'}), 400
            
        response = chatbot.get_response(user_question)
        return jsonify({'response': response})
    
    return app

# Flask 앱 생성
app = create_app()

def main():
    """웹 서버 실행 함수"""
    # 호스트와 포트 설정
    host = os.getenv('FLASK_RUN_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_RUN_PORT', 8080))
    debug = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    main()