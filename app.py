from flask import Flask, render_template, request, jsonify
import sys
import os

# 모듈 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

# 챗봇 모듈 임포트 (src에서 hongikjiki로 변경)
from hongikjiki.utils import setup_logging
from hongikjiki.text_processor import JungbubTextProcessor
from hongikjiki.vector_store import JungbubVectorStore
from hongikjiki.chatbot import HongikJikiBot

from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')
DATA_DIR = os.getenv('DATA_DIR')
CHATBOT_NAME = os.getenv('CHATBOT_NAME', 'Hongik-Jiki')
DEVELOPER_NAME = os.getenv('DEVELOPER_NAME', '조성우')

# Flask 앱 생성
app = Flask(__name__)

# 챗봇 초기화
logger = setup_logging()
text_processor = JungbubTextProcessor()
vector_store = JungbubVectorStore(
    embedding_model_name=EMBEDDING_MODEL,
    persist_directory="./chroma_db"
)
chatbot = HongikJikiBot(vector_store)

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

if __name__ == '__main__':
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
            sys.exit(1)
        
        # 문서 분할
        chunks = text_processor.split_documents(documents)
        
        # 벡터 데이터베이스에 추가
        vector_store.add_documents(chunks)
    
    app.run(debug=True, host='0.0.0.0', port=8080)