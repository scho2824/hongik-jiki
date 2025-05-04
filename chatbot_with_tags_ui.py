"""
태그 표시 기능이 포함된 홍익지기 챗봇 웹 인터페이스
"""
from flask import Flask, render_template, request, jsonify
import os
import logging
import sys

# 로깅 설정
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("HongikJikiWeb")

# 프로젝트 루트 경로 추가
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# 홍익지기 챗봇 및 태그 관련 모듈 임포트
from hongikjiki.chatbot import HongikJikiChatBot
from hongikjiki.tagging.tag_schema import TagSchema

# Flask 앱 생성
app = Flask(__name__)

# 챗봇 초기화 (태그 기능 활성화)
chatbot = HongikJikiChatBot(use_tags=True)

# 태그 스키마 로드 (태그 정보 표시용)
try:
    tag_schema_path = os.path.join(project_root, 'data', 'config', 'tag_schema.yaml')
    tag_schema = TagSchema(tag_schema_path)
    logger.info(f"태그 스키마 로드 완료: {len(tag_schema.tags)} 개의 태그")
    
    # 카테고리별 태그 정리 (UI 표시용)
    categories = {}
    for tag_name, tag in tag_schema.tags.items():
        category = tag.category
        if category not in categories:
            categories[category] = []
        categories[category].append({
            "name": tag_name,
            "emoji": tag.emoji or "",
            "description": tag.description or "",
            "parent": tag.parent
        })
    
    logger.info(f"카테고리별 태그 정리 완료: {len(categories)} 개의 카테고리")
    
except Exception as e:
    logger.error(f"태그 스키마 로드 오류: {e}")
    categories = {}

# 메인 페이지 라우트
@app.route('/')
def index():
    return render_template('index.html', 
                          categories=categories,
                          chatbot_name="홍익지기")

# 챗봇 질문 처리 라우트
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
        logger.info(f"선택된 태그: {selected_tags}")
        
        # 질문에서 태그 추출 (디버깅용)
        extracted_tags = []
        if chatbot.tag_extractor:
            extracted_tags = chatbot.tag_extractor.extract_tags_from_query(question)
            logger.info(f"추출된 태그: {extracted_tags}")
        
        # 챗봇 응답 생성
        response = chatbot.chat(question)
        
        # 관련 태그 및 추천 질문 가져오기
        related_tags = chatbot.get_related_tags(question)
        suggested_questions = chatbot.get_related_questions(question)
        
        # 응답 데이터 구성
        response_data = {
            "response": response,
            "extracted_tags": extracted_tags,
            "related_tags": related_tags,
            "suggested_questions": suggested_questions
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"응답 생성 오류: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": f"오류가 발생했습니다: {str(e)}"}), 500

# 태그 검색 라우트
@app.route('/search_by_tag', methods=['POST'])
def search_by_tag():
    try:
        data = request.json
        tag = data.get('tag', '')
        
        if not tag:
            return jsonify({"error": "태그가 비어있습니다."}), 400
        
        # 태그 기반 검색
        results = chatbot.search_documents_by_tag(tag, k=5)
        
        # 결과 데이터 구성
        search_data = {
            "tag": tag,
            "results": results
        }
        
        return jsonify(search_data)
        
    except Exception as e:
        logger.error(f"태그 검색 오류: {e}")
        return jsonify({"error": f"오류가 발생했습니다: {str(e)}"}), 500

# 정보 페이지 라우트
@app.route('/about')
def about():
    return render_template('about.html', 
                          chatbot_name="홍익지기",
                          tag_count=len(tag_schema.tags) if 'tag_schema' in locals() else 0)

# 태그 목록 라우트
@app.route('/tags')
def tags():
    return render_template('tags.html', 
                          categories=categories)

# 홍익지기 챗봇의 get_related_tags 메서드 구현 (chatbot.py에 추가)
def get_related_tags(self, query: str, max_tags: int = 5) -> list:
    """
    쿼리와 관련된 태그 추출
    
    Args:
        query: 사용자 질문
        max_tags: 최대 태그 수
        
    Returns:
        List[Dict]: 태그 정보 딕셔너리 리스트
    """
    if not self.use_tags or not self.tag_extractor:
        return []
    
    # 쿼리에서 태그 추출
    query_tags = self.tag_extractor.extract_tags_from_query(query)
    
    # 태그 스키마에서 태그 정보 가져오기
    tag_info = []
    for tag in query_tags[:max_tags]:
        tag_obj = self.tag_schema.get_tag(tag)
        if tag_obj:
            tag_info.append({
                "name": tag,
                "category": tag_obj.category,
                "emoji": tag_obj.emoji or "",
                "description": tag_obj.description or ""
            })
    
    return tag_info

# 홍익지기 챗봇의 search_documents_by_tag 메서드 구현 (chatbot.py에 추가)
def search_documents_by_tag(self, tag: str, k: int = 5) -> list:
    """
    특정 태그로 문서 검색
    
    Args:
        tag: 검색할 태그
        k: 최대 결과 수
        
    Returns:
        List[Dict]: 검색 결과 리스트
    """
    if not self.use_tags:
        return []
    
    # 태그 기반 검색 수행
    try:
        if hasattr(self.vector_store, 'search_with_tags'):
            results = self.vector_store.search_with_tags("", [tag], tag_boost=1.0, k=k)
        else:
            # 태그 기반 검색이 지원되지 않는 경우 태그를 쿼리로 사용
            results = self.vector_store.search(f"tag:{tag}", k=k)
        
        # 결과 가공
        search_results = []
        for result in results:
            # 메타데이터에서 제목 및 소스 정보 추출
            metadata = result.get("metadata", {})
            title = metadata.get("title", "제목 없음")
            source = metadata.get("source", "")
            lecture = metadata.get("lecture_number", "")
            
            # 청크 일부만 사용 (너무 길면 UI에 표시하기 어려움)
            content = result.get("content", "")
            if len(content) > 200:
                content = content[:200] + "..."
            
            search_results.append({
                "title": title,
                "content": content,
                "source": f"{source} {lecture}강" if lecture else source,
                "score": result.get("score", 0)
            })
        
        return search_results
        
    except Exception as e:
        logger.error(f"태그 기반 검색 오류: {e}")
        return []

if __name__ == '__main__':
    # 환경 변수로부터 호스트와 포트 설정
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 8080))
    debug = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    # 서버 시작
    logger.info(f"홍익지기 웹 서버 시작: host={host}, port={port}, debug={debug}")
    app.run(debug=debug, host=host, port=port)