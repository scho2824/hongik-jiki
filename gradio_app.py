# gradio_app.py
import os
import sys
import gradio as gr
import tempfile
import time
import json
import logging
import traceback
from pathlib import Path
from dotenv import load_dotenv
from packaging import version as pkg_version

# 직접 로깅 설정 함수 구현
def setup_logging():
    """로깅 설정 강화"""
    logger = logging.getLogger("HongikJikiChatBot")
    logger.setLevel(logging.DEBUG)

    # 이미 핸들러가 있으면 제거
    if logger.handlers:
        logger.handlers.clear()

    # 파일 핸들러
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, "chatbot.log"))
    file_handler.setLevel(logging.DEBUG)

    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # 포매터
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 핸들러 추가
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# 로깅 설정
logger = setup_logging()
logger.info("홍익지기 챗봇 시작")

# 환경 변수 로드
load_dotenv()
logger.info("환경 변수 로드 완료")

# API 키 확인
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    error_msg = "OPENAI_API_KEY 환경 변수가 설정되지 않았습니다."
    logger.error(error_msg)
    print(f"오류: {error_msg}")
    print("다음 방법 중 하나로 API 키를 설정하세요:")
    print("1. export OPENAI_API_KEY=your_api_key")
    print("2. .env 파일에 OPENAI_API_KEY=your_api_key 추가")
    sys.exit(1)

logger.info(f"API 키 확인: {api_key[:5]}...{api_key[-5:]}")
print(f"API 키 확인: {api_key[:5]}...{api_key[-5:]}")

# Gradio 버전 체크
gradio_version = pkg_version.parse(gr.__version__)
logger.info(f"Gradio 버전: {gr.__version__}")
print(f"Gradio 버전: {gr.__version__}")

# 버전에 따른 설정 조정
USE_MESSAGE_FORMAT = gradio_version >= pkg_version.parse("3.32.0")
logger.info(f"메시지 형식 사용: {USE_MESSAGE_FORMAT}")
print(f"메시지 형식 사용: {USE_MESSAGE_FORMAT}")

# 홍익지기 모듈 임포트
try:
    from hongikjiki.langchain_integration.llm import get_llm
    from hongikjiki.vector_store.embeddings import get_embeddings
    from hongikjiki.vector_store.chroma_store import ChromaVectorStore
except ImportError as e:
    logger.error(f"모듈 임포트 오류: {e}")
    sys.exit(1)

# 전역 변수
related_question_buttons = []

# 태그 시스템 로드
try:
    from hongikjiki.tagging.tag_schema import TagSchema
    from hongikjiki.tagging.tag_extractor import TagExtractor
    
    tag_schema_path = "data/config/tag_schema.yaml"
    tag_pattern_path = "data/config/tag_patterns.json"
    
    if os.path.exists(tag_schema_path) and os.path.exists(tag_pattern_path):
        tag_schema = TagSchema(tag_schema_path)
        tag_extractor = TagExtractor(tag_schema, tag_pattern_path)
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

# 관련 질문 데이터 로드
related_questions_map = {}
try:
    qa_file_path = "data/qa/high_insight_qa_dataset_formatted_related.json"
    if os.path.exists(qa_file_path):
        with open(qa_file_path, "r", encoding="utf-8") as f:
            related_questions_map = json.load(f)
        logger.info("관련 질문 데이터 로드 완료")
    else:
        logger.warning(f"관련 질문 파일이 없습니다: {qa_file_path}")
except Exception as e:
    logger.error(f"관련 질문 데이터 로드 오류: {e}")

# 벡터 스토어 초기화
vector_store = None
try:
    logger.info("벡터 스토어 로드 시작...")
    print("벡터 스토어 로드 시작...")

    # Chroma v0.6.0 호환을 위한 방식으로 수정
    import chromadb
    from chromadb.config import Settings
    
    persist_dir = "data/vector_store"
    collection_name = "hongikjiki_jungbub"
    
    # 디렉토리 확인 및 생성
    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir, exist_ok=True)
        logger.info(f"벡터 스토어 디렉토리 생성: {persist_dir}")
    
    # 클라이언트 생성
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )
    
    # 컬렉션 확인 - 0.6.0 방식으로 수정
    collections = client.list_collections()
    collection_exists = False
    for collection in collections:
        if collection.name == collection_name:
            collection_exists = True
            break
    
    # 컬렉션 가져오기 또는 생성
    if collection_exists:
        collection = client.get_collection(name=collection_name)
        logger.info(f"기존 컬렉션 로드: {collection_name}")
    else:
        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"새 컬렉션 생성: {collection_name}")
    
    # 임베딩 모델 초기화
    embeddings = get_embeddings("openai", model="text-embedding-ada-002", api_key=api_key)
    
    # ChromaVectorStore 초기화
    vector_store = ChromaVectorStore(
        collection_name=collection_name,
        persist_directory=persist_dir,
        embeddings=embeddings
    )
    
    # 문서 수 확인
    doc_count = vector_store.count()
    logger.info(f"벡터 스토어 로드 성공: {doc_count}개 문서")
    print(f"벡터 스토어 로드 성공: {doc_count}개 문서")
    
    if doc_count == 0:
        logger.warning("벡터 스토어에 문서가 없습니다.")
        print("벡터 스토어에 문서가 없습니다. 데이터 로드가 필요할 수 있습니다.")
except Exception as e:
    logger.error(f"벡터 스토어 로드 실패: {e}")
    logger.error(traceback.format_exc())
    print(f"벡터 스토어 로드 실패: {e}")
    print("간단 모드로 전환합니다")
    vector_store = None

# LLM 초기화
llm = None
try:
    logger.info("LLM 초기화 시작...")
    llm = get_llm(llm_type="openai", model="gpt-4o", api_key=api_key)
    logger.info("LLM 초기화 성공")
    print("LLM 초기화 성공")
except Exception as e:
    logger.error(f"LLM 초기화 실패: {e}")
    logger.error(traceback.format_exc())
    print(f"LLM 초기화 실패: {e}")
    sys.exit(1)  # LLM은 필수이므로 실패 시 종료

# 챗봇 체인 생성
chatbot_chain = None
if vector_store and vector_store.count() > 0:
    try:
        from hongikjiki.langchain_integration.chain import get_chatbot_chain
        logger.info("챗봇 체인 초기화 시작...")
        chatbot_chain = get_chatbot_chain(llm=llm, vector_store=vector_store)
        logger.info("챗봇 체인 초기화 성공")
        print("챗봇 체인 초기화 성공")
    except Exception as e:
        logger.error(f"챗봇 체인 초기화 실패: {e}")
        logger.error(traceback.format_exc())
        print(f"챗봇 체인 초기화 실패: {e}")
        print("간단 모드로 전환합니다")
        vector_store = None  # 체인 초기화 실패 시 벡터 스토어 무효화
else:
    logger.info("벡터 스토어 없음: 간단 모드로 실행")
    print("벡터 스토어 없음: 간단 모드로 실행")

# 응답 포맷팅 함수
def format_response(results, answer, extracted_tags=None):
    """
    챗봇 응답을 포맷팅하는 함수
    
    Args:
        results: 벡터 검색 결과
        answer: LLM 생성 응답
        extracted_tags: 질문에서 추출한 태그 (선택적)
        
    Returns:
        포맷팅된 응답
    """
    # 태그 추출
    tags = set(extracted_tags or [])
    quoted_insights = []
    source_info = []
    
    # 검색 결과에서 정보 추출
    for i, result in enumerate(results):
        # 메타데이터에서 태그 추출
        metadata = result.get('metadata', {})
        if isinstance(metadata, dict) and 'tags' in metadata:
            result_tags = metadata['tags']
            if isinstance(result_tags, list):
                tags.update(result_tags)
            elif isinstance(result_tags, str):
                tags.update([t.strip() for t in result_tags.split(',')])
        
        # 강의 정보 추출
        lecture_id = metadata.get('lecture_id', '')
        lecture_title = metadata.get('lecture_title', '')
        
        if lecture_id or lecture_title:
            info = f"[문서 {i+1}]"
            if lecture_title:
                info += f" 「{lecture_title}」"
            if lecture_id:
                info += f" (강의 번호: {lecture_id})"
            source_info.append(info)
        
        # 인용문 추출
        content = result.get('content', '')
        if content and ":" in content:  # "Q: ... A: ..." 포맷 처리
            parts = content.split("A: ")
            if len(parts) > 1:
                answer_part = parts[1].strip()
                sentences = answer_part.split(".")
                if sentences:
                    sentence = sentences[0].strip()
                    if 10 <= len(sentence) <= 100:
                        quoted_insights.append(f'"{sentence}"')
    
    # 응답 포맷팅
    formatted = f"{answer}\n\n"
    
    # 출처 정보 추가
    if source_info:
        formatted += f"🔗 출처:\n" + "\n".join(source_info) + "\n\n"
    
    # 인용문 추가
    if quoted_insights:
        formatted += f"🔎 관련 인용:\n{quoted_insights[0]}\n\n"
    
    # 태그 추가
    if tags:
        tag_list = ' '.join([f"#{tag}" for tag in tags])
        formatted += f"🏷️ 관련 태그: {tag_list}"
    
    return formatted

# 관련 질문 생성 함수
def generate_related_questions(tags, current_question):
    """
    태그 기반 관련 질문 생성
    
    Args:
        tags: 태그 집합
        current_question: 현재 질문
        
    Returns:
        관련 질문 리스트
    """
    # 태그별 질문 맵핑
    tag_questions = {
        "감정": [
            "감정이 불안정한 이유는 무엇인가요?",
            "화가 날 때 어떻게 다스려야 할까요?",
            "자신의 감정을 이해하는 방법은 무엇인가요?"
        ],
        "관계": [
            "가족과의 갈등을 어떻게 해결할 수 있나요?",
            "인간관계에서 중요한 것은 무엇인가요?",
            "타인을 이해하는 방법은 무엇인가요?"
        ],
        "정신": [
            "마음과 정신의 차이는 무엇인가요?",
            "정신 수련을 어떻게 해야 하나요?",
            "정신이 흐트러질 때 집중하는 법은?"
        ],
        "수행": [
            "일상에서 어떻게 정법을 실천할 수 있나요?",
            "정법 수행은 무엇부터 시작해야 하나요?",
            "수행이 잘 되지 않을 때는 어떻게 해야 하나요?"
        ],
        "삶": [
            "참된 행복을 찾는 방법은 무엇인가요?",
            "인생의 목적을 어떻게 찾을 수 있을까요?",
            "의미 있는 삶을 사는 방법은 무엇인가요?"
        ],
        "자유": [
            "진정한 자유란 무엇인가요?",
            "자유와 책임의 관계는 무엇인가요?",
            "어떻게 자유로운 삶을 살 수 있을까요?"
        ],
        "본성": [
            "인간의 본성은 무엇인가요?",
            "본성을 회복하는 방법은 무엇인가요?",
            "본성과 습관의 차이는 무엇인가요?"
        ]
    }
    
    related = []
    # 태그 기반 질문 추가
    for tag in tags:
        if tag in tag_questions:
            for question in tag_questions[tag]:
                if question.lower() != current_question.lower() and question not in related:
                    related.append({
                        "question": question,
                        "insight": f"'{tag}' 관련 질문입니다."
                    })
    
    # 일반 질문 추가 (태그가 없거나 적을 경우)
    general_questions = [
        {"question": "정법이란 무엇인가요?", "insight": "정법의 기본 개념에 관한 질문입니다."},
        {"question": "자주독립의 의미는 무엇인가요?", "insight": "정신적 자립에 관한 질문입니다."},
        {"question": "어떻게 마음의 평화를 찾을 수 있나요?", "insight": "마음 수행에 관한 질문입니다."},
        {"question": "홍익인간이란 무엇인가요?", "insight": "홍익인간 철학에 관한 질문입니다."}
    ]
    
    # 관련 질문이 부족하면 일반 질문 추가
    if len(related) < 3:
        for q in general_questions:
            if q["question"].lower() != current_question.lower() and q not in related:
                related.append(q)
                if len(related) >= 3:
                    break
    
    # 최대 3개 반환
    return related[:3]

# 간단한 함수형 챗봇 (벡터 스토어가 없을 때 사용)
class SimpleChatbot:
    def __init__(self, llm):
        self.llm = llm

    def run(self, query):
        try:
            prompt = f"""
            당신은 정법 지식을 제공하는 홍익지기 인공지능 비서입니다.
            사용자의 질문에 대해 정확하고 도움이 되는 답변을 제공해야 합니다.
            
            정법에 대해 알고 있는 내용:
            - 정법은 천공 스승님께서 알려주신 우주 법칙에 대한 가르침입니다.
            - 홍익인간 이념과 관련이 있습니다.
            - 자연의 법칙과 조화, 인간 내면의 성장 등을 중요시합니다.
            
            다음 질문에 대해 정법의 관점에서 답변해주세요:
            {query}
            """
            answer = self.llm.generate(prompt)
            return {"answer": answer}
        except Exception as e:
            logger.error(f"간단 챗봇 응답 생성 오류: {e}")
            return {"answer": f"오류가 발생했습니다: {str(e)}"}

# 질문 처리 및 응답 함수
def answer_question(message, history):
    """
    사용자 질문에 답변 생성
    """
    global related_question_buttons
    
    logger.info(f"사용자 질문: {message}")
    
    # 질문에서 태그 추출
    extracted_tags = []
    if tag_extractor:
        try:
            extracted_tags = tag_extractor.extract_tags_from_query(message)
            logger.info(f"질문에서 추출한 태그: {extracted_tags}")
        except Exception as e:
            logger.warning(f"태그 추출 오류: {e}")
    
    # 벡터 스토어가 없으면 간단 모드 실행
    if not vector_store or vector_store.count() == 0:
        try:
            simple_bot = SimpleChatbot(llm)
            logger.info("간단 모드로 응답 생성 시작")
            response = simple_bot.run(message)
            answer = response.get('answer', '')
            
            # 관련 질문 생성
            related = generate_related_questions(extracted_tags, message)
            
            # 전역 변수에 저장
            related_question_buttons = related
            
            logger.info("간단 모드로 응답 생성 완료")
            
            # 임시 파일 생성 (다운로드용)
            try:
                temp_dir = tempfile.gettempdir()
                temp = tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".txt", dir=temp_dir, encoding="utf-8")
                temp.write(answer)
                temp.close()
                
                # Gradio 버전에 따라 반환 형식 조정
                if USE_MESSAGE_FORMAT:
                    history.append((message, answer))
                    return history, temp.name
                else:
                    return history + [(message, answer)], temp.name
            except Exception as e:
                logger.error(f"임시 파일 생성 오류: {e}")
                
                if USE_MESSAGE_FORMAT:
                    history.append((message, answer))
                    return history, None
                else:
                    return history + [(message, answer)], None
                
        except Exception as e:
            logger.error(f"간단 모드 응답 생성 오류: {e}")
            logger.error(traceback.format_exc())
            error_msg = f"오류가 발생했습니다: {str(e)}"
            
            if USE_MESSAGE_FORMAT:
                history.append((message, error_msg))
                return history, None
            else:
                return history + [(message, error_msg)], None
    
    # 정상 모드: 벡터 스토어 기반 챗봇 실행
    try:
        logger.info("벡터 스토어 기반 챗봇으로 응답 생성 시작")
        
        # 태그 기반 검색 실행
        if hasattr(vector_store, 'advanced_search') and extracted_tags:
            logger.info(f"태그 기반 고급 검색 실행: {extracted_tags}")
            results = vector_store.advanced_search(message, use_tags=True, k=3)
        else:
            logger.info("일반 벡터 검색 실행")
            results = vector_store.search(message, k=3)
        
        # 결과가 없으면 간단 모드로 전환
        if not results:
            logger.warning("검색 결과가 없습니다. 간단 모드로 전환합니다.")
            simple_bot = SimpleChatbot(llm)
            response = simple_bot.run(message)
            answer = response.get('answer', '')
            formatted_answer = f"{answer}\n\n⚠️ 관련 정법 문서를 찾지 못했습니다."
            
            if USE_MESSAGE_FORMAT:
                history.append((message, formatted_answer))
                return history, None
            else:
                return history + [(message, formatted_answer)], None
        
        # 컨텍스트 생성
        context = ""
        for i, doc in enumerate(results):
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})
            source_info = ""
            
            if isinstance(metadata, dict):
                lecture_id = metadata.get('lecture_id', '')
                title = metadata.get('lecture_title', '')
                
                if lecture_id or title:
                    source_info = f" [출처: "
                    if title:
                        source_info += f"{title}"
                    if lecture_id:
                        source_info += f" (강의 {lecture_id})"
                    source_info += "]"
            
            context += f"[문서 {i+1}]{source_info}\n{content}\n\n"
        
        # 프롬프트 생성
        prompt = f"""
        당신은 정법 지식을 제공하는 홍익지기 인공지능 비서입니다.
        사용자의 질문에 대해 정확하고 도움이 되는 답변을 제공해야 합니다.
        아래 제공된 정법 문서를 기반으로 질문에 답변하세요.
        
        ## 중요 지침:
        1. 제공된 정법 문서 내용만 사용하여 답변하세요.
        2. 문서에 없는 내용은 답변하지 마세요.
        3. 답변은 친절하고 이해하기 쉽게 작성하세요.
        4. 답변은 한국어로 작성하세요.
        
        ### 관련 정법 문서:
        {context}
        
        ### 사용자 질문:
        {message}
        
        ### 답변:
        """
        
        # LLM으로 답변 생성
        answer = llm.generate(prompt)
        
        # 응답 포맷팅
        formatted_answer = format_response(results, answer, extracted_tags)
        
        # 관련 질문 생성
        related = generate_related_questions(extracted_tags, message)
        
        # 전역 변수에 저장
        related_question_buttons = related
        
        logger.info("벡터 스토어 기반 응답 생성 완료")
        
        # 임시 파일 생성 (다운로드용)
        try:
            temp_dir = tempfile.gettempdir()
            logger.info(f"임시 파일 디렉토리: {temp_dir}")
            temp = tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".txt", dir=temp_dir, encoding="utf-8")
            temp.write(formatted_answer)
            temp.close()
            logger.info(f"응답 임시 파일 생성: {temp.name}")
            
            if USE_MESSAGE_FORMAT:
                history.append((message, formatted_answer))
                return history, temp.name
            else:
                return history + [(message, formatted_answer)], temp.name
        except Exception as e:
            logger.error(f"임시 파일 생성 오류: {e}")
            logger.error(traceback.format_exc())
            
            if USE_MESSAGE_FORMAT:
                history.append((message, formatted_answer))
                return history, None
            else:
                return history + [(message, formatted_answer)], None
        
    except Exception as e:
        logger.error(f"챗봇 실행 오류: {e}")
        logger.error(traceback.format_exc())
        error_msg = f"오류가 발생했습니다: {str(e)}"
        
        if USE_MESSAGE_FORMAT:
            history.append((message, error_msg))
            return history, None
        else:
            return history + [(message, error_msg)], None

# 관련 질문 클릭 핸들러
def handle_related_question(question_dict):
    """
    관련 질문 클릭 시 처리 함수
    """
    if isinstance(question_dict, dict) and "question" in question_dict:
        question = question_dict["question"]
    else:
        question = str(question_dict)
    return question, gr.update(value=question)

# 관련 질문 목록 가져오기
def get_related_questions():
    """관련 질문 목록 가져오기"""
    try:
        return related_question_buttons
    except Exception as e:
        logger.error(f"관련 질문 목록 가져오기 오류: {e}")
        return []

# 관련 질문 버튼 업데이트 함수
def update_question_buttons():
    """관련 질문 버튼 업데이트"""
    questions = get_related_questions()
    if not questions or len(questions) == 0:
        return (
            gr.update(visible=False, value=""),
            gr.update(visible=False, value=""),
            gr.update(visible=False, value="")
        )
    
    updates = []
    # 각 버튼 업데이트 (tooltip 제거)
    for i in range(3):
        if i < len(questions):
            q = questions[i]
            question = q.get("question", "관련 질문")
            updates.append(gr.update(visible=True, value=question))
        else:
            updates.append(gr.update(visible=False, value=""))
    
    return tuple(updates)

# Gradio 인터페이스 생성
with gr.Blocks(title="홍익지기 챗봇") as demo:
    gr.Markdown("# 🌕 홍익지기 챗봇")
    gr.Markdown("정법 강의를 기반으로 질문에 답하는 GPT 챗봇입니다.")
    gr.Markdown("삶의 방향, 감정, 사회, 영성 등에 대한 통찰을 얻어보세요.")
    
    # 버전에 따른 챗봇 인터페이스 선택
    if USE_MESSAGE_FORMAT:
        chatbot = gr.Chatbot(height=500)
    else:
        chatbot = gr.Chatbot(height=500)
    
    # 입력 및 버튼 영역
    with gr.Row():
        with gr.Column(scale=8):
            msg = gr.Textbox(
                placeholder="질문을 입력하세요...",
                label="질문",
                show_label=False
            )
        
        with gr.Column(scale=1):
            submit_btn = gr.Button("질문하기")
    
    # 파일 다운로드 영역
    download_file = gr.File(label="답변 다운로드", visible=False)
    
    # 예시 질문
    gr.Examples(
        examples=[
            "정법이란 무엇인가요?",
            "영혼과 육신의 관계는?",
            "현대 사회가 무너지는 이유는 무엇인가요?",
            "청년이 사회에서 가져야 할 태도는?",
            "운명은 정해져 있나요?",
            "수행이란 정확히 무엇인가요?",
            "정법은 불교나 유교와 무엇이 다른가요?",
            "감정이 자꾸 요동치는 이유가 뭘까요?",
            "무기력함을 어떻게 이겨낼 수 있죠?"
        ],
        inputs=msg,
        label="💡 예시 질문을 선택해보세요"
    )
    
    # 관련 질문 영역 
    with gr.Accordion("📎 관련 질문", open=True) as related_accordion:
        related_questions_component = gr.JSON(get_related_questions, visible=False)
        
        # 관련 질문 버튼 - tooltip 제거
        question1_btn = gr.Button("관련 질문 1", visible=False)
        question2_btn = gr.Button("관련 질문 2", visible=False)
        question3_btn = gr.Button("관련 질문 3", visible=False)
    
    # 질문 응답 후 질문 버튼 업데이트 함수
    def process_after_answer(chatbot_output, file_output):
        """질문-응답 후 관련 질문 버튼 업데이트"""
        button_updates = update_question_buttons()
        return chatbot_output, file_output, button_updates[0], button_updates[1], button_updates[2]
    
    # 이벤트 핸들러 설정
    submit_response = submit_btn.click(
        answer_question,
        inputs=[msg, chatbot],
        outputs=[chatbot, download_file]
    ).then(
        process_after_answer,
        inputs=[chatbot, download_file],
        outputs=[chatbot, download_file, question1_btn, question2_btn, question3_btn]
    )
    
    msg_response = msg.submit(
        answer_question,
        inputs=[msg, chatbot],
        outputs=[chatbot, download_file]
    ).then(
        process_after_answer,
        inputs=[chatbot, download_file],
        outputs=[chatbot, download_file, question1_btn, question2_btn, question3_btn]
    )
    
    # 관련 질문 버튼 클릭 이벤트
    def click_related_question(question_text):
        return question_text, gr.update(value=question_text)
    
    question1_btn.click(
        click_related_question,
        inputs=[question1_btn],
        outputs=[msg, msg]
    )
    
    question2_btn.click(
        click_related_question,
        inputs=[question2_btn],
        outputs=[msg, msg]
    )
    
    question3_btn.click(
        click_related_question,
        inputs=[question3_btn],
        outputs=[msg, msg]
    )

# 앱 실행
if __name__ == "__main__":
    try:
        # 인터페이스 실행
        logger.info("애플리케이션 실행 시작")
        print("\n홍익지기 챗봇 실행 준비 완료! 웹 브라우저가 잠시 후 열립니다...")
        
        # 포트 및 서버 설정으로 실행
        demo.launch(
            server_name="127.0.0.1",  # 로컬 서버 주소
            share=False,              # 외부 공유 비활성화
            quiet=True                # 로그 출력 축소
        )
    except KeyboardInterrupt:
        logger.info("사용자가 애플리케이션을 종료했습니다")
        print("\n애플리케이션이 종료되었습니다")
    except Exception as e:
        logger.error(f"애플리케이션 실행 중 오류 발생: {e}")
        logger.error(traceback.format_exc())
        print(f"오류: {e}")
        print("자세한 내용은 로그를 확인하세요")