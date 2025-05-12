# hongikjiki/core/chatbot.py
import logging
from hongikjiki.core.formatter import format_response
from hongikjiki.core.related_questions import generate_related_questions
from hongikjiki.utils.file_utils import create_temp_file

logger = logging.getLogger("HongikJikiChatBot")

class HongikJikiChatbot:
    def __init__(self, llm, vector_store, tag_extractor=None):
        """챗봇 클래스 초기화"""
        self.llm = llm
        self.vector_store = vector_store
        self.tag_extractor = tag_extractor
        self.related_questions = []
        
    def extract_tags(self, message):
        """질문에서 태그 추출"""
        extracted_tags = []
        if self.tag_extractor:
            try:
                extracted_tags = self.tag_extractor.extract_tags_from_query(message)
                logger.info(f"질문에서 추출한 태그: {extracted_tags}")
            except Exception as e:
                logger.warning(f"태그 추출 오류: {e}")
        return extracted_tags
        
    def search_documents(self, message, tags=None, k=3):
        """관련 문서 검색"""
        try:
            # 태그 기반 검색 실행
            if hasattr(self.vector_store, 'advanced_search') and tags:
                logger.info(f"태그 기반 고급 검색 실행: {tags}")
                results = self.vector_store.advanced_search(message, use_tags=True, k=k)
            else:
                logger.info("일반 벡터 검색 실행")
                results = self.vector_store.search(message, k=k)
            return results
        except Exception as e:
            logger.error(f"문서 검색 오류: {e}")
            return []
    
    def generate_answer(self, message, results):
        """검색 결과를 기반으로 답변 생성"""
        # 결과가 없으면 간단 응답
        if not results:
            logger.warning("검색 결과가 없습니다.")
            return "죄송합니다. 질문에 관련된 정법 문서를 찾지 못했습니다. 다른 질문을 해주시거나 질문을 조금 더 구체적으로 해주세요."
        
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
        try:
            return self.llm.generate(prompt)
        except Exception as e:
            logger.error(f"LLM 응답 생성 오류: {e}")
            return f"오류가 발생했습니다: {str(e)}"
    
    def answer_question(self, message, history):
        """사용자 질문에 답변 생성"""
        logger.info(f"사용자 질문: {message}")
        
        # 질문에서 태그 추출
        extracted_tags = self.extract_tags(message)
        
        # 벡터 스토어가 없거나 빈 경우 간단 모드 응답
        if not self.vector_store or self.vector_store.count() == 0:
            # 간단 모드 응답 생성
            prompt = f"""
            당신은 정법 지식을 제공하는 홍익지기 인공지능 비서입니다.
            사용자의 질문에 대해 정확하고 도움이 되는 답변을 제공해야 합니다.
            
            정법에 대해 알고 있는 내용:
            - 정법은 천공 스승님께서 알려주신 우주 법칙에 대한 가르침입니다.
            - 홍익인간 이념과 관련이 있습니다.
            - 자연의 법칙과 조화, 인간 내면의 성장 등을 중요시합니다.
            
            다음 질문에 대해 정법의 관점에서 답변해주세요:
            {message}
            """
            answer = self.llm.generate(prompt)
            
            # 관련 질문 생성
            self.related_questions = generate_related_questions(extracted_tags, message)
            
            # 임시 파일 생성
            temp_file = create_temp_file(answer)
            
            return {"answer": answer, "file": temp_file}
        
        # 문서 검색
        results = self.search_documents(message, extracted_tags)
        
        # 결과가 없는 경우
        if not results:
            answer = "죄송합니다. 질문에 관련된 정법 문서를 찾지 못했습니다. 다른 질문을 해주시거나 질문을 조금 더 구체적으로 해주세요."
            formatted_answer = answer
            
            # 관련 질문 생성
            self.related_questions = generate_related_questions(extracted_tags, message)
            
            # 임시 파일 생성
            temp_file = create_temp_file(formatted_answer)
            
            return {"answer": formatted_answer, "file": temp_file}
        
        # 답변 생성
        answer = self.generate_answer(message, results)
        
        # 응답 포맷팅
        formatted_answer = format_response(results, answer, extracted_tags)
        
        # 관련 질문 생성
        self.related_questions = generate_related_questions(extracted_tags, message)
        
        # 임시 파일 생성
        temp_file = create_temp_file(formatted_answer)
        
        return {"answer": formatted_answer, "file": temp_file}
        
    def get_related_questions(self):
        """현재 관련 질문 목록 반환"""
        return self.related_questions
    