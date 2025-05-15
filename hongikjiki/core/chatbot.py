"""
hongikjiki.core.chatbot

홍익지기 챗봇의 핵심 로직을 담당하는 모듈.
이 모듈은 문서 검색, 질문 응답 생성, 포맷팅 등 챗봇의 핵심 기능을 구현합니다.
순수한 비즈니스 로직만 포함하고, 실제 구현체(LLM, 벡터 저장소 등)는 의존성 주입 받습니다.
"""

# For future file path consistency
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[2]

import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import os
import tempfile
import hashlib

logger = logging.getLogger(__name__)

class HongikJikiChatbot:
    """
    홍익지기 챗봇 핵심 클래스
    
    이 클래스는 정법 가르침을 기반으로 한 챗봇의 핵심 로직을 구현합니다.
    외부 의존성(LLM, 벡터 저장소, 태그 추출기)을 주입받아 사용합니다.
    """
    
    def __init__(self, 
                 llm, 
                 vector_store, 
                 tag_extractor=None):
        """
        HongikJikiChatbot 초기화
        
        Args:
            llm: 언어 모델 인터페이스 (generate 메서드 제공)
            vector_store: 벡터 저장소 인터페이스 (search 메서드 제공)
            tag_extractor: 태그 추출기 (옵션)
        """
        self.llm = llm
        self.vector_store = vector_store
        self.tag_extractor = tag_extractor
        self.related_questions: List[Dict[str, str]] = []
        logger.info("홍익지기 챗봇 핵심 엔진 초기화 완료")
    
    def extract_tags(self, message: str) -> List[str]:
        """
        질문에서 태그 추출
        
        Args:
            message: 사용자 질문
            
        Returns:
            List[str]: 추출된 태그 리스트
        """
        extracted_tags = []
        if self.tag_extractor:
            try:
                extracted_tags = self.tag_extractor.extract_tags_from_query(message)
                logger.info(f"질문에서 추출한 태그: {extracted_tags}")
            except Exception as e:
                logger.warning(f"태그 추출 오류: {e}")
        return extracted_tags
    
    def search_documents(self, message: str, tags: Optional[List[str]] = None, k: int = 3) -> List[Dict[str, Any]]:
        """
        관련 문서 검색
        
        Args:
            message: 사용자 질문
            tags: 검색에 사용할 태그 리스트 (옵션)
            k: 반환할 최대 문서 수
            
        Returns:
            List[Dict]: 검색 결과 문서 리스트
        """
        actual_tags = tags or []
        try:
            # 태그 기반 검색 실행
            if hasattr(self.vector_store, 'advanced_search') and actual_tags:
                logger.info(f"태그 기반 고급 검색 실행: {actual_tags}")
                results = self.vector_store.advanced_search(message, use_tags=True, k=k)
            else:
                logger.info("일반 벡터 검색 실행")
                results = self.vector_store.search(message, k=k)
            return results
        except Exception as e:
            logger.error(f"문서 검색 오류: {e}")
            return []
    
    def generate_answer(self, message: str, results: List[Dict[str, Any]]) -> str:
        """
        검색 결과를 기반으로 답변 생성
        
        Args:
            message: 사용자 질문
            results: 검색 결과 문서 리스트
            
        Returns:
            str: 생성된 답변
        """
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
    
    def format_response(self, results: List[Dict[str, Any]], answer: str, extracted_tags: Optional[List[str]] = None) -> str:
        """
        챗봇 응답을 포맷팅하는 함수
        
        Args:
            results: 검색 결과 문서 리스트
            answer: 생성된 답변 텍스트
            extracted_tags: 추출된 태그 리스트
            
        Returns:
            str: 포맷팅된 응답
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
        if tags and len(tags) > 0:  # 태그가 실제로 있는 경우에만
            tag_list = ' '.join([f"#{tag}" for tag in tags])
            formatted += f"🏷️ 관련 태그: {tag_list}"
        
        return formatted
    
    def generate_related_questions(self, tags: List[str], current_question: str) -> List[Dict[str, str]]:
        """
        태그 기반 관련 질문 생성
        
        Args:
            tags: 태그 리스트
            current_question: 현재 질문
            
        Returns:
            List[Dict]: 관련 질문 정보 리스트
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
            "정법": [
                "정법의 핵심 원리는 무엇인가요?",
                "정법에서 말하는 자유의지란 무엇인가요?",
                "정법의 관점에서 행복이란 무엇인가요?"
            ],
            "자유": [
                "진정한 자유란 무엇인가요?",
                "자유로운 삶을 사는 방법은 무엇인가요?",
                "자유와 책임의 관계는 무엇인가요?"
            ],
            "수행": [
                "수행이란 무엇인가요?",
                "일상에서 수행하는 방법은 무엇인가요?",
                "수행을 통해 얻을 수 있는 것은 무엇인가요?"
            ],
            "마음": [
                "마음을 다스리는 방법은 무엇인가요?",
                "마음의 평화를 얻는 방법은 무엇인가요?",
                "마음의 작용 원리는 무엇인가요?"
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
    
    def create_temp_file(self, content: str) -> str:
        """
        임시 파일 생성
        
        Args:
            content: 파일에 저장할 내용
            
        Returns:
            str: 임시 파일 경로
        """
        try:
            # 임시 파일 생성
            with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8', suffix='.txt') as f:
                f.write(content)
                return f.name
        except Exception as e:
            logger.error(f"임시 파일 생성 오류: {e}")
            return ""
    
    def answer_question(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        사용자 질문에 답변 생성
        
        Args:
            message: 사용자 질문
            history: 대화 이력 (각 항목은 {'role': 'user' 또는 'assistant', 'content': '메시지'} 형식)
            
        Returns:
            Dict: 답변 및 관련 정보
        """
        logger.info(f"사용자 질문: {message}")
        if history is None:
            history = []
        
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
            self.related_questions = self.generate_related_questions(extracted_tags, message)
            
            # 임시 파일 생성
            temp_file = self.create_temp_file(answer)
            
            return {"answer": answer, "file": temp_file}
        
        # 문서 검색
        results = self.search_documents(message, extracted_tags)
        
        # 결과가 없는 경우
        if not results:
            answer = "죄송합니다. 질문에 관련된 정법 문서를 찾지 못했습니다. 다른 질문을 해주시거나 질문을 조금 더 구체적으로 해주세요."
            formatted_answer = answer
            
            # 관련 질문 생성
            self.related_questions = self.generate_related_questions(extracted_tags, message)
            
            # 임시 파일 생성
            temp_file = self.create_temp_file(formatted_answer)
            
            return {"answer": formatted_answer, "file": temp_file}
        
        # 답변 생성
        answer = self.generate_answer(message, results)
        
        # 응답 포맷팅
        formatted_answer = self.format_response(results, answer, extracted_tags)
        
        # 관련 질문 생성
        self.related_questions = self.generate_related_questions(extracted_tags, message)
        
        # 임시 파일 생성
        temp_file = self.create_temp_file(formatted_answer)
        
        return {"answer": formatted_answer, "file": temp_file}
    
    def get_related_questions(self) -> List[Dict[str, str]]:
        """
        현재 관련 질문 목록 반환
        
        Returns:
            List[Dict]: 관련 질문 정보 리스트
        """
        return self.related_questions