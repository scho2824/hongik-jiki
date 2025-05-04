"""
개선된 QA 체인 구현 - 인용 기능 추가
"""

import logging
import re
from typing import List, Dict, Any, Optional, Union, Callable, Tuple

from hongikjiki.langchain_integration.base import ChainBase, LLMBase
from hongikjiki.langchain_integration.memory import MemoryBase
from hongikjiki.vector_store.base import VectorStoreBase

logger = logging.getLogger("HongikJikiChatBot")

class PromptTemplate:
    """프롬프트 템플릿 관리 클래스"""
    
    def __init__(self, template: str):
        """
        프롬프트 템플릿 초기화
        
        Args:
            template: 프롬프트 템플릿 문자열 (변수는 {variable_name} 형식)
        """
        self.template = template
    
    def format(self, **kwargs) -> str:
        """
        변수를 대체하여 프롬프트 생성
        
        Args:
            **kwargs: 템플릿 변수 값
            
        Returns:
            str: 포맷팅된 프롬프트
        """
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            logger.error(f"프롬프트 템플릿 변수 누락: {e}")
            # 누락된 변수를 플레이스홀더로 대체
            result = self.template
            for key in re.findall(r'\{([^}]+)\}', self.template):
                if key not in kwargs:
                    result = result.replace("{{" + key + "}}", f"[{key}]")
            return result


class QAChain(ChainBase):
    """개선된 질의응답 체인 구현 - 인용 기능 추가"""
    
    def __init__(self, 
                 llm: LLMBase,
                 vector_store: VectorStoreBase,
                 memory: Optional[MemoryBase] = None,
                 prompt_template: Optional[str] = None,
                 citation_prompt_template: Optional[str] = None,
                 k: int = 4):
        """
        QA 체인 초기화
        
        Args:
            llm: 언어 모델 인스턴스
            vector_store: 벡터 저장소 인스턴스
            memory: 대화 메모리 인스턴스 (선택 사항)
            prompt_template: 사용자 정의 프롬프트 템플릿 (선택 사항)
            citation_prompt_template: 인용 처리를 위한 프롬프트 템플릿 (선택 사항)
            k: 검색할 문서 수
        """
        self.llm = llm
        self.vector_store = vector_store
        self.memory = memory
        self.k = k
        
        # 기본 프롬프트 템플릿 설정
        default_template = """
당신은 정법 지식을 제공하는 홍익지기 인공지능 비서입니다.
사용자의 질문에 대해 정확하고 도움이 되는 답변을 제공해야 합니다.
아래 제공된 정법 문서를 기반으로 질문에 답변하세요.

## 중요 지침:
1. 제공된 정법 문서 내용만 사용하여 답변하세요.
2. 문서에 없는 내용은 답변하지 마세요.
3. 답변에 사용한 출처를 명확히 인용하세요. "[문서 X]"와 같은 형식으로 인용해주세요.
4. 사용자의 질문을 이해하지 못하거나 문서에 관련 내용이 없는 경우, 솔직하게 모른다고 답변하세요.
5. 답변은 친절하고 이해하기 쉽게 작성하세요.
6. 인용문을 사용할 때는 직접 인용 부분을 큰따옴표로 표시하세요. 예: "[문서 1]에 따르면, "정법은 홍익인간의 정신을 담고 있습니다." 라고 되어 있습니다."

### 관련 정법 문서:
{context}

### 대화 기록:
{chat_history}

### 사용자 질문:
{question}

### 답변:
"""
        
        # 인용 처리를 위한 프롬프트 템플릿
        default_citation_template = """
다음은 홍익지기 챗봇이 생성한 답변입니다. 이 답변에 인용문을 명확히 표시해주세요.
각 인용의 원본 문서 번호를 대괄호 안에 표시하고, 직접 인용한 내용은 큰따옴표로 표시하세요.

원본 답변:
{answer}

관련 문서:
{sources}

인용이 표시된 답변을 작성해주세요:
"""
        
        # 프롬프트 템플릿 설정
        self.prompt_template = PromptTemplate(prompt_template or default_template)
        self.citation_template = PromptTemplate(citation_prompt_template or default_citation_template)
    
    def run(self, input_data: Union[str, Dict[str, Any]]) -> str:
        """
        체인 실행
        
        Args:
            input_data: 입력 데이터 (문자열 또는 딕셔너리)
            
        Returns:
            str: 생성된 답변
        """
        try:
            # 입력 데이터 처리
            if isinstance(input_data, str):
                question = input_data
                additional_context = None
            else:
                question = input_data.get("question", "")
                additional_context = input_data.get("context", None)
            
            # 유효한 질문인지 확인
            if not question.strip():
                return "질문을 입력해 주세요."
            
            # 메모리에 사용자 질문 추가
            if self.memory:
                self.memory.add_user_message(question)
            
            # 벡터 저장소에서 관련 문서 검색
            search_results = self.vector_store.search(question, k=self.k)
            
            # 검색 결과가 없는 경우
            if not search_results:
                no_result_response = "죄송합니다. 질문에 관련된 정법 문서를 찾지 못했습니다. 다른 질문을 해주시거나 질문을 조금 더 구체적으로 해주세요."
                if self.memory:
                    self.memory.add_ai_message(no_result_response)
                return no_result_response
            
            # 컨텍스트 구성
            context, sources = self._build_context(search_results, additional_context)
            
            # 대화 기록 가져오기
            chat_history = ""
            if self.memory:
                chat_history = self.memory.get_chat_history(max_tokens=1000)
            
            # 프롬프트 구성
            prompt = self.prompt_template.format(
                context=context,
                chat_history=chat_history,
                question=question
            )
            
            # LLM으로 답변 생성
            answer = self.llm.generate(prompt)
            
            # 인용 처리
            citation_prompt = self.citation_template.format(
                answer=answer,
                sources=sources
            )
            
            # 인용이 표시된 최종 답변 생성
            final_answer = self.llm.generate(citation_prompt)
            
            # 메모리에 AI 답변 추가
            if self.memory:
                self.memory.add_ai_message(final_answer)
            
            return final_answer
            
        except Exception as e:
            logger.error(f"QA 체인 실행 오류: {e}")
            error_msg = f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {str(e)}"
            if self.memory:
                self.memory.add_ai_message(error_msg)
            return error_msg
    
    def _build_context(self, search_results: List[Dict[str, Any]], 
                      additional_context: Optional[str] = None) -> Tuple[str, str]:
        """
        LLM에 제공할 컨텍스트 구성
        
        Args:
            search_results: 벡터 검색 결과
            additional_context: 추가 컨텍스트 (선택 사항)
            
        Returns:
            Tuple[str, str]: 포맷팅된 컨텍스트와 소스 정보
        """
        context_parts = []
        source_parts = []
        
        max_chunk_size = 1000
        
        # 검색 결과 포맷팅
        for i, doc in enumerate(search_results):
            content = doc["content"]
            metadata = doc["metadata"]
            similarity = doc.get("score", 0.0)
            
            # 메타데이터에서 유용한 정보 추출
            source_info = f"강의 {metadata.get('lecture_number', '?')}강"
            if metadata.get('title'):
                source_info += f" - {metadata.get('title')}"
            
            # 청크 정보 추가
            if 'chunk_info' in metadata:
                source_info += f" ({metadata['chunk_info']})"
            
            # 청크 길이 제한
            if len(content) > max_chunk_size:
                content = content[:max_chunk_size] + "... [생략됨]"
            
            # 컨텍스트 부분 포맷팅
            context_part = f"[문서 {i+1}] {source_info}\n{content}\n"
            context_parts.append(context_part)
            
            # 소스 정보 저장
            source_part = f"[문서 {i+1}] {source_info} (유사도: {similarity:.2f})"
            source_parts.append(source_part)
        
        # 모든 부분 결합
        formatted_context = "\n\n".join(context_parts)
        formatted_sources = "\n".join(source_parts)
        
        # 추가 컨텍스트가 있으면 추가
        if additional_context:
            formatted_context += f"\n\n[추가 정보]\n{additional_context}\n"
            formatted_sources += f"\n[추가 정보] 사용자 제공 컨텍스트"
        
        return formatted_context, formatted_sources


class ConversationalQAChain(QAChain):
    """개선된 대화형 질의응답 체인 구현"""
    
    def __init__(self, 
                 llm: LLMBase,
                 vector_store: VectorStoreBase,
                 memory: MemoryBase,
                 prompt_template: Optional[str] = None,
                 citation_prompt_template: Optional[str] = None,
                 k: int = 4):
        """
        대화형 QA 체인 초기화
        
        Args:
            llm: 언어 모델 인스턴스
            vector_store: 벡터 저장소 인스턴스
            memory: 대화 메모리 인스턴스 (필수)
            prompt_template: 사용자 정의 프롬프트 템플릿 (선택 사항)
            citation_prompt_template: 인용 처리를 위한 프롬프트 템플릿 (선택 사항)
            k: 검색할 문서 수
        """
        # 메모리가 필수
        if memory is None:
            raise ValueError("ConversationalQAChain requires a memory instance")
        
        # 기본 프롬프트 템플릿 설정
        default_template = """
당신은 정법 지식을 제공하는 홍익지기 인공지능 비서입니다.
사용자의 질문에 대해 정확하고 도움이 되는 답변을 제공해야 합니다.
아래 제공된 정법 문서와 대화 기록을 기반으로 질문에 답변하세요.

## 중요 지침:
1. 제공된 정법 문서 내용만 사용하여 답변하세요.
2. 문서에 없는 내용은 답변하지 마세요.
3. 답변에 사용한 출처를 명확히 인용하세요. "[문서 X]"와 같은 형식으로 인용해주세요.
4. 사용자의 질문을 이해하지 못하거나 문서에 관련 내용이 없는 경우, 솔직하게 모른다고 답변하세요.
5. 답변은 친절하고 이해하기 쉽게 작성하세요.
6. 인용문을 사용할 때는 직접 인용 부분을 큰따옴표로 표시하세요.
7. 대화의 맥락을 고려하여 이전 질문과 답변을 참고하세요.

정법에 대해 알고 있는 내용:
- 정법은 천공 스승님께서 알려주신 우주 법칙에 대한 가르침입니다.
- 홍익인간 이념과 관련이 있습니다.
- 자연의 법칙과 조화, 인간 내면의 성장 등을 중요시합니다.

### 관련 정법 문서:
{context}

### 대화 기록:
{chat_history}

### 사용자 질문:
{question}

### 답변:
"""
        
        # 부모 클래스 초기화
        super().__init__(
            llm=llm,
            vector_store=vector_store,
            memory=memory,
            prompt_template=prompt_template or default_template,
            citation_prompt_template=citation_prompt_template,
            k=k
        )
    
    def condense_question(self, current_question: str) -> str:
        """
        대화 기록과 현재 질문을 고려하여 독립적인 질문으로 재구성
        
        Args:
            current_question: 현재 사용자 질문
            
        Returns:
            str: 재구성된 독립적인 질문
        """
        # 대화 기록이 없으면 원래 질문 반환
        if not self.memory or len(self.memory.messages) <= 1:
            return current_question
        
        # 질문 재구성을 위한 프롬프트
        condense_template = """
다음은 사용자와의 대화 기록입니다:

{chat_history}

사용자의 가장 최근 질문: "{current_question}"

이전 대화 맥락을 고려하여, 가장 최근 질문을 독립적이고 자기 완결적인 질문으로 재구성해주세요.
이전 대화에서 언급된 중요한 맥락을 모두 포함해야 합니다.
다른 설명 없이 재구성된 질문만 작성해주세요.
"""
        
        prompt = condense_template.format(
            chat_history=self.memory.get_chat_history(),
            current_question=current_question
        )
        
        # LLM으로 재구성된 질문 생성
        try:
            condensed_question = self.llm.generate(prompt)
            condensed_question = condensed_question.strip()
            logger.debug(f"원래 질문: '{current_question}' -> 재구성된 질문: '{condensed_question}'")
            return condensed_question
        except Exception as e:
            logger.warning(f"질문 재구성 오류: {e}, 원래 질문 사용")
            return current_question
    
    def run(self, input_data: Union[str, Dict[str, Any]]) -> str:
        """
        대화형 체인 실행
        
        Args:
            input_data: 입력 데이터 (문자열 또는 딕셔너리)
            
        Returns:
            str: 생성된 답변
        """
        try:
            # 입력 데이터 처리
            if isinstance(input_data, str):
                question = input_data
                additional_context = None
            else:
                question = input_data.get("question", "")
                additional_context = input_data.get("context", None)
            
            # 유효한 질문인지 확인
            if not question.strip():
                return "질문을 입력해 주세요."
            
            # 메모리에 사용자 질문 추가
            self.memory.add_user_message(question)
            
            # 질문 재구성
            condensed_question = self.condense_question(question)
            
            # 벡터 저장소에서 관련 문서 검색 (재구성된 질문 사용)
            search_results = self.vector_store.search(condensed_question, k=self.k)
            
            # 검색 결과가 없는 경우
            if not search_results:
                no_result_response = "죄송합니다. 질문에 관련된 정법 문서를 찾지 못했습니다. 다른 질문을 해주시거나 질문을 조금 더 구체적으로 해주세요."
                self.memory.add_ai_message(no_result_response)
                return no_result_response
            
            # 컨텍스트 구성
            context, sources = self._build_context(search_results, additional_context)
            
            # 대화 기록 가져오기
            chat_history = self.memory.get_chat_history(max_tokens=1000)
            
            # 프롬프트 구성
            prompt = self.prompt_template.format(
                context=context,
                chat_history=chat_history,
                question=question  # 원래 질문 사용
            )
            
            # LLM으로 답변 생성
            answer = self.llm.generate(prompt)
            
            # 인용 처리
            citation_prompt = self.citation_template.format(
                answer=answer,
                sources=sources
            )
            
            # 인용이 표시된 최종 답변 생성
            final_answer = self.llm.generate(citation_prompt)
            
            # 메모리에 AI 답변 추가
            self.memory.add_ai_message(final_answer)
            
            return final_answer
            
        except Exception as e:
            logger.error(f"대화형 QA 체인 실행 오류: {e}")
            error_msg = f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {str(e)}"
            self.memory.add_ai_message(error_msg)
            return error_msg