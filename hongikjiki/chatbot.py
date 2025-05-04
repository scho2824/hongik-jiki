"""
홍익지기 챗봇 클래스 구현
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional, Union
import traceback

from hongikjiki.text_processing.document_processor import DocumentProcessor
from hongikjiki.vector_store.chroma_store import ChromaVectorStore
from hongikjiki.vector_store.embeddings import get_embeddings
from hongikjiki.langchain_integration import (
    get_llm,
    ConversationMemory,
    ContextualMemory,
    ConversationalQAChain
)

# 로거 설정
logger = logging.getLogger("HongikJikiChatBot")

class HongikJikiChatBot:
    """홍익지기 챗봇 클래스"""
    
    def __init__(self, 
                 persist_directory: str = "./data",
                 embedding_type: str = "huggingface",
                 llm_type: str = "openai",
                 collection_name: str = "hongikjiki_documents",
                 max_history: int = 10,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 use_tags: bool = True,
                 **kwargs):
        """
        홍익지기 챗봇 초기화
        
        Args:
            persist_directory: 데이터 저장 디렉토리
            embedding_type: 임베딩 모델 타입
            llm_type: 언어 모델 타입
            collection_name: 벡터 저장소 컬렉션 이름
            max_history: 최대 대화 기록 수
            chunk_size: 문서 청크 크기
            chunk_overlap: 문서 청크 중복 영역 크기
            use_tags: 태그 기반 검색 사용 여부
            **kwargs: 추가 설정 매개변수
        """
        self.persist_directory = persist_directory
        self.embedding_type = embedding_type
        self.llm_type = llm_type
        self.collection_name = collection_name
        self.max_history = max_history
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_tags = use_tags
        
        # 디렉토리 생성
        os.makedirs(persist_directory, exist_ok=True)
        
        # 임베딩 모델 초기화
        embedding_kwargs = kwargs.get("embedding_kwargs", {})
        self.embeddings = get_embeddings(embedding_type, **embedding_kwargs)
        logger.info(f"임베딩 모델 초기화 완료: {embedding_type}")
        
        # 벡터 저장소 경로 로깅
        logger.info(f"벡터 저장소 경로: {os.path.abspath(persist_directory)}")
        
        # 벡터 저장소 초기화 - vector_store_dir 대신 persist_directory 사용
        self.vector_store = ChromaVectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embeddings=self.embeddings
        )
        logger.info("벡터 저장소 초기화 완료")
        
        # 문서 처리기 초기화
        self.document_processor = DocumentProcessor(
            chunk_size=chunk_size,
            overlap=chunk_overlap
        )
        logger.info("문서 처리기 초기화 완료")
        
        # 태그 추출기 초기화 (사용 가능한 경우)
        self.tag_extractor = None
        if self.use_tags:
            try:
                # 프로젝트 루트 경로 추가
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                if project_root not in sys.path:
                    sys.path.append(project_root)
                    
                from hongikjiki.tagging.tag_schema import TagSchema
                from hongikjiki.tagging.tag_extractor import TagExtractor
                
                # 태그 스키마 및 패턴 파일 확인
                tag_schema_path = os.path.join(project_root, 'data', 'config', 'tag_schema.yaml')
                tag_patterns_path = os.path.join(project_root, 'data', 'config', 'tag_patterns.json')
                
                if os.path.exists(tag_schema_path):
                    tag_schema = TagSchema(tag_schema_path)
                    self.tag_extractor = TagExtractor(
                        tag_schema, 
                        patterns_file=tag_patterns_path if os.path.exists(tag_patterns_path) else None
                    )
                    logger.info("태그 추출기 초기화 완료")
            except ImportError:
                logger.info("태그 모듈을 불러올 수 없습니다. 태그 기능은 비활성화됩니다.")
                self.use_tags = False
            except Exception as e:
                logger.warning(f"태그 추출기 초기화 오류: {e}")
                self.use_tags = False
        
        # LLM 초기화 (외부 주입 지원)
        llm_kwargs = kwargs.get("llm_kwargs", {})
        if "llm" in kwargs and kwargs["llm"] is not None:
            self.llm = kwargs["llm"]
        else:
            self.llm = get_llm(llm_type, **llm_kwargs)
        logger.info(f"언어 모델 초기화 완료: {llm_type}")
        
        # 메모리 초기화
        self.memory = ContextualMemory(max_history=max_history)
        logger.info("대화 메모리 초기화 완료")
        
        # 대화 체인 초기화
        self.chain = ConversationalQAChain(
            llm=self.llm,
            vector_store=self.vector_store,
            memory=self.memory,
            k=kwargs.get("search_results_count", 4)
        )
        logger.info("대화 체인 초기화 완료")
        
        logger.info("홍익지기 챗봇 초기화 완료")
    
    def load_documents(self, directory: str) -> int:
        """
        문서 로드 및 벡터 저장소에 추가
        
        Args:
            directory: 문서 디렉토리 경로
            
        Returns:
            int: 처리된 문서 수
        """
        try:
            logger.info(f"{directory} 디렉토리에서 문서 로드 중...")
            
            # 문서 처리
            documents = self.document_processor.process_directory(
                directory, 
                self.chunk_size, 
                self.chunk_overlap
            )
            
            if not documents:
                logger.warning("처리된 문서가 없습니다.")
                return 0
            
            # 벡터 저장소에 추가
            ids = self.vector_store.add_documents(documents)
            
            logger.info(f"총 {len(documents)}개 청크 벡터 저장소에 추가 완료")
            return len(documents)
            
        except Exception as e:
            logger.error(f"문서 로드 오류: {e}")
            logger.error(traceback.format_exc())
            return 0
    
    def load_document(self, file_path: str) -> int:
        """
        단일 문서 로드 및 벡터 저장소에 추가
        
        Args:
            file_path: 문서 파일 경로
            
        Returns:
            int: 처리된 청크 수
        """
        try:
            logger.info(f"{file_path} 파일 처리 중...")
            
            # 문서 처리
            chunks = self.document_processor.process_file(file_path)
            
            if not chunks:
                logger.warning("처리된 청크가 없습니다.")
                return 0
            
            # 벡터 저장소에 추가
            ids = self.vector_store.add_documents(chunks)
            
            logger.info(f"총 {len(chunks)}개 청크 벡터 저장소에 추가 완료")
            return len(chunks)
            
        except Exception as e:
            logger.error(f"문서 처리 오류: {e}")
            logger.error(traceback.format_exc())
            return 0
    
    def chat(self, user_input: str) -> str:
        """
        사용자 입력에 대한 응답 생성

        Args:
            user_input: 사용자 입력 질문

        Returns:
            str: 생성된 응답
        """
        try:
            logger.info(f"사용자 질문: {user_input}")
            
            # 태그 추출 (태그 기능이 활성화된 경우)
            extracted_tags = []
            if self.use_tags and self.tag_extractor:
                extracted_tags = self.tag_extractor.extract_tags_from_query(user_input)
                logger.info(f"추출된 태그: {extracted_tags}")
            
            # 벡터 검색 수행
            logger.info("벡터 검색 시작...")
            # 검색 임계값 설정 - 이 값보다 낮은 유사도는 무시
            similarity_threshold = 0.3  
            
            # 태그 기반 검색 (태그가 있는 경우)
            if self.use_tags and extracted_tags and hasattr(self.vector_store, 'search_with_tags'):
                search_results = self.vector_store.search_with_tags(
                    user_input, 
                    tags=extracted_tags,
                    tag_boost=0.3,
                    k=7
                )
            else:
                search_results = self.vector_store.search(user_input, k=7)

            # 검색 결과 로깅
            logger.info(f"검색 결과 {len(search_results)}개 찾음")
            
            # 유사도 점수에 따라 필터링
            filtered_results = []
            for i, result in enumerate(search_results):
                score = result.get("score", 0)
                content = result.get("content", "")[:100]
                logger.info(f"결과 {i+1}: 점수={score:.4f}, 내용={content}...")
                
                # 임계값보다 높은 결과만 사용
                if score >= similarity_threshold:
                    filtered_results.append(result)
            
            logger.info(f"임계값({similarity_threshold}) 이상 결과: {len(filtered_results)}개")
            
            # 필터링된 결과를 사용
            search_results = filtered_results

            # 결과가 없는 경우 처리
            if not search_results:
                logger.warning("관련 문서를 찾지 못했습니다.")
                return "죄송합니다. 질문에 관련된 정법 문서를 찾지 못했습니다. 다른 질문을 해주시거나 질문을 조금 더 구체적으로 해주세요."

            # 검색 결과에서 관련 내용 추출 (상위 3개만 사용하여 토큰 제한)
            context_texts = []
            for i, result in enumerate(search_results[:3]):  # 상위 3개 결과만 사용
                content = result.get("content", "")
                
                # 콘텐츠 길이 제한 (4000자로 제한)
                if len(content) > 4000:
                    content = content[:4000] + "..."
                
                score = result.get("score", 0)

                # 메타데이터 정보 추가
                metadata = result.get("metadata", {})
                source_info = f"출처: {metadata.get('title', '제목 없음')}"
                if "lecture_number" in metadata:
                    source_info += f", 정법 {metadata.get('lecture_number')}강"
                
                # 태그 정보 추가 (태그가 있는 경우)
                if "tags" in metadata:
                    tags = metadata.get("tags", [])
                    if tags and isinstance(tags, str):
                        tags = [tag.strip() for tag in tags.split(",")]
                    if tags:
                        source_info += f", 태그: {', '.join(tags)}"
                
                # matching_tags 필드가 있는 경우 (검색 결과 재순위화 시)
                if "matching_tags" in result:
                    matching_tags = result.get("matching_tags", [])
                    if matching_tags:
                        source_info += f", 일치 태그: {', '.join(matching_tags)}"

                context_texts.append(f"{content}\n{source_info}")

            logger.info(f"컨텍스트 텍스트 {len(context_texts)}개 생성 (토큰 제한으로 최대 3개만 사용)")

            # 간결한 프롬프트 템플릿 (토큰 제한 해결)
            prompt_template = """
당신은 천공 스승님의 정법 가르침에 기반한 '홍익지기' 챗봇입니다. 

사용자 질문: {question}

참고할 정법 문서:
{context}

위 문서를 바탕으로 사용자의 질문에 명확하고 간결하게 답변해 주세요. 정법의 가르침을 정확하게 전달하면서 실용적인 조언을 제공하세요.
"""

            # 프롬프트 생성
            context_text = "\n\n".join(context_texts)
            prompt = prompt_template.format(
                question=user_input,
                context=context_text
            )

            # 로깅 (디버깅용)
            logger.info("LLM 응답 생성 시작...")
            logger.debug(f"생성 프롬프트 길이: {len(prompt)} 자")

            # LLM으로 응답 생성
            response = self.generate_text(prompt)
            logger.info("LLM 응답 생성 완료")
            logger.debug(f"생성된 응답: {response}")
            
            # 관련 강의 추천 추가 (태그 기반)
            related_lectures = self._extract_related_lectures(search_results)
            if related_lectures:
                response += "\n\n🧭 *더 알고 싶다면, 다음 강의를 참고해보세요:*\n"
                for lecture in related_lectures:
                    response += f"* {lecture}\n"

            return response

        except Exception as e:
            logger.error(f"응답 생성 오류: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return "죄송합니다. 응답 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
    
    def _extract_related_lectures(self, search_results: List[Dict[str, Any]]) -> List[str]:
        """
        검색 결과에서 관련 강의 정보 추출
        
        Args:
            search_results: 검색 결과 리스트
            
        Returns:
            List[str]: 관련 강의 정보 리스트
        """
        lectures = []
        seen_files = set()  # 중복 방지용
        
        for result in search_results:
            metadata = result.get("metadata", {})
            filename = metadata.get("filename", "")
            
            if filename in seen_files:
                continue
                
            # 강의 번호가 있는 경우 사용
            lecture_num = metadata.get("lecture_number")
            title = metadata.get("title", "")
            
            if lecture_num:
                lecture = f"정법 {lecture_num}강: {title}"
            else:
                # 강의 번호가 없는 경우 파일명 사용
                lecture = filename.replace(".txt", "").replace("_", " ")
                
            lectures.append(lecture)
            seen_files.add(filename)
        
        return lectures[:3]  # 최대 3개만 반환
    
    def chat_with_context(self, message: str, context: Optional[str] = None) -> str:
        """
        컨텍스트를 포함한 사용자 메시지에 대한 응답 생성
        
        Args:
            message: 사용자 메시지
            context: 추가 컨텍스트 (선택 사항)
            
        Returns:
            str: 챗봇 응답
        """
        try:
            input_data = {
                "question": message,
                "context": context
            }
            
            # 체인을 통해 응답 생성
            response = self.chain.run(input_data)
            
            return response
            
        except Exception as e:
            logger.error(f"컨텍스트 응답 생성 오류: {e}")
            logger.error(traceback.format_exc())
            return f"죄송합니다. 응답 생성 중 오류가 발생했습니다: {str(e)}"

    def generate_text(self, prompt: str) -> str:
        """
        LLM을 사용하여 텍스트 생성

        Args:
            prompt: 프롬프트 텍스트

        Returns:
            str: 생성된 텍스트
        """
        # llm 객체가 self.llm에 저장되어 있다고 가정
        return self.llm.generate_text(prompt)
    
    def add_context(self, key: str, value: Any) -> None:
        """
        대화 컨텍스트에 정보 추가
        
        Args:
            key: 컨텍스트 키
            value: 컨텍스트 값
        """
        if isinstance(self.memory, ContextualMemory):
            self.memory.add_context(key, value)
            logger.debug(f"컨텍스트 추가: {key}")
    
    def clear_history(self) -> None:
        """대화 기록 초기화"""
        self.memory.clear()
        logger.info("대화 기록 초기화 완료")
    
    def search_documents(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """
        문서 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            
        Returns:
            List[Dict]: 검색 결과
        """
        try:
            # 태그 기반 검색 사용 가능 여부 확인
            if self.use_tags and hasattr(self.vector_store, 'advanced_search'):
                return self.vector_store.advanced_search(query, use_tags=True, k=k)
            else:
                return self.vector_store.search(query, k=k)
        except Exception as e:
            logger.error(f"문서 검색 오류: {e}")
            return []
    
    def get_document_count(self) -> int:
        """
        벡터 저장소의 문서 수 반환
        
        Returns:
            int: 문서 수
        """
        try:
            return self.vector_store.count()
        except Exception as e:
            logger.error(f"문서 개수 조회 오류: {e}")
            return 0
    
    def reset_vector_store(self) -> bool:
        """
        벡터 저장소 초기화
        
        Returns:
            bool: 성공 여부
        """
        try:
            self.vector_store.reset()
            logger.info("벡터 저장소 초기화 완료")
            return True
        except Exception as e:
            logger.error(f"벡터 저장소 초기화 오류: {e}")
            return False
            
    def get_related_questions(self, user_question: str, max_questions: int = 3) -> List[str]:
        """
        태그 기반 관련 질문 추천
        
        Args:
            user_question: 사용자 질문
            max_questions: 최대 추천 질문 수
            
        Returns:
            List[str]: 추천 질문 리스트
        """
        if not self.use_tags or not self.tag_extractor:
            return []
        
        # 태그 추출
        query_tags = self.tag_extractor.extract_tags_from_query(user_question)
        if not query_tags:
            return []
        
        # 태그별 템플릿 질문
        tag_questions = {
            "정법": [
                "정법이란 무엇인가요?", 
                "정법을 어떻게 실천할 수 있나요?",
                "정법과 종교의 차이점은 무엇인가요?"
            ],
            "홍익인간": [
                "홍익인간이란 무엇인가요?", 
                "홍익인간을 실천하는 방법은 무엇인가요?",
                "홍익인간의 정신으로 세상을 바라보면 어떤 점이 달라지나요?"
            ],
            "수행": [
                "정법에서 말하는 수행이란 무엇인가요?", 
                "일상에서 수행을 어떻게 할 수 있나요?",
                "수행을 통해 얻을 수 있는 것은 무엇인가요?"
            ],
            "인간관계": [
                "정법에서 말하는 올바른 인간관계는 무엇인가요?", 
                "인간관계에서 갈등이 생길 때 어떻게 해결해야 하나요?",
                "가족 관계에서 어려움이 있을 때 어떻게 대처해야 하나요?"
            ],
            "자유의지": [
                "정법에서 말하는 자유의지란 무엇인가요?",
                "선택과 책임의 관계는 어떻게 되나요?",
                "인간의 자유의지와 우주 법칙은 어떤 관계가 있나요?"
            ],
            "깨달음": [
                "정법에서 말하는 깨달음이란 무엇인가요?",
                "깨달음에 이르는 방법은 무엇인가요?",
                "깨달은 후의 삶은 어떻게 달라지나요?"
            ],
            "불안": [
                "불안한 마음을 다스리는 방법은 무엇인가요?",
                "걱정과 불안에서 벗어나려면 어떻게 해야 하나요?",
                "불안은 왜 생기는 건가요?"
            ],
            "분노": [
                "분노를 다스리는 방법은 무엇인가요?",
                "화가 날 때 어떻게 대처해야 하나요?",
                "분노의 원인은 무엇인가요?"
            ]
        }
        
        # 추출된 태그에 해당하는 질문 수집
        related_questions = []
        for tag in query_tags:
            if tag in tag_questions:
                related_questions.extend(tag_questions[tag])
        
        # 중복 제거 및 최대 개수 제한
        unique_questions = []
        for q in related_questions:
            if q not in unique_questions:
                unique_questions.append(q)
                if len(unique_questions) >= max_questions:
                    break
        
        return unique_questions


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
        extracted_tags = self.tag_extractor.extract_tags_from_query(query)
        
        # 태그 스키마에서 태그 정보 가져오기
        tag_info = []
        for tag in extracted_tags[:max_tags]:
            tag_obj = self.tag_extractor.tag_schema.get_tag(tag)
            if tag_obj:
                tag_info.append({
                    "name": tag,
                    "category": tag_obj.category,
                    "emoji": tag_obj.emoji or "",
                    "description": tag_obj.description or ""
                })
        
        return tag_info

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