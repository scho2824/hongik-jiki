import os
import sys
import logging
from typing import List, Dict, Any, Optional, Union, Sequence, Tuple

import chromadb
from chromadb.config import Settings

from hongikjiki.vector_store.base import VectorStoreBase

logger = logging.getLogger("HongikJikiChatBot")

class ChromaVectorStore(VectorStoreBase):
    """
    ChromaDB 기반 벡터 저장소 구현
    텍스트 청크를 벡터화하여 저장하고 유사도 검색 기능 제공
    """
    
    def __init__(self, 
                collection_name: str = "hongikjiki_documents",
                persist_directory: str = "./data/vector_store",
                embeddings = None):
        """
        ChromaVectorStore 초기화
        
        Args:
            collection_name: 컬렉션 이름
            persist_directory: 벡터 저장소 지속성 디렉토리
            embeddings: 임베딩 모델 (없으면 기본값 사용)
        """
        super().__init__()
        
        # 디렉토리 생성
        os.makedirs(persist_directory, exist_ok=True)
        
        # 임베딩 설정
        self.embeddings = embeddings
        
        # Chroma 클라이언트 초기화
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # 컬렉션 생성 또는 가져오기
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # 태그 관련 필드 초기화
        self.tag_index = None
        self.tag_aware_search = None
        
        # 태그 인덱스 로드 시도
        try:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            if project_root not in sys.path:
                sys.path.append(project_root)
                
            from hongikjiki.vector_store.tag_index import TagIndex, TagAwareSearch
            
            # 태그 인덱스 파일 확인
            tag_index_path = os.path.join(project_root, 'data', 'tag_data', 'tag_index.json')
            if os.path.exists(tag_index_path):
                logger.info(f"태그 인덱스 로드 중: {tag_index_path}")
                self.tag_index = TagIndex(tag_index_path)
                self.tag_aware_search = TagAwareSearch(self.tag_index)
                logger.info("태그 기반 검색 초기화 완료")
            else:
                logger.info(f"태그 인덱스 파일이 없습니다: {tag_index_path}")
        except ImportError:
            logger.info("태그 모듈을 불러올 수 없습니다. 일반 검색만 사용합니다.")
        except Exception as e:
            logger.warning(f"태그 인덱스 로드 오류: {e}")
        
        logger.info(f"ChromaVectorStore 초기화 완료: 컬렉션={collection_name}, 위치={persist_directory}")
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        텍스트 리스트를 벡터 저장소에 추가
        
        Args:
            texts: 추가할 텍스트 리스트
            metadatas: 각 텍스트에 대한 메타데이터 리스트 (옵션)
            
        Returns:
            List[str]: 추가된 문서 ID 리스트
        """
        if not texts:
            logger.warning("추가할 텍스트가 없습니다.")
            return []
        
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # 메타데이터 정리 (복잡한 타입 처리)
        sanitized_metadatas = []
        for metadata in metadatas:
            if metadata is None:
                sanitized_metadatas.append({})
                continue
                
            sanitized = {}
            for key, value in metadata.items():
                # None 값 처리
                if value is None:
                    sanitized[key] = ""
                # 리스트 처리
                elif isinstance(value, list):
                    # 리스트를 문자열로 변환
                    sanitized[key] = ", ".join(str(item) for item in value) if value else ""
                # 딕셔너리 처리
                elif isinstance(value, dict):
                    # 딕셔너리를 문자열로 변환
                    sanitized[key] = str(value)
                # 기본 타입은 그대로 사용
                elif isinstance(value, (str, int, float, bool)):
                    sanitized[key] = value
                # 그 외 타입은 문자열로 변환
                else:
                    sanitized[key] = str(value)
                    
            sanitized_metadatas.append(sanitized)
        
        # 문서 ID 생성
        ids = [f"doc_{i}_{hash(text) % 10000000}" for i, text in enumerate(texts)]
        
        # 임베딩 계산
        if self.embeddings:
            try:
                embeddings = self.embeddings.embed_documents(texts)
                # 벡터 저장소에 추가
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=sanitized_metadatas  # 정리된 메타데이터 사용
                )
            except Exception as e:
                logger.error(f"벡터 저장소 추가 오류: {e}")
                # 오류 상세 출력
                import traceback
                logger.error(traceback.format_exc())
                
                # 개별 아이템 추가 시도
                successful_ids = []
                for i, (text, metadata) in enumerate(zip(texts, sanitized_metadatas)):
                    try:
                        text_embedding = self.embeddings.embed_documents([text])[0]
                        self.collection.add(
                            ids=[ids[i]],
                            embeddings=[text_embedding],
                            documents=[text],
                            metadatas=[metadata]
                        )
                        successful_ids.append(ids[i])
                        logger.info(f"단일 항목 추가 성공: {ids[i]}")
                    except Exception as item_error:
                        logger.error(f"단일 항목 추가 실패 ({ids[i]}): {item_error}")
                
                return successful_ids
        else:
            # 임베딩 객체가 없는 경우 내부 임베딩 사용
            try:
                self.collection.add(
                    ids=ids,
                    documents=texts,
                    metadatas=sanitized_metadatas  # 정리된 메타데이터 사용
                )
            except Exception as e:
                logger.error(f"벡터 저장소 추가 오류: {e}")
                # 오류 상세 출력
                import traceback
                logger.error(traceback.format_exc())
                
                # 개별 아이템 추가 시도
                successful_ids = []
                for i, (text, metadata) in enumerate(zip(texts, sanitized_metadatas)):
                    try:
                        self.collection.add(
                            ids=[ids[i]],
                            documents=[text],
                            metadatas=[metadata]
                        )
                        successful_ids.append(ids[i])
                        logger.info(f"단일 항목 추가 성공: {ids[i]}")
                    except Exception as item_error:
                        logger.error(f"단일 항목 추가 실패 ({ids[i]}): {item_error}")
                
                return successful_ids
        
        # 태그 인덱스 업데이트 (태그가 있는 경우)
        if self.tag_index:
            for i, metadata in enumerate(metadatas):
                if metadata and "tags" in metadata and metadata["tags"]:
                    doc_id = ids[i]
                    tags = metadata["tags"]
                    if isinstance(tags, str):
                        # 문자열인 경우 쉼표로 구분된 것으로 가정
                        tags = [tag.strip() for tag in tags.split(",")]
                    # 태그의 신뢰도를 1.0으로 설정
                    tag_scores = {tag: 1.0 for tag in tags}
                    self.tag_index.add_document(doc_id, tag_scores)
            
            # 변경사항 저장
            self.tag_index.save_index()
            logger.info(f"태그 인덱스 업데이트 완료: {len(ids)}개 문서")
        
        logger.info(f"{len(texts)}개 텍스트 추가 완료")
        return ids
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        문서 객체 리스트를 벡터 저장소에 추가
        
        Args:
            documents: 문서 객체 리스트 (content 및 metadata 필드 포함)
            
        Returns:
            List[str]: 추가된 문서 ID 리스트
        """
        texts = []
        metadatas = []
        
        for doc in documents:
            if "content" in doc:
                texts.append(doc["content"])
                metadatas.append(doc.get("metadata", {}))
            else:
                logger.warning(f"문서에 content 필드가 없습니다: {doc}")
        
        if not texts:
            logger.warning("추가할 텍스트가 없습니다.")
            return []
        
        return self.add_texts(texts, metadatas)
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        쿼리에 가장 관련성 높은 문서 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            
        Returns:
            List[Dict]: 관련 문서 리스트
        """
        try:
            logger.info(f"벡터 검색 쿼리: '{query}' (k={k})")
            logger.info(f"벡터 저장소 문서 수: {self.collection.count()}")
            
            # 임베딩 계산
            if self.embeddings:
                query_embedding = self.embeddings.embed_query(query)
                logger.info(f"쿼리 임베딩 계산 완료, 차원: {len(query_embedding)}")
                
                # 벡터 유사도 검색
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=k,
                    include=["documents", "metadatas", "distances"]
                )
            else:
                # 임베딩 객체가 없는 경우 내부 임베딩 사용
                logger.info("내부 임베딩 사용")
                results = self.collection.query(
                    query_texts=[query],
                    n_results=k,
                    include=["documents", "metadatas", "distances"]
                )
            
            # 검색 결과가 비어있는지 확인
            if not results["ids"][0]:
                logger.warning("검색 결과가 비었습니다.")
                return []
            
            # 검색 결과 변환
            documents = []
            for i in range(len(results["documents"][0])):
                score = 1.0 - float(results["distances"][0][i])  # 거리를 유사도 점수로 변환
                documents.append({
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": score
                })
                logger.info(f"검색 결과 {i+1}: 점수={score:.4f}, 내용={results['documents'][0][i][:50]}...")
            
            return documents
        except Exception as e:
            import traceback
            logger.error(f"검색 오류: {e}\n{traceback.format_exc()}")
            return []
    
    def search_with_tags(self, query: str, tags: List[str] = None, 
                       tag_boost: float = 0.3, k: int = 4) -> List[Dict[str, Any]]:
        """
        태그 기반으로 강화된 검색 수행
        
        Args:
            query: 검색 쿼리
            tags: 검색에 사용할 태그 리스트
            tag_boost: 태그 가중치 (0~1 사이)
            k: 반환할 결과 수
            
        Returns:
            List[Dict]: 검색 결과 리스트
        """
        if not tags or not self.tag_aware_search:
            return self.search(query, k=k)
        
        logger.info(f"태그 기반 검색: 쿼리='{query}', 태그={tags}, 가중치={tag_boost}")
        
        # 일반 벡터 검색 (더 많은 후보 결과 가져오기)
        candidates = self.search(query, k=k*2)
        
        if not candidates:
            logger.warning("검색 결과가 없습니다.")
            return []
        
        # 태그 기반 재순위화
        reranked_results = self.tag_aware_search.rerank_results_by_tags(
            candidates, tags, tag_boost
        )
        
        # 상위 k개 결과 반환
        return reranked_results[:k]
    
    def extract_query_tags(self, query: str) -> Tuple[str, List[str]]:
        """
        쿼리에서 태그 참조 추출
        
        Args:
            query: 사용자 쿼리
            
        Returns:
            Tuple[str, List[str]]: (정리된 쿼리, 추출된 태그 리스트)
        """
        # 태그 인식 검색이 없으면 원본 쿼리 반환
        if not self.tag_aware_search:
            return query, []
        
        # 태그 스키마 로드 시도
        try:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            if project_root not in sys.path:
                sys.path.append(project_root)
                
            from hongikjiki.tagging.tag_schema import TagSchema
            
            # 태그 스키마 파일 확인
            tag_schema_path = os.path.join(project_root, 'data', 'config', 'tag_schema.yaml')
            if os.path.exists(tag_schema_path):
                tag_schema = TagSchema(tag_schema_path)
                
                # 모든 태그 이름 가져오기
                all_tags = [tag.name for tag in tag_schema.get_all_tags()]
                
                # 태그 추출
                clean_query, extracted_tags = self.tag_aware_search.extract_tags_from_query(query, all_tags)
                logger.info(f"쿼리에서 태그 추출: {extracted_tags}")
                
                return clean_query, extracted_tags
        except Exception as e:
            logger.warning(f"쿼리에서 태그 추출 실패: {e}")
        
        return query, []
    
    def advanced_search(self, query: str, use_tags: bool = True, k: int = 4) -> List[Dict[str, Any]]:
        """
        태그 추출 및 태그 기반 검색을 포함한 고급 검색
        
        Args:
            query: 검색 쿼리
            use_tags: 태그 검색 사용 여부
            k: 반환할 결과 수
            
        Returns:
            List[Dict]: 검색 결과 리스트
        """
        if not use_tags or not self.tag_aware_search:
            return self.search(query, k=k)
        
        # 쿼리에서 태그 추출
        clean_query, tags = self.extract_query_tags(query)
        
        # 태그가 없으면 태그 추출 시도
        if not tags:
            try:
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
                if project_root not in sys.path:
                    sys.path.append(project_root)
                    
                from hongikjiki.tagging.tag_schema import TagSchema
                from hongikjiki.tagging.tag_extractor import TagExtractor
                
                # 태그 스키마 및 패턴 파일 확인
                tag_schema_path = os.path.join(project_root, 'data', 'config', 'tag_schema.yaml')
                tag_patterns_path = os.path.join(project_root, 'data', 'config', 'tag_patterns.json')
                
                if os.path.exists(tag_schema_path):
                    tag_schema = TagSchema(tag_schema_path)
                    tag_extractor = TagExtractor(
                        tag_schema, 
                        patterns_file=tag_patterns_path if os.path.exists(tag_patterns_path) else None
                    )
                    
                    # 쿼리 내용에서 태그 추출
                    tags = tag_extractor.extract_tags_from_query(clean_query)
            except Exception as e:
                logger.warning(f"쿼리에서 태그 추출기 사용 실패: {e}")
        
        # 태그 기반 검색 (태그가 있는 경우)
        if tags:
            logger.info(f"고급 검색 태그 사용: {tags}")
            return self.search_with_tags(clean_query, tags, k=k)
        else:
            return self.search(query, k=k)
    
    def count(self) -> int:
        """
        벡터 저장소의 문서 수 반환
        
        Returns:
            int: 저장된 문서 수
        """
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"문서 수 조회 오류: {e}")
            return 0
    
    def reset(self) -> None:
        """
        벡터 저장소 초기화
        모든 문서를 삭제하고 저장소를 비움
        """
        try:
            self.client.reset()
            logger.warning("벡터 저장소가 초기화되었습니다.")
            
            # 태그 인덱스도 초기화
            if self.tag_index:
                # 태그 인덱스 리셋 (빈 인덱스로 저장)
                self.tag_index = None
                from hongikjiki.vector_store.tag_index import TagIndex
                self.tag_index = TagIndex()
                self.tag_index.save_index()
                logger.warning("태그 인덱스가 초기화되었습니다.")
        except Exception as e:
            logger.error(f"벡터 저장소 초기화 오류: {e}")
            raise
    
    def get_similar_documents(self, text: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        주어진 텍스트와 유사한 문서 검색
        
        Args:
            text: 유사성 검색에 사용할 텍스트
            k: 반환할 결과 수
            
        Returns:
            List[Dict]: 유사한 문서 리스트
        """
        print("🧪 get_similar_documents() 호출됨")
        return self.search(text, k)
    
    def delete(self, document_ids: List[str]) -> None:
        """
        벡터 저장소에서 문서 삭제
        
        Args:
            document_ids: 삭제할 문서 ID 리스트
        """
        try:
            self.collection.delete(ids=document_ids)
            logger.info(f"{len(document_ids)}개 문서 삭제 완료")
            
            # 태그 인덱스에서도 삭제
            if self.tag_index:
                for doc_id in document_ids:
                    self.tag_index.remove_document(doc_id)
                self.tag_index.save_index()
                logger.info(f"태그 인덱스에서 {len(document_ids)}개 문서 삭제 완료")
        except Exception as e:
            logger.error(f"문서 삭제 오류: {e}")
            raise