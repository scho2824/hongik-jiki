# fix_metadata_handling.py
import os
import re

# chroma_store.py 파일 경로
file_path = "hongikjiki/vector_store/chroma_store.py"

# 파일 읽기
with open(file_path, "r") as f:
    content = f.read()

# 위에서 제공한 개선된 add_texts 메서드 코드를 여기에 문자열로 붙여넣기
improved_code = """def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
    \"\"\"
    텍스트 리스트를 벡터 저장소에 추가
    
    Args:
        texts: 추가할 텍스트 리스트
        metadatas: 각 텍스트에 대한 메타데이터 리스트 (옵션)
        
    Returns:
        List[str]: 추가된 문서 ID 리스트
    \"\"\"
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
    
    logger.info(f"{len(texts)}개 텍스트 추가 완료")
    return ids"""

# add_texts 메서드 찾아 교체
pattern = r"def add_texts\(.*?return ids"
flags = re.DOTALL  # . 문자가 줄바꿈도 포함하도록 설정
modified_content = re.sub(pattern, improved_code, content, flags=flags)

# 파일 쓰기
with open(file_path, "w") as f:
    f.write(modified_content)

print(f"{file_path} 파일이 성공적으로 수정되었습니다.")