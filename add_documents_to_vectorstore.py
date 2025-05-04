# add_documents_to_vectorstore.py
"""
벡터 저장소에 문서 추가 및 태그 할당
"""
import os
import json
import logging
logging.basicConfig(level=logging.INFO)

from hongikjiki.chatbot import HongikJikiChatBot
from hongikjiki.text_processing.document_processor import DocumentProcessor

def main():
    """벡터 저장소에 문서 추가"""
    print("문서 추가 시작...")
    
    # 챗봇 및 문서 처리기 초기화
    chatbot = HongikJikiChatBot()
    
    # 태그된 문서 정보 불러오기
    tag_dir = 'data/tag_data/auto_tagged'
    input_dir = 'data/tag_data/input_chunks'
    
    # 태그 정보 수집
    tagged_files = {}
    for file in os.listdir(tag_dir):
        if file.endswith('_tags.json'):
            try:
                with open(os.path.join(tag_dir, file), 'r') as f:
                    tag_data = json.load(f)
                    source_file = tag_data.get('file')
                    tags = tag_data.get('tags', [])
                    if source_file and tags:
                        tagged_files[source_file] = tags
            except Exception as e:
                print(f"태그 파일 처리 오류: {file}, {e}")
    
    # 문서 추가
    for file in os.listdir(input_dir):
        if file in tagged_files:
            file_path = os.path.join(input_dir, file)
            tags = tagged_files[file]
            
            try:
                # 파일 읽기
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # 메타데이터 생성
                metadata = {
                    "filename": file,
                    "tags": tags,
                    "title": f"테스트 문서: {file}"
                }
                
                # 벡터 저장소에 추가
                doc_item = {
                    "content": content,
                    "metadata": metadata
                }
                
                # 벡터 저장소에 문서 추가
                chatbot.vector_store.add_documents([doc_item])
                print(f"문서 추가 완료: {file}, 태그: {tags}")
                
            except Exception as e:
                print(f"문서 추가 오류: {file}, {e}")
    
    # 벡터 저장소 문서 수 확인
    doc_count = chatbot.vector_store.count()
    print(f"벡터 저장소 문서 수: {doc_count}")
    
    print("문서 추가 완료")

if __name__ == "__main__":
    main()