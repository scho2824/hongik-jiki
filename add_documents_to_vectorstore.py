# add_documents_to_vectorstore.py
"""
벡터 저장소에 문서 추가 및 태그 할당
"""
import os
import json
import logging
import sys
logging.basicConfig(level=logging.INFO)

# 환경 변수 로드
from hongikjiki.utils import load_dotenv
load_dotenv()

# API 키 확인
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("오류: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
    print("다음 방법 중 하나로 API 키를 설정하세요:")
    print("1. export OPENAI_API_KEY=your_api_key")
    print("2. .env 파일에 OPENAI_API_KEY=your_api_key 추가")
    sys.exit(1)

print(f"API 키 확인: {api_key[:5]}...{api_key[-5:]}")

from hongikjiki.chatbot import HongikJikiChatBot
from hongikjiki.text_processing.document_processor import DocumentProcessor

def main():
    """벡터 저장소에 문서 추가"""
    print("문서 추가 시작...")

    # 챗봇 및 문서 처리기 초기화 - API 키 명시적 전달
    chatbot = HongikJikiChatBot(
        llm_kwargs={"api_key": api_key}
    )
    
    # 태그된 문서 정보 불러오기
    tag_dir = 'data/tag_data/auto_tagged'
    input_dir = 'data/tag_data/input_chunks'

    # 태그 패턴 파일 경로 지정
    os.environ["TAG_PATTERNS_PATH"] = "data/config/tag_patterns.json"
    
    # 태그 정보 수집
    tagged_files = {}
    print(f"태그 디렉토리: {tag_dir}")

    for file in os.listdir(tag_dir):
        if '_tags' in file and file.endswith('.json'):
            try:
                file_path = os.path.join(tag_dir, file)
                print(f"태그 파일 처리 중: {file}")

                with open(file_path, 'r') as f:
                    tag_data = json.load(f)
                    # 파일 이름 정리 (태그 파일 이름에서 "_tags" 등 제거)
                    base_file = file.split('_tags')[0] + '.json'

                    # 태그 정보 추출 - 여러 형식 지원
                    tags = []
                    if isinstance(tag_data, dict):
                        tags = tag_data.get('tags', [])
                        source_file = tag_data.get('file', base_file)
                    elif isinstance(tag_data, list):
                        tags = tag_data
                        source_file = base_file

                    if tags:
                        tagged_files[source_file] = tags
                        print(f"  - 태그 추출 성공: {source_file}, 태그: {tags}")
            except Exception as e:
                print(f"태그 파일 처리 오류: {file}, {e}")
    
    # 문서 추가
    print(f"\n입력 디렉토리: {input_dir}")
    print(f"태그가 있는 파일 수: {len(tagged_files)}")

    input_files = os.listdir(input_dir)
    print(f"입력 디렉토리 파일 수: {len(input_files)}")

    # 처리할 파일 최대 개수 제한 (너무 많으면 오래 걸림)
    max_files = 50
    files_to_process = input_files[:max_files]
    print(f"처리할 파일 수: {len(files_to_process)} (최대 {max_files}개)")

    success_count = 0
    error_count = 0

    for file in files_to_process:
        file_path = os.path.join(input_dir, file)

        # JSON 파일인지 확인
        if not file.endswith('.json'):
            continue

        try:
            # JSON 파일 읽기
            with open(file_path, 'r') as f:
                file_data = json.load(f)

            # 필요한 데이터 추출
            content = ""
            if isinstance(file_data, dict):
                content = file_data.get('content', file_data.get('text', ''))
            elif isinstance(file_data, str):
                content = file_data

            if not content:
                print(f"⚠️ 내용 없음: {file}")
                continue

            # 태그 가져오기 (없으면 기본 태그 사용)
            tags = tagged_files.get(file, ["일반"])

            # 메타데이터 생성
            metadata = {
                "filename": file,
                "tags": tags,
                "source_id": file.split('.')[0],
                "title": f"정법 문서: {file}"
            }

            # 벡터 저장소에 추가
            doc_item = {
                "content": content,
                "metadata": metadata
            }

            # 벡터 저장소에 문서 추가
            chatbot.vector_store.add_documents([doc_item])
            success_count += 1
            print(f"✅ 문서 추가 완료: {file}, 태그: {tags}")

        except Exception as e:
            error_count += 1
            print(f"❌ 문서 추가 오류: {file}, {e}")

    print(f"\n문서 추가 결과: 성공 {success_count}, 실패 {error_count}")
    
    # 벡터 저장소 문서 수 확인
    doc_count = chatbot.vector_store.count()
    print(f"벡터 저장소 문서 수: {doc_count}")
    
    print("문서 추가 완료")

if __name__ == "__main__":
    main()