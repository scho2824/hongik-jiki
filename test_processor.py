import os
import logging
import json
from hongikjiki.text_processor import HongikJikiTextProcessor

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    # 테스트할 디렉토리 경로 설정
    documents_dir = "data/jungbub_teachings"
    
    # 출력 디렉토리 설정
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 텍스트 프로세서 인스턴스 생성
    processor = HongikJikiTextProcessor()
    
    # 문서 로드
    documents = processor.load_documents(documents_dir)
    print(f"로드된 문서 수: {len(documents)}")
    
    # 첫 번째 문서의 메타데이터 출력
    if documents:
        print("\n첫 번째 문서 메타데이터:")
        print(json.dumps(documents[0]["metadata"], indent=2, ensure_ascii=False))
        
        # 첫 번째 문서의 내용 일부 출력
        content = documents[0]["content"]
        print(f"\n첫 번째 문서 내용 미리보기 (처음 200자):\n{content[:200]}...")
    
    # 문서 분할
    chunks = processor.split_documents(documents)
    print(f"\n생성된 청크 수: {len(chunks)}")
    
    # 첫 번째 청크 출력
    if chunks:
        print("\n첫 번째 청크 메타데이터:")
        print(json.dumps(chunks[0]["metadata"], indent=2, ensure_ascii=False))
        
        # 첫 번째 청크의 내용 출력
        content = chunks[0]["content"]
        print(f"\n첫 번째 청크 내용 미리보기 (처음 200자):\n{content[:200]}...")
    
    # 결과 저장
    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        # 모든 문서의 메타데이터만 저장
        metadata_list = [doc["metadata"] for doc in documents]
        json.dump(metadata_list, f, indent=2, ensure_ascii=False)
    
    print(f"\n메타데이터가 {metadata_file}에 저장되었습니다.")
    
    # 청크 저장
    chunks_file = os.path.join(output_dir, "chunks.json")
    with open(chunks_file, 'w', encoding='utf-8') as f:
        # 모든 청크 정보 저장
        chunks_data = [
            {
                "content": chunk["content"][:200] + "...",  # 내용 일부만 저장
                "metadata": chunk["metadata"]
            }
            for chunk in chunks
        ]
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)
    
    print(f"청크 정보가 {chunks_file}에 저장되었습니다.")

if __name__ == "__main__":
    main()