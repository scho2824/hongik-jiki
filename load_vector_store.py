# load_vector_store.py

import os
from dotenv import load_dotenv
import logging

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataLoader")

def load_documents():
    """정법 문서를 로드하여 벡터 저장소에 저장"""
    try:
        # 수정된 임포트 경로 사용
        from hongikjiki.modules.text_processing.document_processor import DocumentProcessor
        from hongikjiki.modules.vector_store.chroma_store import ChromaVectorStore
        from hongikjiki.modules.vector_store.embeddings import get_embeddings
        
        # 문서 프로세서 초기화
        processor = DocumentProcessor()
        
        # 정법 강의 데이터 디렉토리
        data_dir = "data/jungbub_teachings"
        
        # 디렉토리가 존재하는지 확인
        if not os.path.exists(data_dir):
            logger.error(f"데이터 디렉토리가 존재하지 않습니다: {data_dir}")
            logger.info("데이터 디렉토리를 생성합니다.")
            os.makedirs(data_dir, exist_ok=True)
            logger.info("샘플 정법 문서를 생성합니다.")
            create_sample_documents(data_dir)
        
        # 문서 처리
        logger.info(f"'{data_dir}' 디렉토리에서 문서를 처리합니다...")
        chunks = processor.process_directory(data_dir)
        logger.info(f"{len(chunks)}개의 청크가 생성되었습니다.")
        
        if not chunks:
            logger.warning("처리된 문서가 없습니다. 샘플 문서를 생성합니다.")
            create_sample_documents(data_dir)
            chunks = processor.process_directory(data_dir)
        
        # 임베딩 모델 로드
        embeddings = get_embeddings("openai")
        
        # 벡터 저장소 초기화
        vector_store = ChromaVectorStore(
            collection_name="hongikjiki_jungbub",
            persist_directory="data/vector_store",
            embeddings=embeddings
        )
        
        # 문서 추가
        logger.info("벡터 저장소에 문서를 추가합니다...")
        vector_store.add_documents(chunks)
        
        logger.info("벡터 저장소 구축 완료!")
        return True
    
    except Exception as e:
        logger.error(f"문서 로드 오류: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def create_sample_documents(data_dir):
    """샘플 정법 문서 생성"""
    sample_docs = [
        {
            "title": "정법의 기본 개념",
            "content": """
            정법은 천공 스승님께서 알려주신 우주 법칙에 대한 가르침입니다.
            정법은 인간이 본래의 자기 모습을 회복하고 홍익인간의 삶을 살아가는 방법을 제시합니다.
            정법에서는 자연의 법칙과 조화를 이루며 살아가는 것을 중요시합니다.
            또한 내면의 성장과 깨달음을 통해 진정한 자유를 얻는 것을 목표로 합니다.
            정법은 이론적인 가르침에 그치지 않고 일상 속에서의 실천을 강조합니다.
            """
        },
        {
            "title": "홍익인간의 의미",
            "content": """
            홍익인간이란 널리 인간을 이롭게 한다는 의미입니다.
            이는 단순히 타인에게 봉사하는 것이 아니라, 자신과 타인 모두가 함께 성장하고 발전할 수 있도록 하는 것입니다.
            진정한 홍익인간이 되기 위해서는 먼저 자신의 내면을 바로 세우고, 자주독립의 힘을 길러야 합니다.
            홍익의 정신은 한민족의 건국이념이자, 인류 공동체의 보편적 가치입니다.
            정법에서는 홍익인간의 정신을 실현하는 구체적인 방법을 제시합니다.
            """
        }
    ]
    
    for i, doc in enumerate(sample_docs):
        file_path = os.path.join(data_dir, f"정법기본_{i+1}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"제목: {doc['title']}\n\n")
            f.write(doc['content'].strip())
        logger.info(f"샘플 문서 생성: {file_path}")

if __name__ == "__main__":
    success = load_documents()
    print(f"데이터 로드 결과: {'성공' if success else '실패'}")