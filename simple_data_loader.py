# simple_data_loader.py

import os
from dotenv import load_dotenv
import logging

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SimpleDataLoader")

def load_sample_data():
    """벡터 저장소에 직접 샘플 데이터 추가"""
    try:
        # 수정된 임포트 경로 사용
        from hongikjiki.modules.vector_store.chroma_store import ChromaVectorStore
        from hongikjiki.modules.vector_store.embeddings import get_embeddings
        
        # 샘플 문서 준비
        sample_docs = [
            {
                "content": "정법은 천공 스승님께서 알려주신 우주 법칙에 대한 가르침입니다. 정법은 인간이 본래의 자기 모습을 회복하고 홍익인간의 삶을 살아가는 방법을 제시합니다.",
                "metadata": {
                    "title": "정법의 기본 개념",
                    "lecture_number": 1,
                    "lecture_id": "001",
                    "lecture_title": "정법이란 무엇인가",
                    "source": "정법기본_1.txt"
                }
            },
            {
                "content": "홍익인간이란 널리 인간을 이롭게 한다는 의미입니다. 이는 단순히 타인에게 봉사하는 것이 아니라, 자신과 타인 모두가 함께 성장하고 발전할 수 있도록 하는 것입니다.",
                "metadata": {
                    "title": "홍익인간의 의미",
                    "lecture_number": 2,
                    "lecture_id": "002",
                    "lecture_title": "홍익인간이란",
                    "source": "정법기본_2.txt"
                }
            },
            {
                "content": "진정한 홍익인간이 되기 위해서는 먼저 자신의 내면을 바로 세우고, 자주독립의 힘을 길러야 합니다. 홍익의 정신은 한민족의 건국이념이자, 인류 공동체의 보편적 가치입니다.",
                "metadata": {
                    "title": "홍익인간의 실천",
                    "lecture_number": 3,
                    "lecture_id": "003",
                    "lecture_title": "홍익인간의 실천",
                    "source": "정법기본_3.txt"
                }
            },
            {
                "content": "정법에서는 자연의 법칙과 조화를 이루며 살아가는 것을 중요시합니다. 또한 내면의 성장과 깨달음을 통해 진정한 자유를 얻는 것을 목표로 합니다.",
                "metadata": {
                    "title": "자연과 조화",
                    "lecture_number": 4,
                    "lecture_id": "004",
                    "lecture_title": "자연의 법칙과 조화",
                    "source": "정법기본_4.txt"
                }
            },
            {
                "content": "정법은 이론적인 가르침에 그치지 않고 일상 속에서의 실천을 강조합니다. 매 순간 자신의 생각과 행동을 돌아보며 깨어있는 상태를 유지하는 것이 중요합니다.",
                "metadata": {
                    "title": "일상 속 실천",
                    "lecture_number": 5,
                    "lecture_id": "005",
                    "lecture_title": "일상 속 정법 실천",
                    "source": "정법기본_5.txt"
                }
            }
        ]
        
        # 벡터 저장소 디렉토리 확인
        vector_store_dir = "data/vector_store"
        os.makedirs(vector_store_dir, exist_ok=True)
        
        # 임베딩 모델 로드
        logger.info("임베딩 모델을 로드합니다...")
        embeddings = get_embeddings("openai")
        
        # 벡터 저장소 초기화
        logger.info("벡터 저장소를 초기화합니다...")
        vector_store = ChromaVectorStore(
            collection_name="hongikjiki_jungbub",
            persist_directory=vector_store_dir,
            embeddings=embeddings
        )
        
        # 문서 추가
        logger.info("벡터 저장소에 샘플 문서를 추가합니다...")
        # 문서 형식 생성
        docs = []
        for doc in sample_docs:
            docs.append({
                "content": doc["content"],
                "metadata": doc["metadata"]
            })
        
        # 문서 추가
        vector_store.add_documents(docs)
        
        # 저장
        logger.info("벡터 저장소에 변경사항을 저장합니다...")
        if hasattr(vector_store, "persist"):
            vector_store.persist()
        
        # 문서 개수 확인
        count = vector_store.count()
        logger.info(f"벡터 저장소에 총 {count}개의 문서가 있습니다.")
        
        logger.info("샘플 데이터 로드 완료!")
        return True
        
    except Exception as e:
        logger.error(f"샘플 데이터 로드 오류: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = load_sample_data()
    print(f"샘플 데이터 로드 결과: {'성공' if success else '실패'}")