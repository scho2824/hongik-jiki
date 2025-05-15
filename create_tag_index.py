# create_tag_index.py

import os
import json
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TagIndexCreator")

def create_tag_index():
    """기본 태그 인덱스 파일 생성"""
    try:
        # 디렉토리 경로 설정
        base_dir = os.path.dirname(os.path.abspath(__file__))
        tag_dir = os.path.join(base_dir, "hongikjiki", "data", "tag_data")
        
        # 디렉토리 생성
        os.makedirs(tag_dir, exist_ok=True)
        
        # 태그 인덱스 파일 경로
        tag_index_path = os.path.join(tag_dir, "tag_index.json")
        
        # 기본 태그 인덱스 데이터
        tag_index_data = {
            "tags": {
                "정법": {"count": 5, "documents": []},
                "홍익인간": {"count": 3, "documents": []},
                "자연": {"count": 2, "documents": []},
                "수행": {"count": 1, "documents": []},
                "실천": {"count": 2, "documents": []}
            },
            "documents": {},
            "updated_at": "2024-05-14T12:00:00"
        }
        
        # 파일 저장
        with open(tag_index_path, "w", encoding="utf-8") as f:
            json.dump(tag_index_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"태그 인덱스 파일 생성 완료: {tag_index_path}")
        return True
        
    except Exception as e:
        logger.error(f"태그 인덱스 파일 생성 오류: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = create_tag_index()
    print(f"태그 인덱스 파일 생성 결과: {'성공' if success else '실패'}")