"""
문서 관리 모듈 테스트
"""
import logging
from pathlib import Path
from pprint import pprint
from hongikjiki.document_manager import list_documents, detect_changed_files, update_documents

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_list_documents(data_dir="./data/jungbub_teachings"):
    """문서 목록 테스트"""
    print("\n=== 문서 목록 테스트 ===")
    documents = list_documents(data_dir)
    print(f"총 {len(documents)}개 문서 발견")
    
    # 일부 문서 정보 출력
    if documents:
        print("\n첫 번째 문서 정보:")
        pprint(documents[0])
    
    return documents

def test_detect_changed_files(data_dir="./data/jungbub_teachings"):
    """변경된 파일 감지 테스트"""
    print("\n=== 변경된 파일 감지 테스트 ===")
    changed_files = detect_changed_files(data_dir)
    
    print(f"추가된 파일: {len(changed_files['added'])}")
    print(f"수정된 파일: {len(changed_files['modified'])}")
    print(f"삭제된 파일: {len(changed_files['deleted'])}")
    
    # 추가된 파일 목록 출력
    if changed_files["added"]:
        print("\n추가된 파일 목록:")
        for file_path in changed_files["added"][:5]:  # 최대 5개만 출력
            print(f"  - {file_path}")
    
    return changed_files

def test_update_documents(data_dir="./data/jungbub_teachings", force_reindex=False):
    """문서 업데이트 테스트"""
    print("\n=== 문서 업데이트 테스트 ===")
    print(f"강제 재색인화: {force_reindex}")
    
    result = update_documents(data_dir, force_reindex=force_reindex)
    
    print("\n업데이트 결과:")
    pprint(result)
    
    return result

if __name__ == "__main__":
    # 테스트할 디렉토리 경로
    data_dir = "./data/jungbub_teachings"
    
    if not Path(data_dir).exists():
        print(f"테스트 디렉토리를 찾을 수 없습니다: {data_dir}")
    else:
        # 문서 목록 테스트
        test_list_documents(data_dir)
        
        # 변경된 파일 감지 테스트
        test_detect_changed_files(data_dir)
        
        # 문서 업데이트 테스트 (선택적으로 강제 재색인화)
        # test_update_documents(data_dir, force_reindex=True)