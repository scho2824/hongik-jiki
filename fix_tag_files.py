"""
태그 JSON 파일 형식 수정
"""
import os
import json

def main():
    """태그 파일 형식 수정"""
    print("태그 파일 수정 시작...")
    
    tag_dir = 'data/tag_data/auto_tagged'
    
    # 각 태그 파일 처리
    for file in os.listdir(tag_dir):
        if file.endswith('_tags.json'):
            file_path = os.path.join(tag_dir, file)
            try:
                # 파일 내용 읽기
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # 잘못된 형식 수정
                # 예: '{"file": "test1.txt", "tags": ['홍익인간']}' 형식 수정
                source_file = file.replace('_tags.json', '')
                
                # 태그 추출
                if "홍익인간" in content:
                    tags = ["홍익인간"]
                elif "자유의지" in content:
                    tags = ["자유의지"]
                else:
                    tags = []
                
                # 올바른 JSON 형식으로 저장
                corrected_data = {
                    "file": source_file,
                    "tags": tags
                }
                
                with open(file_path, 'w') as f:
                    json.dump(corrected_data, f, ensure_ascii=False, indent=2)
                
                print(f"파일 수정 완료: {file}")
                
            except Exception as e:
                print(f"파일 수정 오류: {file}, {e}")
    
    print("태그 파일 수정 완료")

if __name__ == "__main__":
    main()
