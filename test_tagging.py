"""
태그 시스템 테스트 스크립트
"""
import os

def test_tagging():
    """간단한 태깅 테스트"""
    print("태그 시스템 테스트 시작...")
    
    # 입력 디렉토리 확인
    input_dir = 'data/tag_data/input_chunks'
    if not os.path.exists(input_dir):
        print(f"입력 디렉토리가 없습니다: {input_dir}")
        return
        
    # 파일 확인
    files = os.listdir(input_dir)
    print(f"파일 수: {len(files)}")
    
    # 파일 내용 읽기
    for file in files:
        if file.endswith('.txt'):
            file_path = os.path.join(input_dir, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    print(f"파일: {file}, 내용: {content[:50]}...")
                    
                    # 간단한 태그 추출
                    tags = []
                    if "홍익인간" in content:
                        tags.append("홍익인간")
                    if "자유의지" in content:
                        tags.append("자유의지")
                        
                    print(f"추출된 태그: {tags}")
                    
                    # 태그 저장
                    output_dir = 'data/tag_data/auto_tagged'
                    os.makedirs(output_dir, exist_ok=True)
                    
                    output_file = os.path.join(output_dir, f"{file}_tags.json")
                    with open(output_file, 'w', encoding='utf-8') as out_f:
                        out_f.write(f'{{"file": "{file}", "tags": {tags}}}')
                        
                    print(f"태그 저장 완료: {output_file}")
            except Exception as e:
                print(f"파일 처리 오류: {file}, {e}")
    
    print("태그 시스템 테스트 완료")

if __name__ == '__main__':
    test_tagging()
