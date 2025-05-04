# copy_files.py라는 파일 생성 후 아래 내용 붙여넣기
import os
import shutil

# 소스 및 대상 디렉토리
source_dir = 'data/jungbub_teachings'
target_dir = 'data/tag_data/input_chunks'

# 대상 디렉토리 생성
os.makedirs(target_dir, exist_ok=True)

# 파일 복사
file_count = 0
if os.path.exists(source_dir):
    print(f'Processing files from {source_dir}...')
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(('.txt', '.md', '.rtf', '.pdf')):
                src_path = os.path.join(root, file)
                dst_path = os.path.join(target_dir, file)
                # 파일 복사
                try:
                    shutil.copy2(src_path, dst_path)
                    file_count += 1
                    print(f'Copied: {file}')
                except Exception as e:
                    print(f'Error copying {file}: {e}')
    
    print(f'Copied {file_count} files to {target_dir}')
else:
    print(f'Source directory {source_dir} not found!')