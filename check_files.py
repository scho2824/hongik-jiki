# check_files.py
from pathlib import Path
import os

# Define paths
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data" / "jungbub_teachings"

# List supported extensions
SUPPORTED_EXTENSIONS = ['.txt', '.pdf', '.docx', '.rtf', '.md']

def count_files_by_extension(directory):
    """Count files by extension in directory and subdirectories"""
    result = {}
    total = 0
    
    for root, _, files in os.walk(directory):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                path = os.path.join(root, file)
                rel_path = os.path.relpath(path, directory)
                result[ext] = result.get(ext, 0) + 1
                total += 1
                # Print a sample of paths to verify
                if result[ext] <= 3:  # Only show first 3 of each type
                    print(f"{ext}: {rel_path}")
    
    return result, total

print(f"Checking files in {DATA_DIR}...")
if not DATA_DIR.exists():
    print(f"Error: Directory {DATA_DIR} does not exist!")
else:
    counts, total = count_files_by_extension(DATA_DIR)
    print(f"Found {total} supported files:")
    for ext, count in counts.items():
        print(f"  {ext}: {count} files")