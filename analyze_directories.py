# analyze_directories.py

import os
import json
import glob
from pathlib import Path

def analyze_directories(chunks_dir, tagged_dir):
    print("\n==== ANALYZING DIRECTORY STRUCTURES ====")
    
    # Analyze input chunks directory
    print(f"\n1. Analyzing original chunks directory: {chunks_dir}")
    chunk_files = []
    for root, dirs, files in os.walk(chunks_dir):
        for filename in files:
            if filename.endswith(".json"):
                chunk_files.append(os.path.join(root, filename))
    
    print(f"Total chunk files found: {len(chunk_files)}")
    
    # Check directory structure
    subdirs = [d for d in os.listdir(chunks_dir) if os.path.isdir(os.path.join(chunks_dir, d))]
    print(f"Subdirectories in chunks dir: {subdirs}")
    
    # Analyze a few original chunk files
    print("\nSample original chunk files:")
    sample_size = min(3, len(chunk_files))
    for i in range(sample_size):
        file_path = chunk_files[i]
        print(f"\nFile {i+1}: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"  Type: {type(data)}")
            if isinstance(data, dict):
                print(f"  Keys: {list(data.keys())}")
                if "content" in data:
                    content_len = len(data["content"]) if data["content"] else 0
                    print(f"  Content length: {content_len}")
                    if content_len > 0:
                        print(f"  Content preview: {data['content'][:50]}...")
                if "metadata" in data:
                    print(f"  Metadata keys: {list(data['metadata'].keys())}")
        except Exception as e:
            print(f"  Error analyzing file: {e}")
    
    # Analyze tagged directory
    print(f"\n2. Analyzing tagged documents directory: {tagged_dir}")
    tagged_files = [os.path.join(tagged_dir, f) for f in os.listdir(tagged_dir) if f.endswith(".json")]
    print(f"Total tagged files found: {len(tagged_files)}")
    
    print("\nSample tagged files:")
    sample_size = min(3, len(tagged_files))
    for i in range(sample_size):
        file_path = tagged_files[i]
        print(f"\nFile {i+1}: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"  Type: {type(data)}")
            if isinstance(data, dict):
                print(f"  Keys: {list(data.keys())}")
                if "document_id" in data:
                    print(f"  Document ID: {data['document_id']}")
                if "file" in data:
                    print(f"  File: {data['file']}")
        except Exception as e:
            print(f"  Error analyzing file: {e}")
    
    # Look for files in processed_chunks directly
    print("\n3. Checking for .txt files in input_chunks/processed_chunks")
    txt_files = glob.glob(f"{chunks_dir}/**/*.txt", recursive=True)
    print(f"Found {len(txt_files)} .txt files")
    if txt_files:
        print("Sample .txt files:")
        for file in txt_files[:3]:
            print(f"  {file}")
    
    print("\n==== ANALYSIS COMPLETE ====")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-dir", required=True, help="Directory containing original chunks")
    parser.add_argument("--tagged-dir", required=True, help="Directory containing tagged documents")
    
    args = parser.parse_args()
    analyze_directories(args.chunks_dir, args.tagged_dir)
    