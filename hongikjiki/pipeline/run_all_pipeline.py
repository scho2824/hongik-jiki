


import os
import subprocess

def run_step(description, command):
    print(f"\n🔹 {description}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"❌ Failed: {description}")
        exit(1)
    else:
        print(f"✅ Completed: {description}")

if __name__ == "__main__":
    print("🚀 Starting Full Hongik-Jiki Pipeline")

    run_step("Step 1: Chunking documents", "python3 hongikjiki/pipeline/run_chunking_pipeline.py")
    run_step("Step 2: Tagging chunks", "python3 hongikjiki/pipeline/run_tagging.py")
    run_step("Step 3: Merging tagged chunks", "python3 hongikjiki/utils/merge_chunks.py --input-dir data/tag_data/auto_tagged --output-file data/processed/jungbub_dataset.json")
    run_step("Step 4: Generating QA dataset", "python3 hongikjiki/qa_generation/generate_qa.py --input_file data/processed/jungbub_dataset.json --output_file data/qa/jungbub_qa_dataset.json")
    run_step("Step 5: Building vector store", "python3 hongikjiki/scripts/build_vector_store.py --qa_file data/qa/jungbub_qa_dataset.json")

    print("\n🎉 All steps completed successfully.")