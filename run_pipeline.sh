#!/bin/bash

# 로그 파일 설정
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="logs/pipeline_run_${timestamp}.log"
mkdir -p logs

echo "🔁 Starting full Hongik-Jiki pipeline..."
echo "📋 Current document statistics:"
echo "- Original documents: $(find data/jungbub_teachings -type f | wc -l)"
echo "- Processed documents: $(jq '. | length' data/processed/jungbub_dataset.json 2>/dev/null || echo 0)"
echo "- Current vector store size: $(du -sh data/vector_store 2>/dev/null | cut -f1 || echo 'N/A')"

# 로그 파일에도 기록하고 터미널에도 출력
python3 hongikjiki/pipeline/run_all_pipeline.py 2>&1 | tee -a "$log_file"

pipeline_exit_code=${PIPESTATUS[0]}

if [ $pipeline_exit_code -eq 0 ]; then
    echo "✅ Pipeline completed successfully."
    echo "- Updated processed documents: $(jq '. | length' data/processed/jungbub_dataset.json 2>/dev/null || echo 0)"
    echo "- Updated vector store size: $(du -sh data/vector_store 2>/dev/null | cut -f1 || echo 'N/A')"
    echo "📝 Log file: $log_file"
else
    echo "❌ Pipeline failed. Check log for details."
    echo "📝 Log file: $log_file"
    exit 1
fi