#!/bin/bash

echo "🔁 Starting full Hongik-Jiki pipeline..."
echo "🧹 Cleaning previous chunks..."
rm -rf data/tag_data/input_chunks/*
python3 hongikjiki/pipeline/run_all_pipeline.py

if [ $? -eq 0 ]; then
    echo "✅ Pipeline completed successfully."
else
    echo "❌ Pipeline failed."
fi