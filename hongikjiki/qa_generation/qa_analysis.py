import json
import csv
from pathlib import Path
from hongikjiki.tagging.tag_schema import TagSchema
from hongikjiki.tagging.tag_extractor import TagExtractor

# 파일 경로 설정
qa_file = "data/qa/jungbub_qa_dataset.json"
output_csv = "data/qa/qa_tag_evaluation.csv"
schema_file = "data/config/tag_schema.yaml"
patterns_file = "data/config/tag_patterns.json"

# 태그 스키마 및 추출기 로드
tag_schema = TagSchema.load_from_file(schema_file)
tag_extractor = TagExtractor(tag_schema, patterns_file)

# QA 데이터 불러오기
with open(qa_file, "r", encoding="utf-8") as f:
    qa_data = json.load(f)

# 분석 결과 저장 리스트
results = []

# 각 QA 항목마다 near-threshold 태그 평가
for item in qa_data:
    question = item.get("question", "")
    answer = item.get("answer", "")
    tags = item.get("tags", [])
    
    near_candidates = tag_extractor.log_near_threshold_candidates(answer)
    top_candidates = sorted(near_candidates, key=lambda x: x[1], reverse=True)[:3]

    results.append({
        "question": question,
        "answer_snippet": answer[:60].replace("\n", " "),
        "current_tags": ", ".join(tags),
        "near_tags": ", ".join(f"{tag}:{score:.2f}" for tag, score in top_candidates)
    })

# CSV로 저장
Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
with open(output_csv, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["question", "answer_snippet", "current_tags", "near_tags"])
    writer.writeheader()
    writer.writerows(results)

print(f"✅ QA 태그 분석 결과 저장 완료 → {output_csv}")
