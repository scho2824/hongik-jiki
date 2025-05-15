import json
import csv
from pathlib import Path
from hongikjiki.modules.tagging.tag_schema import TagSchema
from hongikjiki.modules.tagging.tag_extractor import TagExtractor

def main():
    # 파일 경로 설정
    qa_file = "data/qa/jungbub_qa_dataset.json"
    output_csv = "data/qa/qa_tag_evaluation.csv"
    schema_file = "data/converted_tag_schema.yaml"
    # 태그 스키마 및 추출기 로드
    tag_schema = TagSchema.load_from_yaml(schema_file)
    tag_extractor = TagExtractor(tag_schema)

    # QA 데이터 불러오기
    with open(qa_file, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    # 분석 결과 저장 리스트
    results = []

    # 각 QA 항목마다 near-threshold 태그 평가
    for i, item in enumerate(qa_data):
        if i % 100 == 0 and i > 0:
            print(f"🔍 {i}개 처리 중...")
        question = item.get("question", "")
        answer = item.get("answer", "")
        tags = item.get("tags", [])
        
        near_candidates = tag_extractor.log_near_threshold_candidates(answer)
        top_candidates = sorted(near_candidates, key=lambda x: x[1], reverse=True)[:3]

        results.append({
            "question": question,
            "answer_excerpt": answer[:60].replace("\n", " "),
            "current_tags": ", ".join(tags),
            "tag_suggestions": ", ".join(f"{tag}:{score:.2f}" for tag, score in top_candidates)
        })

    # CSV로 저장
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer_excerpt", "current_tags", "tag_suggestions"])
        writer.writeheader()
        writer.writerows(results)

    print(f"✅ QA 태그 분석 결과 저장 완료 → {output_csv}")

if __name__ == "__main__":
    main()
