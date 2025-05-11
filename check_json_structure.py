import json

file_path = "data/qa/high_insight_qa_dataset.json"

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)  # JSON 배열 시도
    print(f"✅ JSON 배열 형식으로 로드됨: 총 {len(data)}개 항목")
except json.JSONDecodeError:
    print("⚠️ JSON 배열 형식이 아님. jsonl 형식으로 재시도 중...")
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                    data.append(obj)
                except json.JSONDecodeError:
                    print(f"❌ {i+1}번째 줄 파싱 실패: {line[:50]}...")
    print(f"✅ jsonl 형식으로 {len(data)}개 항목 로드 완료")