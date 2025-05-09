from datasets import Dataset
import json

# 1. QA 데이터 불러오기
with open("data/qa/jungbub_qa_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 2. Hugging Face Dataset 객체 생성
dataset = Dataset.from_list(data)

# 3. Hugging Face에 업로드
dataset.push_to_hub("scho2824/hongikjiki-qa")
