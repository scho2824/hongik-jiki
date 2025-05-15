import json
import os
import re
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from hongikjiki.modules.tagging.tag_schema import TagSchema

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
tag_schema = TagSchema.load_from_yaml("data/converted_tag_schema.yaml")

def parse_json_from_response(response_content: str) -> dict:
    """Extract and parse JSON string from OpenAI response."""
    content = re.sub(r"^```(?:json)?\s*", "", response_content)
    content = re.sub(r"\s*```$", "", content).strip()
    json_start = content.find("{")
    json_end = content.rfind("}")
    if json_start != -1 and json_end != -1:
        json_str = content[json_start:json_end+1]
    else:
        json_str = content
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print("❌ JSON 파싱 오류:", e)
        print("⛔ 파싱 실패 응답 내용:", response_content)
        return {
            "insight_summary": "",
            "keywords": [],
            "context_note": "JSON 파싱 실패"
        }

def enhance_qa(qa):
    prompt = f"""다음은 통찰을 담은 문장입니다:

"{qa['quoted_insight']}"

이 인용문을 바탕으로 다음 정보를 작성하세요:
1. 핵심 요지를 1줄로 요약한 'insight_summary'
2. 이 통찰에서 중요한 2~4개의 핵심 단어로 구성된 'keywords'
3. 이 통찰의 맥락을 자연스럽게 설명한 'context_note' (1-2문장)

다음과 같은 JSON 형식으로 답변하세요:
{{
  "insight_summary": "...",
  "keywords": ["...", "..."],
  "context_note": "..."
}}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        content = response.choices[0].message.content or ""
        parsed = parse_json_from_response(content)
        qa.update(parsed)
        if "tags" in qa:
            tag_descriptions = {}
            for tag in qa["tags"]:
                tag_obj = tag_schema.tags.get(tag)
                tag_descriptions[tag] = tag_obj.description if tag_obj and hasattr(tag_obj, "description") else "이 개념"
            qa["tag_descriptions"] = tag_descriptions
    except Exception as e:
        print("❌ 기타 오류 발생:", e)
        qa.update({
            "insight_summary": "",
            "keywords": [],
            "context_note": "자동 생성 실패"
        })
    return qa

def prepare_metadata(qa, lecture_id, lecture_title):
    """Update lecture_id, title, and metadata block."""
    qa["lecture_id"] = lecture_id if lecture_id is not None else qa.get("lecture_id", "")
    qa["lecture_title"] = lecture_title or qa.get("lecture_title", "")
    qa["metadata"] = {
        "lecture_id": qa["lecture_id"],
        "lecture_title": qa["lecture_title"]
    }

def process_file(input_path, output_path, lecture_id=None, lecture_title=""):
    with open(input_path, "r", encoding="utf-8") as infile:
        qa_data = json.load(infile)

    # ------------------------------------------------------------------------
    # Populate metadata fields from arguments or default to empty
    for qa in qa_data:
        prepare_metadata(qa, lecture_id, lecture_title)
    # ------------------------------------------------------------------------

    enhanced = []
    # 병렬 처리로 OpenAI 호출 속도 향상
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(enhance_qa, qa): qa for qa in qa_data}
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                result = future.result()
                enhanced.append(result)
            except Exception as e:
                # 오류 발생 시 원본 qa 유지
                qa = futures[future]
                print("❌ 병렬 처리 오류:", e)
                enhanced.append(qa)

    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(enhanced, outfile, ensure_ascii=False, indent=2)

    print(f"✅ 고밀도 QA 저장 완료 → {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--lecture_id", type=str, default=None, help="Optional lecture identifier")
    parser.add_argument("--lecture_title", type=str, default="", help="Optional lecture title")
    args = parser.parse_args()

    process_file(args.input_file, args.output_file, args.lecture_id, args.lecture_title)