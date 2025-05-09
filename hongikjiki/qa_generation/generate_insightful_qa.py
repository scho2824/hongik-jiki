import json
import os
import re
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

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
        content = response.choices[0].message.content
        # 코드 펜스 제거 및 JSON 본문 추출
        clean_content = re.sub(r"^```(?:json)?\s*", "", content)
        clean_content = re.sub(r"\s*```$", "", clean_content).strip()
        # JSON 본문만 추출
        json_start = clean_content.find("{")
        json_end = clean_content.rfind("}")
        if json_start != -1 and json_end != -1:
            json_str = clean_content[json_start:json_end+1]
        else:
            json_str = clean_content
        try:
            data = json.loads(json_str)
            qa.update(data)
        except json.JSONDecodeError as e:
            print("❌ JSON 파싱 오류:", e)
            print("⛔ 파싱 실패 응답 내용:", content)
            qa.update({
                "insight_summary": "",
                "keywords": [],
                "context_note": "JSON 파싱 실패"
            })
    except Exception as e:
        print("❌ 기타 오류 발생:", e)
        qa.update({
            "insight_summary": "",
            "keywords": [],
            "context_note": "자동 생성 실패"
        })
    return qa

def process_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as infile:
        qa_data = json.load(infile)

    enhanced = [enhance_qa(qa) for qa in tqdm(qa_data)]

    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(enhanced, outfile, ensure_ascii=False, indent=2)

    print(f"✅ 고밀도 QA 저장 완료 → {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    process_file(args.input_file, args.output_file)