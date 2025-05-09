from hongikjiki.tagging.tag_extractor import TagExtractor
from hongikjiki.tagging.tag_schema import TagSchema

import json
import logging
from typing import List, Dict
from tqdm import tqdm  # 진행 상황 표시용
import random

# 로깅 설정
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize tag extractor once for reuse
tag_schema = TagSchema("data/config/tag_schema.yaml")
tag_extractor = TagExtractor(tag_schema, "data/config/tag_patterns.json")

def generate_multiple_qa(text: str, tags: Dict[str, float]) -> List[Dict[str, str]]:
    qa_list = []

    # 다양한 질문 템플릿에서 무작위로 2개 선택
    base_questions = [
        "이 문장에서 전하려는 핵심 개념은 무엇인가요?",
        "이 가르침의 중심 메시지는 무엇인가요?",
        "이 내용을 통해 무엇을 배울 수 있나요?",
        "이 내용을 일상에 적용한다면 어떤 변화가 있을까요?",
        "이 내용은 어떤 삶의 태도를 권유하고 있나요?",
        "이 내용은 당신의 가치관에 어떤 영향을 줄 수 있나요?",
    ]
    selected_questions = random.sample(base_questions, k=2)
    for q in selected_questions:
        qa_list.append({
            "question": q,
            "cleaned_question": q,
            "quoted_insight": text.strip().split(".")[0] + ".",
            "insight_explanation": "",
            "answer": text.strip(),
            "tags": list(tags.keys()),
            "source_text": text.strip()
        })

    # 질문 3~4: 태그 기반 질문 (상위 태그 1~2개)
    tag_insight_templates = [
        "이 내용은 '{tag}'와 관련하여 어떤 통찰을 줍니까?",
        "'{tag}'라는 관점에서 이 내용을 어떻게 해석할 수 있나요?",
        "이 내용은 '{tag}' 개념을 어떻게 설명하고 있나요?"
    ]

    # Safely obtain the schema dictionary regardless of attribute name
    internal_schema = getattr(tag_schema, "schema", getattr(tag_schema, "_schema", {}))
    tag_descriptions = {
        tag: internal_schema.get(tag, {}).get("description", tag)
        for tag in tags.keys()
    }

    top_tags = sorted(tags.items(), key=lambda x: x[1], reverse=True)[:2]
    for tag, _ in top_tags:
        description = tag_descriptions.get(tag, tag)
        question_template = random.choice(tag_insight_templates)
        qa_list.append({
            "question": question_template.format(tag=description),
            "cleaned_question": question_template.format(tag=description),
            "quoted_insight": text.strip().split(".")[0] + ".",
            "insight_explanation": "",
            "answer": text.strip(),
            "tags": [tag],
            "source_text": text.strip()
        })

    # 중복 제거 (보장된 태그 유일성)
    for qa_item in qa_list:
        qa_item["tags"] = list(set(qa_item["tags"]))
    return qa_list

def load_dataset(input_path: str) -> List[Dict[str, str]]:
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)

def advanced_qa_generator(dataset: List[Dict[str, str]], min_length: int = 30) -> List[Dict[str, str]]:
    qa_pairs = []
    
    # 질문 템플릿
    question_templates = [
        "이 문장은 무엇을 말하고 있나요?",
        "이 내용에서 가장 중요한 개념은 무엇인가요?",
        "이 가르침의 핵심 메시지는 무엇인가요?",
        "이 내용을 일상에 어떻게 적용할 수 있을까요?",
        "이 내용에서 강조하는 가치는 무엇인가요?"
    ]
    
    # 청크 처리 통계
    total_chunks = len(dataset)
    too_short = 0
    processed = 0
    
    logger.info(f"총 {total_chunks}개 청크 처리 시작")
    
    for item in tqdm(dataset):
        # Unify text field from content or page_content
        item["text"] = item.get("content") or item.get("page_content", "")
        text = item["text"]
        
        # 길이 확인
        if len(text.strip()) < min_length:
            too_short += 1
            continue
        
        # 태그 추출
        try:
            main_tags, near_tags = tag_extractor.extract_tags(text, return_near=True)
            tags = main_tags if main_tags else dict(near_tags[:2])  # Use near tags if main_tags is empty
        except Exception as e:
            logger.warning(f"Tag extraction failed for chunk: {e}")
            tags = {}
        
        # 다양한 질문 생성: 핵심 개념, 실천, 태그 기반 등
        multiple_qa = generate_multiple_qa(text, tags)
        qa_pairs.extend(multiple_qa)
        
        processed += 1
    
    # 통계 로깅
    logger.info(f"처리 결과: 총 {total_chunks}개 중 {processed}개 처리됨")
    logger.info(f"너무 짧아서 건너뛴 청크: {too_short}개")
    logger.info(f"생성된 QA 쌍: {len(qa_pairs)}개")
    
    return qa_pairs

def save_qa_dataset(output_path: str, qa_pairs: List[Dict[str, str]]):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--min_length", type=int, default=30, 
                       help="최소 텍스트 길이 (이보다 짧은 청크는 처리하지 않음)")
    args = parser.parse_args()

    logger.info(f"📥 입력 파일: {args.input_file}")
    logger.info(f"📤 출력 파일: {args.output_file}")

    dataset = load_dataset(args.input_file)
    qa_pairs = advanced_qa_generator(dataset, args.min_length)
    save_qa_dataset(args.output_file, qa_pairs)

    logger.info(f"✅ 총 {len(qa_pairs)}개의 QA 쌍이 생성되어 저장되었습니다.")