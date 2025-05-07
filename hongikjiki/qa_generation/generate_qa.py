from hongikjiki.tagging.tag_extractor import TagExtractor
from hongikjiki.tagging.tag_schema import TagSchema

import json
import logging
from typing import List, Dict
from tqdm import tqdm  # 진행 상황 표시용

# 로깅 설정
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize tag extractor once for reuse
tag_schema = TagSchema("data/config/tag_schema.yaml")
tag_extractor = TagExtractor(tag_schema, "data/config/tag_patterns.json")

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
            tags = list(tag_extractor.extract_tags(text).keys())
        except Exception as e:
            logger.warning(f"Tag extraction failed for chunk: {e}")
            tags = []
        
        # 단락 내용에 따라 적절한 질문 선택
        # 간단한 구현: 태그와 텍스트 길이에 따라 질문 선택
        if len(tags) > 0 and '깨달음' in tags:
            question = question_templates[2]  # 핵심 메시지
        elif len(tags) > 0 and '실천' in tags:
            question = question_templates[3]  # 일상 적용
        elif len(text) > 500:
            question = question_templates[1]  # 중요 개념
        elif len(tags) > 0 and '가치' in tags:
            question = question_templates[4]  # 강조하는 가치
        else:
            question = question_templates[0]  # 기본 질문
        
        answer = text.strip()
        
        qa_pairs.append({
            "question": question, 
            "answer": answer, 
            "tags": tags
        })
        
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