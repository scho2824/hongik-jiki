from hongikjiki.modules.tagging.tag_extractor import TagExtractor
from hongikjiki.modules.tagging.tag_schema import TagSchema

import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from tqdm import tqdm  # 진행 상황 표시용
import random
import os
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[3]
import glob

# 로깅 설정
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 안전하게 태그 추출기 초기화
try:
    tag_schema = TagSchema.load_from_yaml(str(ROOT_DIR / "data/converted_tag_schema.yaml"))
    tag_extractor = TagExtractor(tag_schema)
except Exception as e:
    logger.warning(f"태그 추출기 초기화 실패: {e}")
    tag_schema = None
    tag_extractor = None

def generate_multiple_qa(text: str, tags: Dict[str, float], source: Optional[str] = None, 
                        original_tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    하나의 텍스트로부터 여러 QA 쌍을 생성합니다.
    
    Args:
        text: 소스 텍스트
        tags: 태그 사전 (태그명: 신뢰도)
        source: 소스 식별자 (선택 사항)
        original_tags: 기존 태그 목록 (선택 사항)
        
    Returns:
        여러 QA 쌍 리스트
    """
    qa_list = []
    if not text:
        return qa_list
        
    orig_tags = original_tags or []

    # 다양한 질문 템플릿에서 무작위로 2개 선택
    base_questions = [
        "이 문장에서 전하려는 핵심 개념은 무엇인가요?",
        "이 가르침의 중심 메시지는 무엇인가요?",
        "이 내용을 통해 무엇을 배울 수 있나요?",
        "이 내용을 일상에 적용한다면 어떤 변화가 있을까요?",
        "이 내용은 어떤 삶의 태도를 권유하고 있나요?",
        "이 내용은 당신의 가치관에 어떤 영향을 줄 수 있나요?",
    ]
    available_questions = min(len(base_questions), 2)
    selected_questions = random.sample(base_questions, k=available_questions) if available_questions > 0 else []
    
    for q in selected_questions:
        qa_item_tags = list(tags.keys())
        combined_tags = list(set(qa_item_tags + orig_tags))
        description = "이 개념"
        
        # 텍스트에서 첫 2문장 추출 (안전하게)
        sentences = text.strip().split(".")
        quoted_insight = " ".join(sentences[:min(2, len(sentences))]).strip() + "."
        
        qa_list.append({
            "question": q,
            "cleaned_question": q,
            "quoted_insight": quoted_insight,
            "insight_explanation": f"이 내용은 '{description}'에 대한 통찰을 제공합니다.",
            "answer": text.strip(),
            "tags": combined_tags,
            "source_text": text.strip(),
            "source": source
        })

    # 질문 3~4: 태그 기반 질문 (상위 태그 1~2개)
    tag_insight_templates = [
        "이 내용은 '{tag}'와 관련하여 어떤 통찰을 줍니까?",
        "'{tag}'라는 관점에서 이 내용을 어떻게 해석할 수 있나요?",
        "이 내용은 '{tag}' 개념을 어떻게 설명하고 있나요?"
    ]

    # 태그 설명 가져오기 (안전하게)
    tag_descriptions = {}
    if tag_schema:
        for tag in tags.keys():
            tag_obj = tag_schema.tags.get(tag)
            if tag_obj and hasattr(tag_obj, 'description'):
                tag_descriptions[tag] = tag_obj.description
            else:
                tag_descriptions[tag] = "이 개념"
    else:
        tag_descriptions = {tag: "이 개념" for tag in tags.keys()}

    # 상위 태그 선택 (안전하게)
    top_tags = sorted(tags.items(), key=lambda x: x[1], reverse=True)
    if len(top_tags) > 2:
        top_tags = top_tags[:2]
        
    for tag, _ in top_tags:
        description = tag_descriptions.get(tag, "이 개념")
        if tag_insight_templates:
            question_template = random.choice(tag_insight_templates)
            qa_item_tags = [tag]
            combined_tags = list(set(qa_item_tags + orig_tags))
            
            qa_list.append({
                "question": question_template.format(tag=description),
                "cleaned_question": question_template.format(tag=description),
                "quoted_insight": quoted_insight,
                "insight_explanation": f"이 문장은 '{description}' 개념과 관련이 있습니다.",
                "answer": text.strip(),
                "tags": combined_tags,
                "source_text": text.strip(),
                "source": source
            })

    # 중복 제거 (보장된 태그 유일성)
    for qa_item in qa_list:
        qa_item["tags"] = list(set(qa_item["tags"]))
    return qa_list

def load_dataset(input_dir: Path) -> List[Dict[str, Any]]:
    """
    주어진 디렉토리에서 JSON 및 JSONL 파일을 로드합니다.
    
    Args:
        input_dir: 입력 디렉토리 경로
        
    Returns:
        로드된 데이터 리스트
    """
    dataset = []
    
    input_path = input_dir
    if not input_path.exists():
        logger.warning(f"입력 디렉토리가 존재하지 않습니다: {input_dir}")
        return dataset
        
    try:
        all_files = list(input_path.rglob("*.json")) + list(input_path.rglob("*.jsonl"))

        for file_path in all_files:
            try:
                with file_path.open("r", encoding="utf-8") as f:
                    if str(file_path).endswith(".jsonl"):
                        for line in f:
                            if line.strip():
                                dataset.append(json.loads(line))
                    else:
                        data = json.load(f)
                        if isinstance(data, list):
                            dataset.extend(data)
                        else:
                            dataset.append(data)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
    except Exception as e:
        logger.error(f"Error scanning directory {input_dir}: {e}")
        
    return dataset

def advanced_qa_generator(dataset: List[Dict[str, Any]], min_length: int = 30) -> List[Dict[str, Any]]:
    """
    텍스트 데이터셋으로부터 QA 쌍을 생성합니다.
    
    Args:
        dataset: 청크 데이터 리스트
        min_length: 처리할 최소 텍스트 길이
        
    Returns:
        생성된 QA 쌍 리스트
    """
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
        item_text = item.get("content") or item.get("page_content", "")
        if not item_text:
            # 안전하게 텍스트 필드 확인
            item_text = item.get("text", "")
        
        # 길이 확인
        if len(item_text.strip()) < min_length:
            too_short += 1
            continue
        
        # 태그 추출
        tags = {}
        near_tags = []
        try:
            if tag_extractor:
                result = tag_extractor.extract_tags(item_text, return_near=True)
                if isinstance(result, tuple) and len(result) == 2:
                    main_tags, near_tags = result
                    tags = main_tags if main_tags else dict(near_tags[:2] if near_tags else {})
                elif isinstance(result, dict):
                    tags = result
        except Exception as e:
            logger.warning(f"Tag extraction failed for chunk (excerpt: {item_text[:50]}...): {e}")
        
        source = item.get("source")
        original_tags = item.get("tags", [])
        
        # 다양한 질문 생성: 핵심 개념, 실천, 태그 기반 등
        multiple_qa = generate_multiple_qa(item_text, tags, source=source, original_tags=original_tags)
        
        for qa in multiple_qa:
            # 기존 문서 메타데이터가 있다면 보존
            qa["metadata"] = item.get("metadata", {})
        
        qa_pairs.extend(multiple_qa)
        processed += 1
    
    # 통계 로깅
    logger.info(f"처리 결과: 총 {total_chunks}개 중 {processed}개 처리됨")
    logger.info(f"너무 짧아서 건너뛴 청크: {too_short}개")
    logger.info(f"생성된 QA 쌍: {len(qa_pairs)}개")
    
    return qa_pairs

def save_qa_dataset(output_path: Path, qa_pairs: List[Dict[str, Any]], append: bool = False) -> None:
    """
    QA 쌍을 JSON 파일로 저장합니다.
    
    Args:
        output_path: 출력 파일 경로
        qa_pairs: 저장할 QA 쌍 리스트
        append: 기존 파일에 추가할지 여부
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if append and output_path.exists():
            with output_path.open("r", encoding="utf-8") as f:
                try:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        existing_data = []
                except Exception:
                    existing_data = []
            combined_data = existing_data + qa_pairs
        else:
            combined_data = qa_pairs
            
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"QA 데이터 저장 실패: {e}")

class QAGenerator:
    """
    QA 생성기 래퍼 클래스
    """
    def __init__(self, min_length: int = 30):
        """
        QA 생성기를 초기화합니다.
        
        Args:
            min_length: 처리할 최소 텍스트 길이
        """
        self.min_length = min_length

    def generate(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        청크 데이터로부터 QA 쌍을 생성합니다.
        
        Args:
            dataset: 청크 데이터 리스트
            
        Returns:
            QA 쌍 리스트
        """
        return advanced_qa_generator(dataset, self.min_length)

    def generate_from_file(self, input_file: Path) -> List[Dict[str, Any]]:
        """
        파일로부터 데이터를 로드하고 QA 쌍을 생성합니다.
        
        Args:
            input_file: 입력 파일 경로
            
        Returns:
            QA 쌍 리스트
        """
        if input_file.is_dir():
            data = load_dataset(input_file)
        else:
            data = load_dataset(input_file.parent)
        return self.generate(data)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--min_length", type=int, default=30, 
                       help="최소 텍스트 길이 (이보다 짧은 청크는 처리하지 않음)")
    parser.add_argument("--append", action="store_true", help="기존 출력 파일에 QA 데이터를 추가합니다.")
    args = parser.parse_args()

    input_dir: Path = Path(args.input_dir)
    output_file: Path = Path(args.output_file)

    logger.info(f"📥 입력 디렉토리: {input_dir}")
    logger.info(f"📤 출력 파일: {output_file}")

    dataset = load_dataset(input_dir)
    qa_pairs = advanced_qa_generator(dataset, args.min_length)
    save_qa_dataset(output_file, qa_pairs, append=args.append)

    logger.info(f"✅ 총 {len(qa_pairs)}개의 QA 쌍이 생성되어 저장되었습니다.")