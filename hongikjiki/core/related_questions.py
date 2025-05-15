# hongikjiki/core/related_questions.py
import os
import json
import random
import logging

logger = logging.getLogger("HongikJikiChatBot")

def load_tag_questions(json_path="data/tag_question_map.json"):
    """외부 JSON에서 태그별 질문 로드"""
    if not os.path.exists(json_path):
        logger.warning(f"Tag question map not found: {json_path}")
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_related_questions(tags, current_question):
    """태그 기반 관련 질문 생성"""
    # 태그별 질문 맵핑
    tag_questions = load_tag_questions()
    
    related = []
    # 태그 기반 질문 추가
    for tag in tags:
        if tag in tag_questions:
            for question in tag_questions[tag]:
                if question.lower() != current_question.lower() and question not in related:
                    related.append({
                        "question": question,
                        "insight": f"'{tag}' 관련 질문입니다."
                    })
        else:
            logger.debug(f"태그 '{tag}'에 해당하는 질문이 없습니다.")
    
    # 일반 질문 추가 (태그가 없거나 적을 경우)
    general_questions = [
        {"question": "정법이란 무엇인가요?", "insight": "정법의 기본 개념에 관한 질문입니다."},
        {"question": "자주독립의 의미는 무엇인가요?", "insight": "정신적 자립에 관한 질문입니다."},
        {"question": "어떻게 마음의 평화를 찾을 수 있나요?", "insight": "마음 수행에 관한 질문입니다."},
        {"question": "홍익인간이란 무엇인가요?", "insight": "홍익인간 철학에 관한 질문입니다."}
    ]
    
    random.shuffle(related)
    
    # 관련 질문이 부족하면 일반 질문 추가
    if len(related) < 3:
        for q in general_questions:
            if q["question"].lower() != current_question.lower() and q not in related:
                related.append(q)
                if len(related) >= 3:
                    break
    
    # 최대 3개 반환, 중복 제거
    seen = set()
    final = []
    for q in related:
        if q["question"].lower() not in seen:
            final.append(q)
            seen.add(q["question"].lower())
    return final[:3]