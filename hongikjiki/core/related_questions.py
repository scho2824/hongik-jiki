# hongikjiki/core/related_questions.py
import logging

logger = logging.getLogger("HongikJikiChatBot")

def generate_related_questions(tags, current_question):
    """태그 기반 관련 질문 생성"""
    # 태그별 질문 맵핑
    tag_questions = {
        "감정": [
            "감정이 불안정한 이유는 무엇인가요?",
            "화가 날 때 어떻게 다스려야 할까요?",
            "자신의 감정을 이해하는 방법은 무엇인가요?"
        ],
        "관계": [
            "가족과의 갈등을 어떻게 해결할 수 있나요?",
            "인간관계에서 중요한 것은 무엇인가요?",
            "타인을 이해하는 방법은 무엇인가요?"
        ],
        # 기존 코드에서 가져온 다른 태그들...
    }
    
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
    
    # 일반 질문 추가 (태그가 없거나 적을 경우)
    general_questions = [
        {"question": "정법이란 무엇인가요?", "insight": "정법의 기본 개념에 관한 질문입니다."},
        {"question": "자주독립의 의미는 무엇인가요?", "insight": "정신적 자립에 관한 질문입니다."},
        {"question": "어떻게 마음의 평화를 찾을 수 있나요?", "insight": "마음 수행에 관한 질문입니다."},
        {"question": "홍익인간이란 무엇인가요?", "insight": "홍익인간 철학에 관한 질문입니다."}
    ]
    
    # 관련 질문이 부족하면 일반 질문 추가
    if len(related) < 3:
        for q in general_questions:
            if q["question"].lower() != current_question.lower() and q not in related:
                related.append(q)
                if len(related) >= 3:
                    break
    
    # 최대 3개 반환
    return related[:3]