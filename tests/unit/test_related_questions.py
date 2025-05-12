import pytest
from hongikjiki.core.related_questions import generate_related_questions

def test_generate_related_questions_with_tags():
    """태그 기반 관련 질문 생성 테스트"""
    tags = ["감정", "관계"]
    current_question = "테스트 질문"
    
    related = generate_related_questions(tags, current_question)
    
    # 검증
    assert len(related) == 3  # 항상 3개 반환
    assert all(isinstance(q, dict) for q in related)
    assert all("question" in q for q in related)
    assert all("insight" in q for q in related)
    
    # 내용 검증
    questions = [q["question"] for q in related]
    assert any("감정" in q["insight"] for q in related)
    assert current_question.lower() not in [q.lower() for q in questions]

def test_generate_related_questions_no_tags():
    """태그가 없는 경우 일반 질문 반환 테스트"""
    tags = []
    current_question = "테스트 질문"
    
    related = generate_related_questions(tags, current_question)
    
    # 태그가 없어도 3개의 일반 질문을 반환해야 함
    assert len(related) == 3
    assert all(isinstance(q, dict) for q in related)
