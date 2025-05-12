# tests/unit/test_formatter.py
import pytest
from hongikjiki.core.formatter import format_response

def test_format_response_basic():
    """기본 포맷팅 테스트"""
    # 기본 테스트 케이스
    results = [
        {
            'content': 'Q: 테스트 질문? A: 테스트 답변입니다.',
            'metadata': {'tags': ['태그1', '태그2'], 'lecture_id': '001', 'lecture_title': '테스트 강의'}
        }
    ]
    answer = "테스트 답변"
    
    # 함수 실행
    formatted = format_response(results, answer)
    
    # 검증
    assert "테스트 답변" in formatted
    assert "출처" in formatted
    assert "테스트 강의" in formatted
    assert "강의 번호: 001" in formatted
    assert "#태그1" in formatted
    assert "#태그2" in formatted

def test_format_response_empty_results():
    """빈 검색 결과 테스트"""
    results = []
    answer = "테스트 답변"
    
    formatted = format_response(results, answer)
    
    # 기본 답변은 유지되어야 함
    assert formatted.strip().startswith("테스트 답변")
    # 출처 정보가 없어야 함
    assert "출처" not in formatted

def test_format_response_with_extracted_tags():
    """추출된 태그가 있는 경우 테스트"""
    results = [{'content': 'test', 'metadata': {}}]
    answer = "테스트 답변"
    extracted_tags = ["추출태그1", "추출태그2"]
    
    formatted = format_response(results, answer, extracted_tags)
    
    assert "#추출태그1" in formatted
    assert "#추출태그2" in formatted