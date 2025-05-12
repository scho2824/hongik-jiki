import pytest
from unittest.mock import MagicMock

@pytest.fixture
def mock_llm():
    """LLM 모의 객체 생성"""
    mock = MagicMock()
    mock.generate.return_value = "모의 응답입니다."
    return mock

@pytest.fixture
def mock_vector_store():
    """벡터 스토어 모의 객체 생성"""
    mock = MagicMock()
    mock.search.return_value = [
        {
            'content': 'Q: 테스트 질문? A: 테스트 답변입니다.',
            'metadata': {'tags': ['테스트태그'], 'lecture_id': '001', 'lecture_title': '테스트 강의'},
            'score': 0.95
        }
    ]
    mock.count.return_value = 10
    return mock

@pytest.fixture
def mock_tag_extractor():
    """태그 추출기 모의 객체"""
    mock = MagicMock()
    mock.extract_tags_from_query.return_value = ["테스트태그1", "테스트태그2"]
    return mock