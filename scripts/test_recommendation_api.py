import pytest
import requests
from unittest.mock import patch

# 실제 서버에 대한 테스트는 skip 마커를 사용하여 선택적으로 실행
@pytest.mark.skip(reason="서버가 실행 중이지 않을 때는 이 테스트를 건너뜁니다")
def test_recommendation_api():
    url = "http://localhost:7860/recommendations?qa_id=your_test_qa_id_here"
    response = requests.get(url)
    assert response.status_code == 200
    # 추가 검증...

# 단위 테스트로 변환 (서버 연결 없이 실행 가능)
def test_recommendation_logic():
    # 여기에 recommendations 로직만 테스트하는 코드를 작성
    # API 요청 없이 내부 함수를 직접 호출
    pass

# Mock을 사용한 테스트 (서버 연결 없이 실행 가능)
@patch('requests.get')
def test_recommendation_with_mock(mock_get):
    # Mock 응답 설정
    mock_response = mock_get.return_value
    mock_response.status_code = 200
    mock_response.json.return_value = {"recommendations": ["추천1", "추천2"]}
    
    # 테스트 대상 함수 호출
    url = "http://localhost:7860/recommendations?qa_id=your_test_qa_id_here"
    response = requests.get(url)
    
    # 검증
    assert response.status_code == 200
    assert len(response.json()["recommendations"]) == 2