#!/bin/bash
# 홍익지기 챗봇 데모 실행 스크립트

# 스크립트 종료 시 하위 프로세스도 종료
trap 'kill $(jobs -p) 2>/dev/null' EXIT

# 환경 변수 확인
if [ -f .env ]; then
  echo "환경 변수 파일 .env를 로드합니다."
  export $(grep -v '^#' .env | xargs)
else
  echo "경고: .env 파일이 없습니다."
fi

# API 키 확인
if [ -z "$OPENAI_API_KEY" ]; then
  echo "오류: OPENAI_API_KEY가 설정되지 않았습니다."
  echo "다음 방법 중 하나로 API 키를 설정하세요:"
  echo "1. export OPENAI_API_KEY=your_api_key"
  echo "2. .env 파일에 OPENAI_API_KEY=your_api_key 추가"
  exit 1
fi

echo "API 키가 설정되어 있습니다."
echo "Gradio 앱을 시작합니다..."

# 데모 앱 실행
python gradio_app_demo.py

echo "앱이 종료되었습니다."