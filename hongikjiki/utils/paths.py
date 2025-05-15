# hongikjiki/__init__.py 또는 hongikjiki/utils/paths.py

from pathlib import Path

# 프로젝트 루트 디렉토리 (hongikjiki 패키지의 상위 디렉토리)
ROOT_DIR = Path(__file__).resolve().parents[1]

# 데이터 디렉토리 경로
DATA_DIR = ROOT_DIR / "data"
CONFIG_DIR = DATA_DIR / "config"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
TAG_DATA_DIR = DATA_DIR / "tag_data"
# 기타 필요한 경로 정의...