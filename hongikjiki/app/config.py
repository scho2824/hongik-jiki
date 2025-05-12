# hongikjiki/app/config.py
import os
from dotenv import load_dotenv
from packaging import version as pkg_version
import gradio as gr

# 환경 변수 로드
load_dotenv()

# API 키
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# 경로 설정
PERSIST_DIR = "data/vector_store"
COLLECTION_NAME = "hongikjiki_jungbub"
TAG_SCHEMA_PATH = "data/config/tag_schema.yaml"
TAG_PATTERN_PATH = "data/config/tag_patterns.json"
QA_FILE_PATH = "data/qa/high_insight_qa_dataset_formatted_related.json"

# Gradio 버전 체크
GRADIO_VERSION = pkg_version.parse(gr.__version__)
USE_MESSAGE_FORMAT = GRADIO_VERSION >= pkg_version.parse("3.32.0")