# test_tagging.py
import pytest
from hongikjiki.modules.tagging.tag_schema import TagSchema
from hongikjiki.modules.tagging.tag_extractor import TagExtractor

@pytest.fixture
def tag_schema():
    schema_path = "data/config/tag_schema.yaml"
    return TagSchema(schema_path)

@pytest.fixture
def tag_extractor(tag_schema):
    return TagExtractor(tag_schema)

def test_tag_schema_loads(tag_schema):
    assert len(tag_schema.tags) > 0

def test_tag_extraction(tag_extractor):
    text = "기쁨과 평화가 가득한 하루입니다."
    tags = tag_extractor.extract_tags(text)
    assert isinstance(tags, list)
    assert len(tags) >= 1