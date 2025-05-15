# test_tagging_tools.py
from hongikjiki.modules.tagging.tag_schema import TagSchema
from hongikjiki.modules.tagging.tag_extractor import TagExtractor
from hongikjiki.modules.tagging.tagging_tools import TaggingSession, TaggingBatch

try:
    # 태그 스키마 초기화
    schema_path = "data/config/tag_schema.yaml"  # 실제 경로로 수정
    tag_schema = TagSchema(schema_path)
    tag_extractor = TagExtractor(tag_schema)
    
    # TaggingSession 테스트
    try:
        session = TaggingSession(tag_schema, tag_extractor)
        print("✅ TaggingSession 초기화 성공")
        
        # 간단한 문서 로드 테스트
        sample_doc = {
            "id": "test_doc",
            "content": "정법은 우주의 법칙을 따르는 홍익인간의 철학입니다.",
            "metadata": {"source": "테스트"}
        }
        
        result = session.load_document(sample_doc["id"], sample_doc["content"], sample_doc["metadata"])
        print(f"✅ 문서 로드 성공: {len(result.get('suggested_tags', {}))} 태그 제안됨")
    except Exception as e:
        print(f"❌ TaggingSession 테스트 실패: {e}")
    
    # TaggingBatch 테스트
    try:
        batch = TaggingBatch(tag_schema, tag_extractor)
        print("✅ TaggingBatch 초기화 성공")
    except Exception as e:
        print(f"❌ TaggingBatch 테스트 실패: {e}")
    
except Exception as e:
    print(f"❌ 전체 테스트 실패: {e}")