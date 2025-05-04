# scripts/save_tag_schema.py

from hongikjiki.tagging.tag_schema import TagSchema

if __name__ == "__main__":
    schema = TagSchema()
    schema.save_schema("data/config/tag_schema.yaml")
    print("✅ 태그 스키마가 성공적으로 저장되었습니다.")