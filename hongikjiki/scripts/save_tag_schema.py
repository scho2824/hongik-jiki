# scripts/save_tag_schema.py
# scripts/save_tag_schema.py

from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[2]

from hongikjiki.modules.tagging.tag_schema import TagSchema

if __name__ == "__main__":
    schema = TagSchema()
    output_path = ROOT_DIR / "data" / "config" / "tag_schema.yaml"
    schema.save_schema(str(output_path))
    print("✅ 태그 스키마가 성공적으로 저장되었습니다.")