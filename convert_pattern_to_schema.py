# convert_patterns_to_schema.py

import json
import yaml
from hongikjiki.modules.tagging.tag_schema import Tag, TagSchema

def convert_pattern_json_to_schema(pattern_json_path: str, output_yaml_path: str):
    with open(pattern_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 일부 패턴은 "tags": { ... } 아래에 있을 수 있음
    tag_data = data.get("tags", data)

    schema = TagSchema()
    schema.tags = {}

    for tag_name, entry in tag_data.items():
        if isinstance(entry, dict):
            tag = Tag(
                name=tag_name,
                category="임시 분류",  # 필요시 수정
                keywords=entry.get("keywords", []),
                phrases=entry.get("phrases", []),
                patterns=entry.get("patterns", []),
                description=None
            )
        else:
            tag = Tag(
                name=tag_name,
                category="임시 분류",
                patterns=[str(entry)]
            )

        schema.add_tag(tag)

    schema.save_schema(output_yaml_path)
    print(f"✅ Converted pattern.json → {output_yaml_path}")

# 사용 예시
# convert_pattern_json_to_schema("data/pattern.json", "data/converted_tag_schema.yaml")

if __name__ == "__main__":
    convert_pattern_json_to_schema(
        "data/config/tag_patterns.json",
        "data/converted_tag_schema.yaml"
    )