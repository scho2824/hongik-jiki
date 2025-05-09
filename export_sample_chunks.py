import json
import random

with open("data/processed/jungbub_dataset.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

sampled = random.sample(chunks, 10)

for i, chunk in enumerate(sampled):
    print(f"\n🔹 청크 {i+1}")
    print(f"📄 길이: {len(chunk['content'])}자")
    print(f"📌 source_id: {chunk['metadata'].get('source_id')}")
    print(f"🏷️ tags: {chunk.get('tags', {})}")
    print(f"📝 preview: {chunk['content'][:120].replace(chr(10), ' ')}...")