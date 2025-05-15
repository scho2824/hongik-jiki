

import pytest
from hongikjiki.modules.vector_store.chroma_store import ChromaVectorStore

def test_chroma_add_and_search(tmp_path):
    # 1. 벡터 저장소 초기화
    store = ChromaVectorStore(persist_directory=str(tmp_path))

    # 2. 텍스트 추가
    texts = ["정법은 우주의 이치를 따르는 삶의 원리입니다."]
    metadatas = [{"source": "test_doc", "source_id": "doc1"}]
    added_ids = store.add_texts(texts, metadatas)

    # 3. 저장 확인
    assert len(added_ids) == 1
    assert store.count() >= 1

    # 4. 검색 실행
    results = store.search("우주의 원리", k=1)
    assert isinstance(results, list)
    assert len(results) >= 1
    assert "정법" in results[0]["content"]