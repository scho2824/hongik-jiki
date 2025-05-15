from hongikjiki.modules.text_processing.document_loader import DocumentLoader
import pytest
from hongikjiki.modules.text_processing.document_processor import DocumentProcessor
from hongikjiki.modules.text_processing.document_chunker import DocumentChunker
from hongikjiki.modules.text_processing.text_normalizer import TextNormalizer
from hongikjiki.modules.text_processing.metadata_extractor import MetadataExtractor

# 더미 로더
class DummyLoader(DocumentLoader):
    def load_document(self, path):
        return {
            "content": "이것은 테스트 문서입니다. 반복되지 않도록 구성되어 있습니다.",
            "metadata": {"source": path}
        }

def test_process_file_creates_chunks():
    processor = DocumentProcessor(
        document_loader=DummyLoader(),
        text_normalizer=TextNormalizer(),
        metadata_extractor=MetadataExtractor(),
        document_chunker=DocumentChunker(chunk_size=20, overlap=5)
    )

    chunks = processor.process_file("dummy.txt")
    assert len(chunks) >= 1
    for chunk in chunks:
        assert "content" in chunk
        assert "metadata" in chunk
        assert "filename" in chunk["metadata"]

def test_chunk_content_integrity():
    processor = DocumentProcessor(
        document_loader=DummyLoader(),
        text_normalizer=TextNormalizer(),
        metadata_extractor=MetadataExtractor(),
        document_chunker=DocumentChunker(chunk_size=20, overlap=5)
    )

    chunks = processor.process_file("dummy.txt")
    all_text = " ".join(chunk["content"] for chunk in chunks)
    assert "테스트 문서" in all_text
    assert "반복되지 않도록" in all_text