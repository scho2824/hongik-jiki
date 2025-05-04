"""
텍스트 처리 모듈 테스트
"""
import unittest
import os
import tempfile
from pathlib import Path
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 테스트할 모듈 임포트
from hongikjiki.text_processing.document_loader import DocumentLoader
from hongikjiki.text_processing.text_normalizer import TextNormalizer
from hongikjiki.text_processing.document_chunker import DocumentChunker
from hongikjiki.text_processing.metadata_extractor import MetadataExtractor
from hongikjiki.text_processing.document_processor import DocumentProcessor

class DocumentLoaderTest(unittest.TestCase):
    """문서 로더 테스트 클래스"""
    
    def setUp(self):
        """테스트 환경 설정"""
        self.loader = DocumentLoader()
        # 임시 테스트 파일 생성
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = self.temp_dir.name
        
        # 테스트용 텍스트 파일 생성
        self.text_file = os.path.join(self.test_dir, "test_document.txt")
        with open(self.text_file, "w", encoding="utf-8") as f:
            f.write("이것은 테스트 문서입니다.\n정법 문서 테스트를 위한 내용입니다.")
    
    def tearDown(self):
        """테스트 환경 정리"""
        self.temp_dir.cleanup()
    
    def test_load_text_file(self):
        """텍스트 파일 로드 테스트"""
        doc_data = self.loader.load_document(self.text_file)
        
        self.assertIsNotNone(doc_data)
        self.assertIn("content", doc_data)
        self.assertIn("metadata", doc_data)
        self.assertEqual(doc_data["metadata"]["filename"], "test_document.txt")
        self.assertTrue("이것은 테스트 문서입니다." in doc_data["content"])
    
    def test_unsupported_format(self):
        """지원하지 않는 형식 테스트"""
        # 지원하지 않는 확장자 파일 생성
        unknown_file = os.path.join(self.test_dir, "unknown.xyz")
        with open(unknown_file, "w") as f:
            f.write("내용")
        
        result = self.loader.load_document(unknown_file)
        self.assertIsNone(result)


class TextNormalizerTest(unittest.TestCase):
    """텍스트 정규화 테스트 클래스"""
    
    def setUp(self):
        """테스트 환경 설정"""
        self.normalizer = TextNormalizer()
    
    def test_normalize_text(self):
        """텍스트 정규화 테스트"""
        # 중복 공백, 특수 문자 등이 포함된 텍스트
        text = "이것은   정규화 테스트  입니다.  \n\n  특수문자: !@#$%^&*() 처리 테스트"
        
        normalized = self.normalizer.normalize(text)
        
        self.assertIsNotNone(normalized)
        self.assertNotEqual(normalized, text)  # 정규화로 인해 변경되어야 함
        self.assertFalse("  " in normalized)  # 중복 공백 제거 확인


class DocumentChunkerTest(unittest.TestCase):
    """문서 청킹 테스트 클래스"""
    
    def setUp(self):
        """테스트 환경 설정"""
        self.chunker = DocumentChunker(chunk_size=100, overlap=20)
    
    def test_split_documents(self):
        """문서 분할 테스트"""
        # 긴 문서 준비
        long_text = "안녕하세요. " * 50  # 충분히 긴 텍스트
        document = {
            "content": long_text,
            "metadata": {"filename": "test.txt", "source": "테스트"}
        }
        
        chunks = self.chunker.split_documents([document])
        
        self.assertGreater(len(chunks), 1)  # 여러 청크로 나뉘어야 함
        self.assertLessEqual(len(chunks[0]["content"]), 100)  # 청크 크기 제한 확인
        
        # 메타데이터 보존 확인
        self.assertEqual(chunks[0]["metadata"]["filename"], "test.txt")
        self.assertEqual(chunks[0]["metadata"]["source"], "테스트")
        
        # 청크 번호와 총 청크 수 메타데이터 확인
        self.assertIn("chunk", chunks[0]["metadata"])
        self.assertIn("total_chunks", chunks[0]["metadata"])


class MetadataExtractorTest(unittest.TestCase):
    """메타데이터 추출 테스트 클래스"""
    
    def setUp(self):
        """테스트 환경 설정"""
        self.extractor = MetadataExtractor()
    
    def test_extract_metadata(self):
        """메타데이터 추출 테스트"""
        # 테스트용 텍스트 준비
        text = "제목: 테스트 문서\n저자: 홍길동\n주제: 정법 테스트\n\n본문 내용입니다..."
        filename = "test_doc.txt"
        base_metadata = {"format": ".txt", "file_path": "/tmp/test_doc.txt"}
        
        metadata = self.extractor.extract_metadata(text, filename, base_metadata)
        
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata["filename"], filename)
        self.assertEqual(metadata["format"], ".txt")
        
        # 파일명에서 제목 추출 확인
        self.assertIn("title", metadata)


class DocumentProcessorTest(unittest.TestCase):
    """문서 처리기 통합 테스트 클래스"""
    
    def setUp(self):
        """테스트 환경 설정"""
        # 의존성 인스턴스 생성
        self.loader = DocumentLoader()
        self.normalizer = TextNormalizer()
        self.extractor = MetadataExtractor()
        self.chunker = DocumentChunker(chunk_size=200, overlap=50)
        
        # 처리기 인스턴스 생성
        self.processor = DocumentProcessor(
            document_loader=self.loader,
            text_normalizer=self.normalizer,
            metadata_extractor=self.extractor,
            document_chunker=self.chunker,
            chunk_size=200,
            overlap=50
        )
        
        # 임시 테스트 디렉토리 및 파일 생성
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = self.temp_dir.name
        
        # 테스트용 텍스트 파일 생성
        self.text_file = os.path.join(self.test_dir, "test_document.txt")
        with open(self.text_file, "w", encoding="utf-8") as f:
            f.write("이것은 테스트 문서입니다.\n" * 30)  # 충분히 긴 텍스트
        
        # 추가 테스트 파일
        self.text_file2 = os.path.join(self.test_dir, "another_document.txt")
        with open(self.text_file2, "w", encoding="utf-8") as f:
            f.write("다른 테스트 문서입니다.\n" * 20)
    
    def tearDown(self):
        """테스트 환경 정리"""
        self.temp_dir.cleanup()
    
    def test_process_file(self):
        """단일 파일 처리 테스트"""
        chunks = self.processor.process_file(self.text_file)
        
        self.assertIsNotNone(chunks)
        self.assertGreater(len(chunks), 0)
        
        # 첫 번째 청크 내용 확인
        self.assertIn("이것은 테스트 문서입니다", chunks[0]["content"])
        
        # 메타데이터 확인
        self.assertEqual(chunks[0]["metadata"]["filename"], "test_document.txt")
        self.assertIn("file_hash", chunks[0]["metadata"])
    
    def test_process_directory(self):
        """디렉토리 처리 테스트"""
        chunks = self.processor.process_directory(self.test_dir)
        
        self.assertIsNotNone(chunks)
        self.assertGreater(len(chunks), 0)
        
        # 여러 파일의 청크가 포함되어 있는지 확인
        file_names = set()
        for chunk in chunks:
            file_names.add(chunk["metadata"]["filename"])
        
        self.assertGreaterEqual(len(file_names), 2)  # 최소 2개 파일 처리 확인
    
    def test_duplicate_detection(self):
        """중복 감지 테스트"""
        # 동일한 내용의 파일 복사
        duplicate_file = os.path.join(self.test_dir, "duplicate.txt")
        with open(duplicate_file, "w", encoding="utf-8") as f:
            with open(self.text_file, "r", encoding="utf-8") as source:
                f.write(source.read())
        
        # 첫 번째 파일 처리
        self.processor.process_file(self.text_file)
        
        # 중복 내용 확인
        content = "이것은 테스트 문서입니다.\n" * 30
        self.assertTrue(self.processor.is_duplicate(content))
        
        # 다른 내용 확인
        self.assertFalse(self.processor.is_duplicate("완전히 다른 내용"))


if __name__ == "__main__":
    unittest.main()