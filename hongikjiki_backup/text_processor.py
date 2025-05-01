import os
import re
import hashlib
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("HongikJikiChatBot")

class HongikJikiTextProcessor:
    """정법 텍스트 처리 클래스"""
    
    def __init__(self):
        # 정규 표현식 패턴 초기화
        self.lecture_pattern = re.compile(r'정법(\d+)강')
        self.title_patterns = [
            r'제목:\s*(.+)',
            r'강의명:\s*(.+)',
            r'\[정법강의\]\s*(.+)'
        ]
    
    def load_documents(self, directory: str) -> List[Dict[str, Any]]:
        """
        디렉토리에서 정법 문서를 로드하고 메타데이터 추출
        
        Args:
            directory: 문서가 저장된 디렉토리 경로
            
        Returns:
            List[Dict]: 각 문서의 내용과 메타데이터를 포함하는 딕셔너리 리스트
        """
        documents = []
        
        logger.info(f"{directory} 폴더에서 정법 문서 로드 중...")
        
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            file_ext = os.path.splitext(filename)[1].lower()
            
            try:
                content = None
                
                # 파일 형식에 따른 처리
                if file_ext in ['.txt', '.md']:
                    content = self._load_text_file(file_path)
                elif file_ext == '.rtf':
                    content = self._load_rtf_file(file_path)
                elif file_ext == '.pdf':
                    content = self._load_pdf_file(file_path)
                elif file_ext == '.docx':
                    content = self._load_docx_file(file_path)
                else:
                    logger.warning(f"지원하지 않는 파일 형식: {filename}")
                    continue
                    
                if content:
                    # 내용이 너무 짧으면 처리 오류 가능성 있음
                    if len(content.strip()) < 10:
                        logger.warning(f"파일 내용이 너무 짧아 처리 오류 의심: {filename}")
                    
                    # 타임스탬프 제거
                    content = self._remove_timestamps(content)
                    
                    # 메타데이터 추출
                    metadata = self._extract_metadata(content, filename)
                    
                    documents.append({
                        "content": content,
                        "metadata": metadata
                    })
                    
                    logger.debug(f"문서 로드 완료: {filename}, 메타데이터: {metadata}")
            
            except Exception as e:
                logger.error(f"파일 로드 실패: {filename}, 오류: {e}")
                logger.exception(e)  # 상세 오류 로깅
        
        logger.info(f"총 {len(documents)}개 문서 로드 완료")
        return documents
    
    def _load_text_file(self, file_path: str) -> str:
        """일반 텍스트 파일 로드"""
        encodings = ['utf-8', 'cp949', 'euc-kr']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        raise UnicodeDecodeError(f"지원하는 인코딩으로 파일을 열 수 없습니다: {file_path}")
    
    def _load_rtf_file(self, file_path: str) -> str:
        """RTF 파일 로드 및 처리"""
        try:
            # striprtf 라이브러리 사용 시도
            try:
                from striprtf.striprtf import rtf_to_text
                
                with open(file_path, 'rb') as f:
                    rtf_bytes = f.read()
                
                # RTF 텍스트를 추출하기 위한 여러 인코딩 시도
                encodings = ['utf-8', 'cp949', 'euc-kr', 'latin-1']
                
                for encoding in encodings:
                    try:
                        rtf_text = rtf_bytes.decode(encoding)
                        plain_text = rtf_to_text(rtf_text)
                        
                        # 성공적으로 변환된 경우 결과 반환
                        if plain_text and not plain_text.startswith('{\\rtf'):
                            # 불필요한 공백 제거 및 정리
                            plain_text = re.sub(r'\s+', ' ', plain_text)
                            plain_text = re.sub(r'^\s+|\s+$', '', plain_text, flags=re.MULTILINE)
                            return plain_text.strip()
                    except UnicodeDecodeError:
                        continue
                
                logger.warning(f"RTF 파일 인코딩 감지 실패: {file_path}, 대체 방법 시도")
            
            except ImportError:
                logger.warning("striprtf 라이브러리를 찾을 수 없습니다. 대체 방법 사용.")
            
            # 대체 방법: 직접 RTF 파싱
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # 바이너리 데이터에서 텍스트 추출 시도
            encodings = ['utf-8', 'cp949', 'euc-kr', 'latin-1']
            
            for encoding in encodings:
                try:
                    text = content.decode(encoding)
                    
                    # RTF 헤더 제거
                    if text.startswith('{\\rtf'):
                        # RTF 태그 및 헤더 제거
                        text = re.sub(r'\\[a-z0-9]+', ' ', text)  # RTF 명령어 제거
                        text = re.sub(r'\{|\}', '', text)  # 중괄호 제거
                        text = re.sub(r'\\\'[0-9a-f]{2}', '', text)  # 이스케이프 시퀀스 제거
                        text = re.sub(r'\\[^a-z]', '', text)  # 특수 명령 제거
                        
                        # 한글 유니코드 시퀀스 처리
                        text = re.sub(r'\\u-?[0-9]+\?', '', text)
                        
                        # 불필요한 공백 정리
                        text = re.sub(r'\s+', ' ', text)
                        text = text.strip()
                        
                        # 의미 있는 텍스트가 있는지 확인
                        if len(text) > 100 and not text.startswith('Times New Roman'):
                            return text
                except UnicodeDecodeError:
                    continue
            
            # 모든 방법 실패 시 기본 방법으로 시도
            try:
                text = content.decode('latin-1')
                # 기본적인 정규식 정리
                text = re.sub(r'[^\x20-\x7E\uAC00-\uD7A3]+', ' ', text)
                text = re.sub(r'\s+', ' ', text)
                return text.strip()
            except Exception:
                logger.error(f"모든 RTF 처리 방법 실패: {file_path}")
                return f"[RTF 파일 처리 실패: {os.path.basename(file_path)}]"
                
        except Exception as e:
            logger.error(f"RTF 파일 처리 오류: {e}")
            raise
    
    def _load_pdf_file(self, file_path: str) -> str:
        """PDF 파일 로드 및 텍스트 추출"""
        try:
            # PyMuPDF(fitz) 라이브러리 사용 시도
            try:
                import fitz  # PyMuPDF
                
                doc = fitz.open(file_path)
                text = ""
                
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text += page.get_text()
                    text += "\n\n"  # 페이지 구분
                
                doc.close()
                return text.strip()
                
            except ImportError:
                logger.warning("PyMuPDF(fitz) 라이브러리를 찾을 수 없습니다. PyPDF2로 시도합니다.")
                
                # PyPDF2 사용 시도
                try:
                    from PyPDF2 import PdfReader
                    
                    reader = PdfReader(file_path)
                    text = ""
                    
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n\n"
                    
                    return text.strip()
                    
                except ImportError:
                    logger.error("PDF 처리를 위한 라이브러리(PyMuPDF 또는 PyPDF2)가 설치되지 않았습니다.")
                    return f"[PDF 파일 처리 실패: {os.path.basename(file_path)}]"
        
        except Exception as e:
            logger.error(f"PDF 파일 처리 오류: {e}")
            return f"[PDF 파일 처리 오류: {os.path.basename(file_path)}]"
    
    def _load_docx_file(self, file_path: str) -> str:
        """DOCX 파일 로드 및 텍스트 추출"""
        try:
            # python-docx 라이브러리 사용 시도
            try:
                import docx
                
                doc = docx.Document(file_path)
                paragraphs = [p.text for p in doc.paragraphs]
                text = "\n".join(paragraphs)
                
                return text.strip()
                
            except ImportError:
                logger.error("DOCX 처리를 위한 python-docx 라이브러리가 설치되지 않았습니다.")
                return f"[DOCX 파일 처리 실패: {os.path.basename(file_path)}]"
        
        except Exception as e:
            logger.error(f"DOCX 파일 처리 오류: {e}")
            return f"[DOCX 파일 처리 오류: {os.path.basename(file_path)}]"
    
    def _remove_timestamps(self, content: str) -> str:
        """내용에서 타임스탬프 제거"""
        # 다양한 타임스탬프 패턴 제거
        # 1. 범위 타임스탬프 (예: 10.4 - 16.9:)
        content = re.sub(r'\d+\.\d+\s*-\s*\d+\.\d+[:：]\s*', '', content)
        
        # 2. 단일 타임스탬프 (예: 10:30:)
        content = re.sub(r'^\d+:\d+[:：]\s*', '', content, flags=re.MULTILINE)
        
        # 3. 영상 시간 표시 (예: [00:15])
        content = re.sub(r'\[\d+:\d+\]', '', content)
        
        # 연속된 빈 줄 제거
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        
        return content.strip()
    
    def _extract_metadata(self, content: str, filename: str) -> Dict[str, Any]:
        """텍스트에서 메타데이터 추출"""
        # 기본 메타데이터 설정
        metadata = {
            "filename": filename,
            "source": "천공 스승님 정법 가르침",
            "lecture_number": None,
            "title": None,
            "file_hash": hashlib.md5(content.encode('utf-8')).hexdigest(),
            "content_type": None,
            "format": os.path.splitext(filename)[1].lower(),
            "category": "미분류",  # 기본 카테고리
            "tags": []
        }
        
        # 파일명에서 강의 번호 추출 시도
        lecture_match = self.lecture_pattern.search(filename)
        if lecture_match:
            metadata["lecture_number"] = int(lecture_match.group(1))
        
        # 내용에서 강의 번호 추출 시도
        if not metadata["lecture_number"]:
            content_lecture_match = self.lecture_pattern.search(content)
            if content_lecture_match:
                metadata["lecture_number"] = int(content_lecture_match.group(1))
        
        # 강의 번호가 없는 경우 처리
        if not metadata["lecture_number"]:
            # 파일명에서 숫자 패턴 찾기
            numbers = re.findall(r'\d+', filename)
            if numbers:
                # 가장 큰 숫자를 강의 번호로 추정
                metadata["lecture_number"] = max([int(n) for n in numbers])
        
        # 제목 추출 시도 (다양한 패턴)
        for pattern in self.title_patterns:
            title_match = re.search(pattern, content[:500])  # 앞부분만 검색
            if title_match:
                metadata["title"] = title_match.group(1).strip()
                break
        
        # 제목이 없는 경우 처리
        if not metadata["title"]:
            # 파일명을 기반으로 제목 생성
            basename = os.path.basename(filename)
            name_without_ext = os.path.splitext(basename)[0]
            # 특수문자 및 숫자 제거하여 정리
            clean_name = re.sub(r'[_\-\d]+', ' ', name_without_ext).strip()
            if clean_name:
                metadata["title"] = clean_name
            else:
                # 내용의 첫 줄에서 유의미한 텍스트 추출
                first_lines = content.strip().split('\n')[:3]
                for line in first_lines:
                    clean_line = line.strip()
                    if len(clean_line) > 5 and not clean_line.startswith('http'):
                        metadata["title"] = clean_line[:50]  # 최대 50자로 제한
                        break
                
                # 여전히 제목이 없으면 기본값 설정
                if not metadata["title"]:
                    metadata["title"] = f"무제 문서 ({os.path.basename(filename)})"
        
        # 내용 유형 및 카테고리 추출
        content_type = self._detect_content_type(content)
        metadata["content_type"] = content_type
        
        # 내용 유형에 따른 카테고리 추론
        metadata["category"] = self._infer_category(content, content_type)
        
        # 태그 추출
        metadata["tags"] = self._extract_tags(content, content_type)
        
        return metadata
    
    def _detect_content_type(self, content: str) -> str:
        """문서 내용 기반 컨텐츠 유형 감지"""
        # 질문-답변 형식 감지
        if re.search(r'질문\s*:|Q:|Q\s*\.', content):
            return "lecture_qa"
        
        # 짧은 내용은 명언일 가능성
        if len(content) < 500:
            return "quote"
        
        # 뉴스 기사 감지
        if "출처:" in content or "기자" in content or "보도" in content:
            return "article"
            
        # 시/산문 감지
        if re.search(r'\n\s+\n', content) and len(re.findall(r'[.!?]', content)) < 20:
            return "poem"
            
        # 기본값은 강의
        return "lecture"
    
    def _infer_category(self, content: str, content_type: str) -> str:
        """내용 유형 및 키워드 기반 카테고리 추론"""
        # 기본 카테고리 매핑
        type_to_category = {
            "lecture": "정법강의",
            "lecture_qa": "질의응답",
            "quote": "명언/어록",
            "article": "뉴스/기사",
            "poem": "시/산문"
        }
        
        # 기본 카테고리 설정
        category = type_to_category.get(content_type, "미분류")
        
        # 키워드 기반 서브카테고리 추정
        keywords_map = {
            "홍익인간": "홍익사상",
            "제사": "전통의례",
            "용서": "인간관계",
            "탐진치": "인성수양",
            "선악": "도덕윤리",
            "병": "건강/치유",
            "깨달음": "영적성장",
            "대자연": "자연원리",
            "법칙": "우주법칙",
            "3대7": "법칙원리"
        }
        
        # 내용에서 키워드 탐색
        for keyword, subcategory in keywords_map.items():
            if keyword in content[:1000]:  # 처음 1000자 내에 키워드가 있는지 확인
                # 서브카테고리 추가
                if category == "미분류":
                    return subcategory
                else:
                    return f"{category}/{subcategory}"
        
        return category
    
    def _extract_tags(self, content: str, content_type: str) -> List[str]:
        """문서 내용 기반 태그 추출"""
        tags = []
        
        # 1. 내용 유형 태그 추가
        if content_type:
            tags.append(content_type)
        
        # 2. 중요 키워드 기반 태그 추출
        important_keywords = [
            "홍익인간", "용서", "깨달음", "정의", "선악", "인과", "탐진치", 
            "대자연", "법칙", "질서", "음양", "3대7", "천지인"
        ]
        
        # 내용에서 중요 키워드 탐색
        for keyword in important_keywords:
            if keyword in content:
                tags.append(keyword)
        
        return list(set(tags))  # 중복 제거
    
    def split_documents(self, documents: List[Dict[str, Any]], chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """
        문서를 청크로 분할하고 메타데이터 유지
        
        Args:
            documents: 문서 딕셔너리 리스트
            chunk_size: 각 청크의 최대 문자 수
            overlap: 연속된 청크 간의 중복 문자 수
                
        Returns:
            List[Dict]: 분할된 청크와 메타데이터
        """
        chunks = []
        
        for doc in documents:
            content = doc["content"]
            metadata = doc["metadata"]
            
            # 짧은 콘텐츠는 분할하지 않음
            if len(content) < chunk_size:
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = 0
                chunks.append({
                    "content": content,
                    "metadata": chunk_metadata
                })
                continue
            
            # 일반 문서는 문단 기준으로 분할
            doc_chunks = self._split_by_paragraph(content, metadata, chunk_size, overlap)
            chunks.extend(doc_chunks)
        
        logger.info(f"{len(documents)}개 문서를 {len(chunks)}개 청크로 분할")
        return chunks
    
    def _split_by_paragraph(self, content: str, metadata: Dict[str, Any], chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        """문단 기반 분할"""
        chunks = []
        
        # 문단으로 먼저 분할
        paragraphs = re.split(r'\n\s*\n', content)
        
        current_chunk = ""
        for para in paragraphs:
            # 현재 문단이 너무 긴 경우
            if len(para) > chunk_size:
                # 현재 청크가 있으면 추가
                if current_chunk:
                    chunk_metadata = metadata.copy()
                    chunk_metadata["chunk_index"] = len(chunks)
                    chunks.append({
                        "content": current_chunk.strip(),
                        "metadata": chunk_metadata
                    })
                    current_chunk = ""
                
                # 긴 문단 문장 단위로 분할
                sentences = re.split(r'(?<=[.!?])\s+', para)
                temp_chunk = ""
                
                for sentence in sentences:
                    if len(temp_chunk) + len(sentence) <= chunk_size:
                        temp_chunk += sentence + " "
                    else:
                        # 청크 추가
                        if temp_chunk:
                            chunk_metadata = metadata.copy()
                            chunk_metadata["chunk_index"] = len(chunks)
                            chunks.append({
                                "content": temp_chunk.strip(),
                                "metadata": chunk_metadata
                            })
                        temp_chunk = sentence + " "
                
                # 남은 문장 처리
                if temp_chunk:
                    current_chunk = temp_chunk
            else:
                # 현재 청크에 문단 추가 가능한지 확인
                if len(current_chunk) + len(para) <= chunk_size:
                    current_chunk += para + "\n\n"
                else:
                    # 현재 청크 추가
                    chunk_metadata = metadata.copy()
                    chunk_metadata["chunk_index"] = len(chunks)
                    chunks.append({
                        "content": current_chunk.strip(),
                        "metadata": chunk_metadata
                    })
                    current_chunk = para + "\n\n"
        
        # 마지막 청크 추가
        if current_chunk:
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = len(chunks)
            chunks.append({
                "content": current_chunk.strip(),
                "metadata": chunk_metadata
            })
        
        return chunks
    
    def is_duplicate(self, content: str, existing_hashes: List[str]) -> bool:
        """해시 기반 중복 확인"""
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        return content_hash in existing_hashes