import os
import re
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("HongikJikiChatBot")

class DocumentLoader:
    """
    다양한 형식의 문서를 로드하고 텍스트를 추출하는 클래스
    """
    
    def __init__(self):
        """DocumentLoader 초기화"""
        pass
    
    def load_document(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        파일 경로로부터 문서를 로드하고 텍스트와 기본 메타데이터 추출
        
        Args:
            file_path: 로드할 파일 경로
            
        Returns:
            Dict: 문서 내용과 기본 메타데이터를 포함하는 딕셔너리 또는 None (로드 실패)
        """
        try:
            filename = os.path.basename(file_path)
            file_ext = os.path.splitext(filename)[1].lower()
            
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
                return None
                
            if content:
                # 내용이 너무 짧으면 처리 오류 가능성 있음
                if len(content.strip()) < 10:
                    logger.warning(f"파일 내용이 너무 짧아 처리 오류 의심: {filename}")
                
                # 기본 메타데이터 생성
                metadata = {
                    "filename": filename,
                    "format": file_ext,
                    "file_path": file_path
                }
                
                return {
                    "content": content,
                    "metadata": metadata
                }
                
            return None
            
        except Exception as e:
            logger.error(f"파일 로드 실패: {file_path}, 오류: {e}")
            logger.exception(e)  # 상세 오류 로깅
            return None
    
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

    def load_documents_from_dir(self, base_dir: str) -> list[Dict[str, Any]]:
        """
        주어진 디렉토리의 모든 하위 파일을 순회하며 문서를 로드합니다.
        지원되는 파일 확장자: .txt, .md, .rtf, .pdf, .docx

        Returns:
            로드된 문서들의 리스트 (content, metadata 포함)
        """
        supported_exts = ['.txt', '.md', '.rtf', '.pdf', '.docx']
        loaded_documents = []

        for root, _, files in os.walk(base_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in supported_exts:
                    file_path = os.path.join(root, file)
                    doc = self.load_document(file_path)
                    if doc:
                        # 추가 메타 정보 (하위 폴더명을 label로 저장)
                        relative = os.path.relpath(root, base_dir)
                        doc["metadata"]["label"] = relative
                        loaded_documents.append(doc)

        logger.info(f"{len(loaded_documents)}개의 문서를 로드했습니다.")
        return loaded_documents