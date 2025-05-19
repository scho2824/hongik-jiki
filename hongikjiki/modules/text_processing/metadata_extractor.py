import os
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[3]
import re
import hashlib
import logging
import sys
from typing import Dict, Any, List, Optional, Tuple


logger = logging.getLogger("HongikJikiChatBot")

class MetadataExtractor:
    """
    문서에서 메타데이터를 추출하는 클래스
    파일 내용과 파일명을 기반으로 강의 번호, 제목, 내용 유형, 카테고리, 태그 등의 메타데이터를 추출
    """
    
    def __init__(self):
        """MetadataExtractor 초기화"""
        # 정규 표현식 패턴 초기화
        self.lecture_pattern = re.compile(r'정법(\d+)강')
        self.title_patterns = [
            r'제목:\s*(.+)',
            r'강의명:\s*(.+)',
            r'\[정법강의\]\s*(.+)',
            r'정법강의 \d+강 가이드북: (.*?)[\n-]',
            r'제(\d+)부\s+(.*?)[\n]'
        ]
    
    def extract_metadata(self, content: str, filename: str, base_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        파일 내용과 파일명을 기반으로 메타데이터 추출

        Args:
            content: 분석할 텍스트 내용
            filename: 파일 이름
            base_metadata: 기존 메타데이터 (있는 경우)

        Returns:
            Dict: 추출된 메타데이터 딕셔너리
        """
        metadata = base_metadata or {}

        # 기본 필드 초기화
        metadata.update({
            "filename": filename,
            "source": "천공 스승님 정법 가르침",
            "file_hash": hashlib.md5(content.encode('utf-8')).hexdigest(),
            "format": os.path.splitext(filename)[1].lower(),
            "category": "미분류",
            "tags": []
        })

        # RTF 파일인 경우 파일명 기반 추출
        if filename.lower().endswith('.rtf'):
            return self._extract_metadata_from_filename(content, filename, metadata)

        # 강의 번호 추출 - content 우선, 없으면 filename
        lecture_number = self._extract_lecture_number(content, filename)
        if lecture_number:
            metadata["lecture_number"] = lecture_number

        # 제목 추출 - content 우선, 없으면 filename
        title = self._extract_title(content, filename)
        if title:
            metadata["title"] = title

        # 컨텐츠 유형 감지
        content_type = self._detect_content_type(content, filename)
        metadata["content_type"] = content_type
        metadata["category"] = self._infer_category(content, content_type)

        # 태그 추출
        metadata["tags"] = self._extract_tags(content, filename, metadata)

        return metadata

    def _extract_lecture_number(self, content: str, filename: str) -> Optional[int]:
        """
        내용 및 파일명에서 강의 번호 추출
        
        Args:
            content: 문서 내용
            filename: 파일 이름
            
        Returns:
            Optional[int]: 추출된 강의 번호 또는 None
        """
        # 내용에서 강의 번호 추출 시도
        lecture_patterns = [
            r'정법(\d+)강',
            r'강의 (\d+)강',
            r'(\d+)강 가이드북',
            r'(\d+)회 가이드북',
            r'(\d+)강'
        ]
        
        # 내용의 처음 500자에서 검색
        for pattern in lecture_patterns:
            match = re.search(pattern, content[:500])
            if match:
                return int(match.group(1))
        
        # 파일명에서 강의 번호 추출 시도
        lecture_match = self.lecture_pattern.search(filename)
        if lecture_match:
            return int(lecture_match.group(1))
        
        return None

    def _extract_title(self, content: str, filename: str) -> Optional[str]:
        """
        내용 및 파일명에서 제목 추출
        
        Args:
            content: 문서 내용
            filename: 파일 이름
            
        Returns:
            Optional[str]: 추출된 제목 또는 None
        """
        # 내용에서 제목 추출 시도
        for pattern in self.title_patterns:
            match = re.search(pattern, content[:1000])
            if match:
                return match.group(1).strip()
        
        # 파일명에서 제목 추출 시도
        basename = os.path.splitext(filename)[0]
        clean_name = re.sub(r'[_\-\d]+', ' ', basename).strip()
        if clean_name:
            return clean_name
        
        return None

    def _detect_content_type(self, content: str, filename: str) -> str:
        """
        내용 및 파일명 기반 컨텐츠 유형 감지
        
        Args:
            content: 문서 내용
            filename: 파일 이름
            
        Returns:
            str: 감지된 내용 유형
        """
        # 내용 기반 감지
        if re.search(r'질문\s*:|Q:|Q\s*\.', content):
            return "lecture_qa"
        if len(content) < 500:
            return "quote"
        if "출처:" in content or "기자" in content or "보도" in content:
            return "article"
        if re.search(r'\n\s+\n', content) and len(re.findall(r'[.!?]', content)) < 20:
            return "poem"
        
        # 파일명 기반 감지
        filename_lower = filename.lower()
        if "qa" in filename_lower or "질문" in filename_lower:
            return "lecture_qa"
        if "quote" in filename_lower or "명언" in filename_lower:
            return "quote"
        if "article" in filename_lower or "기사" in filename_lower:
            return "article"
        if "poem" in filename_lower or "시" in filename_lower:
            return "poem"
        
        return "lecture"

    def _infer_category(self, content: str, content_type: str) -> str:
        """
        내용 유형 및 키워드 기반 카테고리 추론
        
        Args:
            content: 문서 내용
            content_type: 감지된 내용 유형
            
        Returns:
            str: 추론된 카테고리
        """
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
            if keyword in content[:1000]:
                if category == "미분류":
                    return subcategory
                else:
                    return f"{category}/{subcategory}"
        
        return category

    def _extract_tags(self, content: str, filename: str, metadata: Dict[str, Any]) -> List[str]:
        """
        내용 및 파일명에서 태그 추출
        
        Args:
            content: 문서 내용
            filename: 파일 이름
            metadata: 기존 메타데이터
            
        Returns:
            List[str]: 추출된 태그 리스트
        """
        tags = []
        
        # 강의 번호 태그 추가
        if "lecture_number" in metadata:
            tags.append(f"정법{metadata['lecture_number']}강")
        
        # 파일명 기반 태그
        filename_lower = filename.lower()
        if "guide" in filename_lower or "가이드" in filename_lower:
            tags.append("가이드북")
        if "summary" in filename_lower or "요약" in filename_lower:
            tags.append("요약")
        if "note" in filename_lower or "노트" in filename_lower:
            tags.append("학습노트")
        
        # 내용 기반 태그
        content_tags = self._extract_content_tags(content)
        tags.extend(content_tags)
        
        return list(set(tags))  # 중복 제거

    def _extract_content_tags(self, content: str) -> List[str]:
        """
        내용에서 태그 추출
        
        Args:
            content: 문서 내용
            
        Returns:
            List[str]: 추출된 태그 리스트
        """
        tags = []
        
        # 주요 키워드 기반 태그 추출
        keywords_map = {
            "정법": "정법",
            "우주법칙": ["우주", "법칙"],
            "진리": "진리",
            "인간의 본성": ["본성", "인간의 본질"],
            "선과 악": ["선", "악"],
            "자유의지": ["자유의지", "선택", "책임"],
            "죽음과 삶": ["죽음", "삶"],
            "깨달음": ["깨달음", "깨닫"],
            "자기성찰": ["성찰", "자기를 돌아보"],
            "수행": "수행",
            "행공": "행공",
            "기도와 명상": ["명상", "기도"],
            "인간관계": ["인간관계", "관계"],
            "가족과 공동체": "가족",
            "정치": ["국가", "정치"],
            "홍익인간": ["홍익인간", "홍익"]
        }
        
        for tag, keywords in keywords_map.items():
            if isinstance(keywords, list):
                if any(keyword in content for keyword in keywords):
                    tags.append(tag)
            elif keywords in content:
                tags.append(tag)
        
        return tags

    def _extract_metadata_from_filename(self, content: str, filename: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        RTF 파일의 경우 파일명 기반으로 메타데이터 추출
        
        Args:
            content: 문서 내용
            filename: 파일 이름
            metadata: 기존 메타데이터
            
        Returns:
            Dict: 추출된 메타데이터
        """
        # 강의 번호 추출
        lecture_match = self.lecture_pattern.search(filename)
        if lecture_match:
            metadata["lecture_number"] = int(lecture_match.group(1))

        # 제목 추출
        basename = os.path.splitext(filename)[0]
        clean_name = re.sub(r'[_\-\d]+', ' ', basename).strip()
        if clean_name:
            metadata["title"] = clean_name

        # 컨텐츠 유형 감지
        content_type = self._detect_content_type_from_filename(filename)
        metadata["content_type"] = content_type
        metadata["category"] = self._infer_category_from_filename(filename)

        # 태그 추출
        metadata["tags"] = self._extract_tags_from_filename(filename)

        return metadata

    def _detect_content_type_from_filename(self, filename: str) -> str:
        """
        파일명 기반 컨텐츠 유형 감지
        
        Args:
            filename: 파일 이름
            
        Returns:
            str: 감지된 내용 유형
        """
        filename_lower = filename.lower()
        
        if "qa" in filename_lower or "질문" in filename_lower:
            return "lecture_qa"
        elif "quote" in filename_lower or "명언" in filename_lower:
            return "quote"
        elif "article" in filename_lower or "기사" in filename_lower:
            return "article"
        elif "poem" in filename_lower or "시" in filename_lower:
            return "poem"
        else:
            return "lecture"

    def _infer_category_from_filename(self, filename: str) -> str:
        """
        파일명 기반 카테고리 추론
        
        Args:
            filename: 파일 이름
            
        Returns:
            str: 추론된 카테고리
        """
        filename_lower = filename.lower()
        
        if "guide" in filename_lower or "가이드" in filename_lower:
            return "가이드북"
        elif "summary" in filename_lower or "요약" in filename_lower:
            return "요약"
        elif "note" in filename_lower or "노트" in filename_lower:
            return "학습노트"
        else:
            return "강의"

    def _extract_tags_from_filename(self, filename: str) -> List[str]:
        """
        파일명에서 태그 추출
        
        Args:
            filename: 파일 이름
            
        Returns:
            List[str]: 추출된 태그 리스트
        """
        tags = []
        filename_lower = filename.lower()
        
        # 기본 태그 추출
        if "guide" in filename_lower or "가이드" in filename_lower:
            tags.append("가이드북")
        if "summary" in filename_lower or "요약" in filename_lower:
            tags.append("요약")
        if "note" in filename_lower or "노트" in filename_lower:
            tags.append("학습노트")
            
        # 강의 번호 태그 추가
        lecture_match = self.lecture_pattern.search(filename)
        if lecture_match:
            lecture_num = lecture_match.group(1)
            tags.append(f"정법{lecture_num}강")
            
        return tags