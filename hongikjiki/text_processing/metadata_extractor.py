import os
import re
import hashlib
import logging
import sys
from typing import Dict, Any, List, Optional

logger = logging.getLogger("HongikJikiChatBot")

class MetadataExtractor:
    """
    문서에서 메타데이터를 추출하는 클래스
    강의 번호, 제목, 내용 유형, 카테고리, 태그 등의 메타데이터를 추출
    """
    
    def __init__(self):
        """MetadataExtractor 초기화"""
        # 정규 표현식 패턴 초기화
        self.lecture_pattern = re.compile(r'정법(\d+)강')
        self.title_patterns = [
            r'제목:\s*(.+)',
            r'강의명:\s*(.+)',
            r'\[정법강의\]\s*(.+)'
        ]
    
    def extract_metadata(self, content: str, filename: str, base_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        텍스트에서 메타데이터 추출 (개선된 버전)

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
            "lecture_number": metadata.get("lecture_number"),
            "title": metadata.get("title"),
            "file_hash": hashlib.md5(content.encode('utf-8')).hexdigest(),
            "content_type": metadata.get("content_type"),
            "format": metadata.get("format", os.path.splitext(filename)[1].lower()),
            "category": metadata.get("category", "미분류"),
            "tags": metadata.get("tags", [])
        })

        # 강의 번호 추출 - 다양한 패턴 인식
        lecture_patterns = [
            r'정법(\d+)강',
            r'강의 (\d+)강',
            r'(\d+)강 가이드북',
            r'(\d+)회 가이드북'
        ]
        for pattern in lecture_patterns:
            match = re.search(pattern, content[:500] + " " + filename)
            if match:
                metadata["lecture_number"] = int(match.group(1))
                break

        # 제목 추출 - 가이드북 형식 인식 추가
        title_patterns = [
            r'제목:\s*(.+)',
            r'강의명:\s*(.+)',
            r'\[정법강의\]\s*(.+)',
            r'정법강의 \d+강 가이드북: (.*?)[\n-]',
            r'제(\d+)부\s+(.*?)[\n]'
        ]
        for pattern in title_patterns:
            match = re.search(pattern, content[:1000])
            if match:
                metadata["title"] = match.group(1).strip()
                break

        # Q&A 형식 감지
        if "질문 :" in content[:3000] or "질문:" in content[:3000]:
            metadata["content_type"] = "lecture_qa"
            metadata["category"] = "질의응답"

        # 가이드북 형식 감지
        if "가이드북" in filename or "가이드북" in content[:500]:
            metadata["content_type"] = "guidebook"
            metadata["category"] = "가이드북"

        # 생활도 형식 감지
        if "생활도" in filename or "생활도" in content[:500]:
            metadata["content_type"] = "daily_wisdom"
            metadata["category"] = "생활도"
            
        # 태그 추출 - 신규 추가
        if not metadata.get("tags"):
            metadata["tags"] = self.extract_tags(content, filename, metadata)

        return metadata
    
    def extract_tags(self, content: str, filename: str, existing_metadata: Dict[str, Any] = None) -> List[str]:
        """
        문서 내용에서 관련 태그를 추출
        
        Args:
            content: 문서 내용
            filename: 파일 이름
            existing_metadata: 기존 메타데이터 (있는 경우)
            
        Returns:
            List[str]: 추출된 태그 리스트
        """
        # 태그 모듈 로드 시도
        tag_extractor = None
        try:
            # 프로젝트 루트 경로 추가
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            if project_root not in sys.path:
                sys.path.append(project_root)
                
            from hongikjiki.tagging.tag_schema import TagSchema
            from hongikjiki.tagging.tag_extractor import TagExtractor
            
            # 태그 스키마 및 패턴 파일 경로 확인
            tag_schema_path = os.path.join(project_root, 'data', 'config', 'tag_schema.yaml')
            tag_patterns_path = os.path.join(project_root, 'data', 'config', 'tag_patterns.json')
            
            if os.path.exists(tag_schema_path):
                tag_schema = TagSchema(tag_schema_path)
                tag_extractor = TagExtractor(
                    tag_schema, 
                    patterns_file=tag_patterns_path if os.path.exists(tag_patterns_path) else None
                )
        except ImportError:
            logger.warning("태그 모듈을 불러올 수 없습니다. 기본 태그 추출 방식을 사용합니다.")
        except Exception as e:
            logger.error(f"태그 추출기 초기화 오류: {e}")
        
        # 기존 태그가 있으면 반환
        if existing_metadata and "tags" in existing_metadata and existing_metadata["tags"]:
            return existing_metadata["tags"]
        
        # 태그 추출기 사용 가능하면 사용
        if tag_extractor:
            tag_scores = tag_extractor.extract_tags(content)
            # 신뢰도 0.6 이상인 태그만 선택
            return [tag for tag, score in tag_scores.items() if score >= 0.6]
        
        # 태그 추출기가 없으면 기본 추출 방식 사용
        return self._extract_basic_tags(content, filename)
    
    def _extract_basic_tags(self, content: str, filename: str) -> List[str]:
        """
        기본적인 태그 추출 로직
        
        Args:
            content: 문서 내용
            filename: 파일 이름
            
        Returns:
            List[str]: 추출된 기본 태그 리스트
        """
        tags = []
        
        # 주요 키워드 기반 태그 추출
        # 우주와 진리 카테고리
        if "정법" in content:
            tags.append("정법")
        
        if "우주" in content and ("법칙" in content or "원리" in content):
            tags.append("우주법칙")
        
        if "진리" in content:
            tags.append("진리")
            
        # 인간 본성과 삶 카테고리
        if "본성" in content or "인간의 본질" in content:
            tags.append("인간의 본성")
            
        if "선" in content and "악" in content:
            tags.append("선과 악")
            
        if "자유의지" in content or "선택" in content and "책임" in content:
            tags.append("자유의지")
            
        if "죽음" in content and "삶" in content:
            tags.append("죽음과 삶")
            
        # 탐구와 인식 카테고리
        if "깨달음" in content or "깨닫" in content:
            tags.append("깨달음")
            
        if "성찰" in content or "자기를 돌아보" in content:
            tags.append("자기성찰")
            
        # 실천과 방법 카테고리
        if "수행" in content:
            tags.append("수행")
            
        if "행공" in content:
            tags.append("행공")
            
        if "명상" in content or "기도" in content:
            tags.append("기도와 명상")
            
        # 사회와 현실 카테고리
        if "인간관계" in content or "관계" in content and "사람" in content:
            tags.append("인간관계")
            
        if "가족" in content:
            tags.append("가족과 공동체")
            
        if "국가" in content or "정치" in content:
            tags.append("정치")
            
        # 감정 상태 카테고리
        emotions = {
            "불안": ["불안", "걱정", "두려움"],
            "분노": ["분노", "화", "짜증"],
            "슬픔": ["슬픔", "우울", "비통"],
            "평온": ["평온", "평화", "고요"],
            "기쁨": ["기쁨", "행복", "즐거움"]
        }
        
        for emotion, keywords in emotions.items():
            for keyword in keywords:
                if keyword in content:
                    tags.append(emotion)
                    break
        
        # 홍익인간 특별 태그 (핵심 개념)
        if "홍익인간" in content or "홍익" in content and "인간" in content:
            tags.append("홍익인간")
        
        return list(set(tags))  # 중복 제거
    
    def _extract_lecture_number(self, content: str, filename: str, default_number: int = None) -> int:
        """
        내용 및 파일명에서 강의 번호 추출
        
        Args:
            content: 문서 내용
            filename: 파일 이름
            default_number: 기본 강의 번호 (이미 설정된 경우)
            
        Returns:
            int: 추출된 강의 번호 또는 None
        """
        if default_number:
            return default_number
        
        # 파일명에서 강의 번호 추출 시도
        lecture_match = self.lecture_pattern.search(filename)
        if lecture_match:
            return int(lecture_match.group(1))
        
        # 내용에서 강의 번호 추출 시도
        content_lecture_match = self.lecture_pattern.search(content)
        if content_lecture_match:
            return int(content_lecture_match.group(1))
        
        # 강의 번호가 없는 경우 처리
        # 파일명에서 숫자 패턴 찾기
        numbers = re.findall(r'\d+', filename)
        if numbers:
            # 가장 큰 숫자를 강의 번호로 추정
            return max([int(n) for n in numbers])
        
        return None
    
    def _extract_title(self, content: str, filename: str, default_title: str = None) -> str:
        """
        내용 및 파일명에서 제목 추출
        
        Args:
            content: 문서 내용
            filename: 파일 이름
            default_title: 기본 제목 (이미 설정된 경우)
            
        Returns:
            str: 추출된 제목
        """
        if default_title:
            return default_title
        
        # 제목 추출 시도 (다양한 패턴)
        for pattern in self.title_patterns:
            title_match = re.search(pattern, content[:500])  # 앞부분만 검색
            if title_match:
                return title_match.group(1).strip()
        
        # 제목이 없는 경우 처리
        # 파일명을 기반으로 제목 생성
        basename = os.path.basename(filename)
        name_without_ext = os.path.splitext(basename)[0]
        # 특수문자 및 숫자 제거하여 정리
        clean_name = re.sub(r'[_\-\d]+', ' ', name_without_ext).strip()
        if clean_name:
            return clean_name
        
        # 내용의 첫 줄에서 유의미한 텍스트 추출
        first_lines = content.strip().split('\n')[:3]
        for line in first_lines:
            clean_line = line.strip()
            if len(clean_line) > 5 and not clean_line.startswith('http'):
                return clean_line[:50]  # 최대 50자로 제한
        
        # 여전히 제목이 없으면 기본값 설정
        return f"무제 문서 ({os.path.basename(filename)})"
    
    def _detect_content_type(self, content: str) -> str:
        """
        문서 내용 기반 컨텐츠 유형 감지
        
        Args:
            content: 문서 내용
            
        Returns:
            str: 감지된 내용 유형
        """
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
            if keyword in content[:1000]:  # 처음 1000자 내에 키워드가 있는지 확인
                # 서브카테고리 추가
                if category == "미분류":
                    return subcategory
                else:
                    return f"{category}/{subcategory}"
        
        return category