# hongikjiki/core/formatter.py
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[2]
import logging
import re

logger = logging.getLogger("HongikJikiChatBot")

def format_response(results, answer, extracted_tags=None):
    """챗봇 응답을 포맷팅하는 함수"""
    # 태그 추출
    tags = set(sorted(extracted_tags or []))
    quoted_insights = []
    source_info = []
    
    # 검색 결과에서 정보 추출
    for i, result in enumerate(results):
        # 메타데이터에서 태그 추출
        metadata = result.get('metadata', {})
        if isinstance(metadata, dict) and 'tags' in metadata:
            result_tags = metadata['tags']
            if isinstance(result_tags, list):
                tags.update(result_tags)
            elif isinstance(result_tags, str):
                tags.update([t.strip() for t in result_tags.split(',')])
        
        # 강의 정보 추출
        lecture_id = metadata.get('lecture_id', '')
        lecture_title = metadata.get('lecture_title', '')
        
        if lecture_id or lecture_title:
            info = f"📘 강의 {i+1}:"
            if lecture_title:
                info += f" 「{lecture_title}」"
            if lecture_id:
                info += f" (강의 번호: {lecture_id})"
            source_info.append(info)
        
        # 인용문 추출
        content = result.get('content', '')
        if content:
            match = re.search(r"A[:：]\s*(.+?)(?:\.|\n|$)", content)
            if match:
                sentence = match.group(1).strip()
                if 10 <= len(sentence) <= 100:
                    quoted_insights.append(f'"{sentence}"')
    
    # 응답 포맷팅
    formatted = f"{answer}\n\n"
    
    # 출처 정보 추가
    if source_info:
        formatted += f"🔗 출처:\n" + "\n".join(source_info) + "\n\n"
    
    # 인용문 추가
    if quoted_insights:
        formatted += f"🔎 관련 인용:\n> {quoted_insights[0]}\n\n"
    
    # 태그 추가
    if tags and len(tags) > 0:  # 태그가 실제로 있는 경우에만
        tag_list = ' '.join([f"#{tag}" for tag in tags])
        formatted += f"🏷️ 관련 태그: {tag_list}"
    
    return formatted