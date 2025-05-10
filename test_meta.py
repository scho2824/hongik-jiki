# test_meta.py
import logging
from hongikjiki.text_processing.metadata_extractor import MetadataExtractor

# 콘솔에 debug 출력하도록 로거 설정
logging.basicConfig(level=logging.DEBUG)

# 샘플 텍스트 & 파일명
content = """
정법64강 가이드북: 효와 성장
제목: 부모와 자녀의 성장
Q: 이 강의는 무엇을 말하고 있나요?
A: 자녀의 성장기에 부모의 정성이 결정적이다.
"""
filename = "lecture_64_guidance.txt"

# 메타데이터 추출
me = MetadataExtractor()
meta = me.extract_metadata(content, filename)
print("\n>>>>> 추출된 메타데이터:", meta)