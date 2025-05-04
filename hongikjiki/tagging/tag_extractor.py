"""
Tag Extractor for Hongik-Jiki Chatbot

Extracts relevant tags from document content using pattern matching and semantic similarity.
"""
import re
import json
import os
from typing import Dict, List, Optional, Set, Tuple, Any
import logging
from collections import defaultdict

from sentence_transformers import SentenceTransformer, util
import torch

from .tag_schema import TagSchema, Tag

logger = logging.getLogger("HongikJikiChatBot")

class TagExtractor:
    """
    Extracts relevant tags from document content
    """
    def __init__(self, tag_schema: TagSchema, 
                patterns_file: Optional[str] = None,
                min_confidence: float = 0.5):
        """
        Initialize the tag extractor
        
        Args:
            tag_schema: Tag schema object
            patterns_file: Path to JSON patterns file (optional)
            min_confidence: Minimum confidence threshold for tag assignment
        """
        self.tag_schema = tag_schema
        self.min_confidence = min_confidence
        self.patterns = {}
        
        # Load patterns from file if provided
        if patterns_file and os.path.exists(patterns_file):
            self.load_patterns(patterns_file)
        else:
            self._init_default_patterns()

        self.embedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        self.tag_phrase_embeddings = {}
        for tag, data in self.patterns.items():
            phrases = data.get("phrases", [])
            if phrases:
                joined_phrases = " ".join(phrases)
                self.tag_phrase_embeddings[tag] = self.embedding_model.encode(joined_phrases, convert_to_tensor=True)
    
    def _init_default_patterns(self):
        """Initialize default patterns for tag extraction"""
        # Pattern structure: {tag_name: {patterns: [...], keywords: [...], phrases: [...]}}
        self.patterns = {
            # 1. 우주와 진리 (Universe & Truth)
            "정법": {
                "patterns": [r"정법([이가]\s|[은는]\s)"],
                "keywords": ["정법"],
                "phrases": ["우주의 법칙", "정법 가르침"],
                "weight": 1.0
            },
            "우주법칙": {
                "patterns": [r"우주\s?법칙", r"자연의\s?법칙"],
                "keywords": ["우주법칙", "법칙"],
                "phrases": ["우주의 원리", "자연의 원리", "법칙대로", "질서"],
                "weight": 0.8
            },
            "진리": {
                "patterns": [r"진리[는을이가]", r"참된\s?진리"],
                "keywords": ["진리"],
                "phrases": ["참된 깨달음", "본질적 진실"],
                "weight": 0.8
            },
            
            # 2. 인간 본성과 삶 (Human Nature & Life)
            "인간의 본성": {
                "patterns": [r"인간[의]?\s?본성", r"사람[의]?\s?본질"],
                "keywords": ["인간본성", "본성"],
                "phrases": ["인간의 본능", "내면의 특성"],
                "weight": 0.8
            },
            "선과 악": {
                "patterns": [r"선[과]?\s?악", r"옳고\s?그름"],
                "keywords": ["선과 악", "선악"],
                "phrases": ["도덕적 판단", "옳고 그름", "윤리"],
                "weight": 0.8
            },
            "자유의지": {
                "patterns": [r"자유의?지", r"자유로운\s?선택", r"의?지"],
                "keywords": ["자유의지", "의지", "선택"],
                "phrases": ["스스로 선택", "결정권", "자율성"],
                "weight": 0.8
            },
            
            # 3. 탐구와 인식 (Inquiry & Awareness)
            "자기성찰": {
                "patterns": [r"자기\s?성찰", r"성찰[을을]"],
                "keywords": ["성찰", "자기성찰"],
                "phrases": ["자신을 돌아보", "내면을 들여다", "스스로를 관찰"],
                "weight": 0.8
            },
            "깨달음": {
                "patterns": [r"깨달음[을을]?", r"깨우?치"],
                "keywords": ["깨달음", "깨치다"],
                "phrases": ["진리를 깨닫", "마음이 열리", "각성"],
                "weight": 0.8
            },
            
            # 4. 실천과 방법 (Practice & Method)
            "수행": {
                "patterns": [r"수행[을을]?", r"수행하[는다며]"],
                "keywords": ["수행"],
                "phrases": ["정진", "마음 수련", "영적 훈련"],
                "weight": 0.9
            },
            "행공": {
                "patterns": [r"행공[을을]?", r"행공하[는다며]"],
                "keywords": ["행공"],
                "phrases": ["기운을 다스리", "에너지 수련"],
                "weight": 0.9
            },
            "기도와 명상": {
                "patterns": [r"기도[를을]?", r"명상[을을]?"],
                "keywords": ["기도", "명상"],
                "phrases": ["마음을 고요히", "집중하여 생각"],
                "weight": 0.8
            },
            
            # 5. 사회와 현실 (Society & Reality)
            "인간관계": {
                "patterns": [r"인간\s?관계", r"사람\s?관계"],
                "keywords": ["인간관계", "관계"],
                "phrases": ["사람과의 관계", "관계 속에서", "대인관계"],
                "weight": 0.8
            },
            "갈등 해결": {
                "patterns": [r"갈등[을]?\s?해결", r"갈등[을]?\s?풀"],
                "keywords": ["갈등", "화해"],
                "phrases": ["갈등 상황", "마찰을 해소", "대립을 해결"],
                "weight": 0.8
            },
            "리더십": {
                "patterns": [r"리더[의]?", r"지도자[의]?"],
                "keywords": ["리더십", "지도자"],
                "phrases": ["이끄는 역할", "지도력", "통솔력"],
                "weight": 0.7
            },
            
            # 6. 감정 상태 (Emotional States)
            "불안": {
                "patterns": [r"불안[한을이가]?", r"걱정[이을]?"],
                "keywords": ["불안", "걱정", "염려"],
                "phrases": ["마음이 편치 않", "두려움", "신경이 쓰"],
                "weight": 0.7
            },
            "분노": {
                "patterns": [r"분노[가을]?", r"화[가을]?", r"짜증[이이]?"],
                "keywords": ["분노", "화", "짜증"],
                "phrases": ["화가 나", "참을 수 없는 감정", "마음이 불편"],
                "weight": 0.7
            },
            "평온": {
                "patterns": [r"평온[한을이가]?", r"평화로[운움]"],
                "keywords": ["평온", "평화", "고요"],
                "phrases": ["마음의 안정", "고요한 상태", "내면의 평화"],
                "weight": 0.7
            },
            
            # 7. 삶의 단계 (Life Stages)
            "청년기": {
                "patterns": [r"청년[들의]?", r"20대", r"30대"],
                "keywords": ["청년", "젊은이"],
                "phrases": ["젊은 시절", "청년 세대", "성인 초기"],
                "weight": 0.6
            },
            "중년의 위기": {
                "patterns": [r"중년[의]?", r"40대", r"50대"],
                "keywords": ["중년", "중년기"],
                "phrases": ["인생의 전환점", "위기", "중간점"],
                "weight": 0.6
            },
            "노년의 지혜": {
                "patterns": [r"노년[의]?", r"노인[들]?", r"60대"],
                "keywords": ["노년", "노인", "노후"],
                "phrases": ["인생의 후반", "노년기", "삶의 마무리"],
                "weight": 0.6
            }
        }
    
    def load_patterns(self, file_path: str) -> None:
        """
        Load tag patterns from a JSON file
        
        Args:
            file_path: Path to patterns file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.patterns = json.load(f)
            logger.info(f"Tag patterns loaded from {file_path}")
        except Exception as e:
            logger.error(f"Error loading tag patterns from {file_path}: {e}")
            # Fall back to default patterns
            self._init_default_patterns()
    
    def save_patterns(self, file_path: str) -> None:
        """
        Save current tag patterns to a JSON file
        
        Args:
            file_path: Output file path
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.patterns, f, ensure_ascii=False, indent=2)
            logger.info(f"Tag patterns saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving tag patterns to {file_path}: {e}")
    
    def extract_tags(self, content: str, existing_tags: List[str] = None) -> Dict[str, float]:
        """
        Extract relevant tags from document content
        
        Args:
            content: Document text content
            existing_tags: Any pre-existing tags (optional)
            
        Returns:
            Dict mapping tag names to confidence scores (0-1)
        """
        # Initialize with existing tags if provided
        tag_scores = {}
        if existing_tags:
            for tag in existing_tags:
                if tag in self.tag_schema.tags:
                    tag_scores[tag] = 1.0  # Existing tags get maximum confidence
        
        # Score content against all tag patterns
        for tag_name, pattern_data in self.patterns.items():
            if tag_name in tag_scores:
                continue  # Skip if already added from existing tags
            
            score = self._calculate_tag_score(content, pattern_data)
            
            if score >= self.min_confidence:
                tag_scores[tag_name] = score
        
        # Add parent tags for any child tags with high confidence
        parent_tags = self._get_parent_tags(tag_scores)
        for parent, score in parent_tags.items():
            if parent not in tag_scores or tag_scores[parent] < score:
                tag_scores[parent] = score
        
        return tag_scores
    
    def _calculate_tag_score(self, content: str, pattern_data: Dict[str, Any]) -> float:
        """
        Calculate a confidence score for a tag based on pattern matches
        
        Args:
            content: Document content
            pattern_data: Pattern data for the tag
            
        Returns:
            float: Confidence score (0-1)
        """
        total_score = 0.0
        weight = pattern_data.get("weight", 1.0)
        
        # Check regex patterns (strongest evidence)
        pattern_matches = 0
        patterns = pattern_data.get("patterns", [])
        for pattern in patterns:
            matches = re.findall(pattern, content)
            pattern_matches += len(matches)
        
        # More matches = higher confidence, with diminishing returns
        if pattern_matches > 0:
            pattern_score = min(1.0, 0.5 + (pattern_matches * 0.1))
            total_score += pattern_score * 0.6  # Patterns are weighted highest
        
        # Check direct keyword matches
        keyword_matches = 0
        keywords = pattern_data.get("keywords", [])
        for keyword in keywords:
            # Simple keyword matching
            if keyword in content:
                keyword_matches += content.count(keyword)
        
        if keyword_matches > 0:
            keyword_score = min(1.0, 0.3 + (keyword_matches * 0.1))
            total_score += keyword_score * 0.3  # Keywords weighted second
        
        # Check phrase matches (weakest evidence, but helpful for context)
        phrase_matches = 0
        phrases = pattern_data.get("phrases", [])
        for phrase in phrases:
            if phrase in content:
                phrase_matches += content.count(phrase)
        
        if phrase_matches > 0:
            phrase_score = min(1.0, 0.2 + (phrase_matches * 0.1))
            total_score += phrase_score * 0.1  # Phrases weighted lowest
        
        # Apply tag-specific weight factor
        final_score = total_score * weight
        
        # Return normalized score
        return min(1.0, final_score)
    
    def _get_parent_tags(self, tag_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Get parent tags for any child tags with high confidence
        
        Args:
            tag_scores: Dict of tag names to confidence scores
            
        Returns:
            Dict of parent tag names to confidence scores
        """
        parent_scores = {}
        
        for tag_name, score in tag_scores.items():
            tag = self.tag_schema.get_tag(tag_name)
            if tag and tag.parent:
                # Parent gets a slightly lower confidence than the child
                parent_score = score * 0.9
                if tag.parent not in parent_scores or parent_scores[tag.parent] < parent_score:
                    parent_scores[tag.parent] = parent_score
        
        return parent_scores
    
    def extract_tags_from_query(self, query: str, max_tags: int = 3) -> List[str]:
        """
        Extract relevant tags from a user query using pattern matching, fuzzy keyword similarity, and embedding similarity.

        Args:
            query: User question or query
            max_tags: Maximum number of tags to return

        Returns:
            List of most relevant tag names
        """
        from difflib import SequenceMatcher

        def is_similar(a: str, b: str, threshold: float = 0.8) -> bool:
            return SequenceMatcher(None, a, b).ratio() > threshold

        query_min_confidence = self.min_confidence * 0.7
        tag_scores = {}

        # Encode query once
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)

        for tag_name, pattern_data in self.patterns.items():
            score = self._calculate_tag_score(query, pattern_data)

            # Fuzzy keyword matching to boost score
            for keyword in pattern_data.get("keywords", []):
                for word in query.split():
                    if is_similar(word.lower(), keyword.lower()):
                        score += 0.2
                        break

            # Embedding similarity with tag phrases
            tag_embedding = self.tag_phrase_embeddings.get(tag_name)
            if tag_embedding is not None:
                similarity = util.pytorch_cos_sim(query_embedding, tag_embedding).item()
                if similarity > 0.5:
                    score += similarity * 0.3  # Scale contribution

            if score >= query_min_confidence:
                tag_scores[tag_name] = min(score, 1.0)

        sorted_tags = sorted(tag_scores.items(), key=lambda x: x[1], reverse=True)
        return [tag for tag, score in sorted_tags[:max_tags]]
    
    def update_pattern(self, tag_name: str, 
                     patterns: List[str] = None,
                     keywords: List[str] = None, 
                     phrases: List[str] = None,
                     weight: float = None) -> None:
        """
        Update pattern data for a specific tag
        
        Args:
            tag_name: The tag to update
            patterns: New regex patterns (optional)
            keywords: New keywords (optional)
            phrases: New phrases (optional)
            weight: New weight (optional)
        """
        if tag_name not in self.patterns:
            self.patterns[tag_name] = {
                "patterns": [],
                "keywords": [],
                "phrases": [],
                "weight": 1.0
            }
        
        pattern_data = self.patterns[tag_name]
        
        if patterns is not None:
            pattern_data["patterns"] = patterns
        
        if keywords is not None:
            pattern_data["keywords"] = keywords
        
        if phrases is not None:
            pattern_data["phrases"] = phrases
        
        if weight is not None:
            pattern_data["weight"] = weight