"""
Tag Extractor for Hongik-Jiki Chatbot

Extracts relevant tags from document content using pattern matching and semantic similarity.
"""
import re
import json
import os
from typing import Dict, List, Optional, Set, Tuple, Any, Union
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
            raise FileNotFoundError(f"Tag patterns file not found: {patterns_file}")

        self.embedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        self.tag_phrase_embeddings = {}
        for tag, data in self.patterns.items():
            phrases = data.get("phrases", [])
            if phrases:
                joined_phrases = " ".join(phrases)
                self.tag_phrase_embeddings[tag] = self.embedding_model.encode(joined_phrases, convert_to_tensor=True)
    
    
    def load_patterns(self, file_path: str) -> None:
        """
        Load tag patterns from a JSON file

        Args:
            file_path: Path to patterns file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Unwrap if patterns are under a top-level "tags" key
            if isinstance(data, dict) and 'tags' in data and isinstance(data['tags'], dict):
                self.patterns = data['tags']
            else:
                self.patterns = data
            # Normalize pattern_data for each tag
            normalized = {}
            for tag, pdata in self.patterns.items():
                if isinstance(pdata, list):
                    normalized[tag] = {
                        "patterns": pdata,
                        "keywords": [],
                        "phrases": [],
                        "weight": 1.0
                    }
                elif isinstance(pdata, dict):
                    # Ensure all keys exist
                    normalized[tag] = {
                        "patterns": pdata.get("patterns", []),
                        "keywords": pdata.get("keywords", []),
                        "phrases": pdata.get("phrases", []),
                        "weight": pdata.get("weight", 1.0)
                    }
                else:
                    # Single string or unexpected type
                    normalized[tag] = {
                        "patterns": [str(pdata)],
                        "keywords": [],
                        "phrases": [],
                        "weight": 1.0
                    }
            self.patterns = normalized
            logger.info(f"Tag patterns normalized for {len(self.patterns)} tags")
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
    
    def extract_tags(self, content: str, existing_tags: List[str] = None, return_near: bool = False) -> Union[Dict[str, float], Tuple[Dict[str, float], List[Tuple[str, float]]]]:
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

        # Log near-threshold candidates for diagnostics
        near_candidates = self.log_near_threshold_candidates(content)
        if near_candidates:
            logger.debug(f"Near-threshold candidates for content: {near_candidates}")

        if return_near:
            return tag_scores, near_candidates
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
    def log_near_threshold_candidates(self, content: str) -> List[Tuple[str, float]]:
        """
        Return tags that scored just below the threshold for further inspection.

        Args:
            content: Document content to analyze

        Returns:
            List of (tag, score) tuples that nearly matched
        """
        near_candidates = []
        for tag_name, pattern_data in self.patterns.items():
            score = self._calculate_tag_score(content, pattern_data)
            if self.min_confidence - 0.1 <= score < self.min_confidence:
                near_candidates.append((tag_name, round(score, 3)))
        return sorted(near_candidates, key=lambda x: x[1], reverse=True)