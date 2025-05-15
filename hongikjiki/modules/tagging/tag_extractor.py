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
                min_confidence: float = 0.5):
        """
        Initialize the tag extractor
        
        Args:
            tag_schema: Tag schema object
            min_confidence: Minimum confidence threshold for tag assignment
        """
        self.tag_schema = tag_schema
        self.min_confidence = min_confidence

        self.embedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        self.tag_phrase_embeddings = {}
        for tag in tag_schema.get_all_tags():
            if tag.phrases:
                joined = " ".join(tag.phrases)
                self.tag_phrase_embeddings[tag.name] = self.embedding_model.encode(joined, convert_to_tensor=True)
    
    def extract_tags(self, content: str, existing_tags: Optional[List[str]] = None, return_near: bool = False) -> Union[Dict[str, float], Tuple[Dict[str, float], List[Tuple[str, float]]]]:
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
        for tag in self.tag_schema.get_all_tags():
            tag_name = tag.name
            if tag_name in tag_scores:
                continue  # Skip if already added from existing tags
            
            score = self._calculate_tag_score(content, tag)
            
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
            formatted = ", ".join([f"{tag}:{score}" for tag, score in near_candidates])
            logger.debug(f"🟡 Near-threshold tags for document: {formatted}")

        if return_near:
            return tag_scores, near_candidates
        return tag_scores
    
    def _calculate_tag_score(self, content: str, tag: Tag) -> float:
        """
        Calculate a confidence score for a tag based on pattern matches
        
        Args:
            content: Document content
            tag: Tag object
            
        Returns:
            float: Confidence score (0-1)
        """
        total_score = 0.0
        weight = getattr(tag, "weight", 1.0)
        
        # Check regex patterns (strongest evidence)
        pattern_matches = 0
        patterns = getattr(tag, "patterns", []) or []
        for pattern in patterns:
            matches = re.findall(pattern, content)
            pattern_matches += len(matches)
        
        # More matches = higher confidence, with diminishing returns
        if pattern_matches > 0:
            pattern_score = min(1.0, 0.5 + (pattern_matches * 0.1))
            total_score += pattern_score * 0.6  # Patterns are weighted highest
        
        # Check direct keyword matches
        keyword_matches = 0
        keywords = getattr(tag, "keywords", []) or []
        for keyword in keywords:
            # Simple keyword matching
            if keyword in content:
                keyword_matches += content.count(keyword)
        
        if keyword_matches > 0:
            keyword_score = min(1.0, 0.3 + (keyword_matches * 0.1))
            total_score += keyword_score * 0.3  # Keywords weighted second
        
        # Check phrase matches (weakest evidence, but helpful for context)
        phrase_matches = 0
        phrases = getattr(tag, "phrases", []) or []
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

        for tag in self.tag_schema.get_all_tags():
            tag_name = tag.name
            score = self._calculate_tag_score(query, tag)

            # Fuzzy keyword matching to boost score
            for keyword in getattr(tag, "keywords", []) or []:
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
    
    def log_near_threshold_candidates(self, content: str) -> List[Tuple[str, float]]:
        """
        Return tags that scored just below the threshold for further inspection.

        Args:
            content: Document content to analyze

        Returns:
            List of (tag, score) tuples that nearly matched
        """
        near_candidates = []
        for tag in self.tag_schema.get_all_tags():
            score = self._calculate_tag_score(content, tag)
            if self.min_confidence - 0.1 <= score < self.min_confidence:
                near_candidates.append((tag.name, round(score, 3)))
        # Cap the number of reported candidates to avoid flooding logs
        return sorted(near_candidates, key=lambda x: x[1], reverse=True)[:5]