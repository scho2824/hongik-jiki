"""
Tag Analyzer for Hongik-Jiki Chatbot

Analyzes tag relationships, co-occurrence patterns, and usage statistics.
"""
import os
import json
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set, Any, Optional
import logging
import math

from .tag_schema import TagSchema, Tag

logger = logging.getLogger("HongikJikiChatBot")

class TagAnalyzer:
    """
    Analyzes tag usage patterns and relationships
    """
    def __init__(self, tag_schema: TagSchema):
        """
        Initialize the tag analyzer
        
        Args:
            tag_schema: Tag schema object
        """
        self.tag_schema = tag_schema
        self.tag_occurrences = Counter()  # Count of each tag
        self.tag_co_occurrences = defaultdict(Counter)  # {tag1: {tag2: count}}
        
    def process_document_tags(self, doc_id: str, tags: Dict[str, float]) -> None:
        """
        Process tags from a document for analysis
        
        Args:
            doc_id: Document identifier
            tags: Dict mapping tag names to confidence scores
        """
        # Get list of tag names
        tag_names = list(tags.keys())
        
        # Update individual tag occurrences
        for tag in tag_names:
            self.tag_occurrences[tag] += 1
        
        # Update co-occurrence counts
        for i, tag1 in enumerate(tag_names):
            for tag2 in tag_names[i+1:]:
                self.tag_co_occurrences[tag1][tag2] += 1
                self.tag_co_occurrences[tag2][tag1] += 1
    
    def load_tag_statistics(self, file_path: str) -> bool:
        """
        Load tag statistics from a JSON file
        
        Args:
            file_path: Path to statistics file
            
        Returns:
            bool: Success status
        """
        try:
            if not os.path.exists(file_path):
                logger.warning(f"Tag statistics file not found: {file_path}")
                return False
                
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load tag occurrences
            self.tag_occurrences = Counter(data.get("tag_occurrences", {}))
            
            # Load co-occurrences
            co_occurrences = data.get("tag_co_occurrences", {})
            self.tag_co_occurrences = defaultdict(Counter)
            for tag1, co_tags in co_occurrences.items():
                self.tag_co_occurrences[tag1] = Counter(co_tags)
            
            logger.info(f"Tag statistics loaded from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading tag statistics from {file_path}: {e}")
            return False
    
    def save_tag_statistics(self, file_path: str) -> bool:
        """
        Save tag statistics to a JSON file
        
        Args:
            file_path: Output file path
            
        Returns:
            bool: Success status
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Prepare data for serialization
            data = {
                "tag_occurrences": dict(self.tag_occurrences),
                "tag_co_occurrences": {tag: dict(co_tags) for tag, co_tags in self.tag_co_occurrences.items()}
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Tag statistics saved to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving tag statistics to {file_path}: {e}")
            return False
    
    def get_popular_tags(self, limit: int = 10) -> List[Tuple[str, int]]:
        """
        Get the most frequently used tags
        
        Args:
            limit: Maximum number of tags to return
            
        Returns:
            List of (tag_name, count) tuples
        """
        return self.tag_occurrences.most_common(limit)
    
    def get_related_tags(self, tag_name: str, limit: int = 5) -> List[Tuple[str, float]]:
        """
        Get tags most frequently co-occurring with a given tag
        
        Args:
            tag_name: The tag to find related tags for
            limit: Maximum number of related tags to return
            
        Returns:
            List of (tag_name, relatedness_score) tuples
        """
        if tag_name not in self.tag_co_occurrences:
            return []
        
        # Get co-occurrence counts
        co_tags = self.tag_co_occurrences[tag_name]
        
        # Calculate relatedness scores using Jaccard similarity
        related_tags = []
        tag_count = self.tag_occurrences[tag_name]
        
        for co_tag, co_count in co_tags.items():
            co_tag_count = self.tag_occurrences[co_tag]
            
            # Avoid division by zero
            if tag_count == 0 or co_tag_count == 0:
                continue
                
            # Jaccard similarity: intersection size / union size
            jaccard = co_count / (tag_count + co_tag_count - co_count)
            related_tags.append((co_tag, jaccard))
        
        # Sort by relatedness score
        related_tags.sort(key=lambda x: x[1], reverse=True)
        
        return related_tags[:limit]
    
    def generate_tag_recommendations(self, input_tags: List[str], limit: int = 3) -> List[Tuple[str, float]]:
        """
        Recommend additional tags based on input tags
        
        Args:
            input_tags: List of input tag names
            limit: Maximum number of recommendations
            
        Returns:
            List of (tag_name, score) tuples
        """
        if not input_tags:
            return []
        
        # Count co-occurrences with each input tag
        recommendations = Counter()
        
        for tag in input_tags:
            if tag in self.tag_co_occurrences:
                for co_tag, count in self.tag_co_occurrences[tag].items():
                    if co_tag not in input_tags:
                        recommendations[co_tag] += count
        
        # Calculate recommendation scores
        scored_recommendations = []
        total_input_occurrences = sum(self.tag_occurrences[tag] for tag in input_tags)
        
        if total_input_occurrences == 0:
            return []
            
        for tag, count in recommendations.items():
            # Simple scoring: co-occurrence count / total input tag occurrences
            score = count / total_input_occurrences
            scored_recommendations.append((tag, score))
        
        # Sort by score
        scored_recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return scored_recommendations[:limit]
    
    def analyze_tag_relationships(self) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of tag relationships
        
        Returns:
            Dict with analysis results
        """
        results = {
            "popular_tags": self.get_popular_tags(20),
            "cluster_analysis": self._perform_cluster_analysis(),
            "tag_relationships": {},
            "category_analysis": self._analyze_categories()
        }
        
        # Analyze relationships for top tags
        top_tags = [tag for tag, _ in self.get_popular_tags(30)]
        for tag in top_tags:
            results["tag_relationships"][tag] = {
                "related_tags": self.get_related_tags(tag, 10),
                "count": self.tag_occurrences[tag]
            }
        
        return results
    
    def _perform_cluster_analysis(self) -> List[Dict[str, Any]]:
        """
        Perform cluster analysis to find groups of related tags
        
        Returns:
            List of cluster information dictionaries
        """
        # Skip if not enough data
        if len(self.tag_occurrences) < 5:
            return []
        
        # Create tag index mapping
        tags = list(self.tag_occurrences.keys())
        tag_indices = {tag: i for i, tag in enumerate(tags)}
        
        # Create co-occurrence matrix
        n_tags = len(tags)
        co_matrix = np.zeros((n_tags, n_tags))
        
        for tag1, co_tags in self.tag_co_occurrences.items():
            if tag1 in tag_indices:
                i = tag_indices[tag1]
                for tag2, count in co_tags.items():
                    if tag2 in tag_indices:
                        j = tag_indices[tag2]
                        co_matrix[i, j] = count
        
        # Simple clustering using connected components
        # This is a basic approach - more sophisticated clustering could be used
        clusters = []
        visited = set()
        
        for i, tag in enumerate(tags):
            if tag in visited:
                continue
            
            # Start a new cluster
            cluster = [tag]
            visited.add(tag)
            
            # Find connected tags (with strong connections)
            queue = [i]
            while queue:
                idx = queue.pop(0)
                # Find tags with strong connections
                for j in range(n_tags):
                    if j != idx and co_matrix[idx, j] > 0:
                        related_tag = tags[j]
                        # Calculate connection strength
                        strength = co_matrix[idx, j] / max(
                            self.tag_occurrences[tags[idx]],
                            self.tag_occurrences[related_tag]
                        )
                        
                        # Add to cluster if connection is strong enough
                        if strength > 0.3 and related_tag not in visited:
                            cluster.append(related_tag)
                            visited.add(related_tag)
                            queue.append(j)
            
            # Add cluster if it has multiple tags
            if len(cluster) > 1:
                clusters.append({
                    "tags": cluster,
                    "size": len(cluster),
                    "total_occurrences": sum(self.tag_occurrences[t] for t in cluster)
                })
        
        # Sort clusters by size
        clusters.sort(key=lambda x: x["total_occurrences"], reverse=True)
        
        return clusters[:10]  # Return top 10 clusters
    
    def _analyze_categories(self) -> Dict[str, Any]:
        """
        Analyze tag usage by category
        
        Returns:
            Dict with category analysis results
        """
        category_counts = Counter()
        category_tags = defaultdict(list)
        
        # Group tags by category
        for tag_name, count in self.tag_occurrences.items():
            tag = self.tag_schema.get_tag(tag_name)
            if tag:
                category_counts[tag.category] += count
                category_tags[tag.category].append((tag_name, count))
        
        # Sort tags within each category
        for category in category_tags:
            category_tags[category].sort(key=lambda x: x[1], reverse=True)
        
        return {
            "category_counts": dict(category_counts),
            "top_tags_by_category": {
                category: tags[:5] for category, tags in category_tags.items()
            }
        }
    
    def calculate_tag_similarity_matrix(self) -> Tuple[List[str], np.ndarray]:
        """
        Calculate tag similarity matrix based on co-occurrence patterns
        
        Returns:
            Tuple of (tag_list, similarity_matrix)
        """
        # Get tags with sufficient occurrences
        min_occurrences = 3
        frequent_tags = [tag for tag, count in self.tag_occurrences.items() 
                         if count >= min_occurrences]
        
        # Create similarity matrix
        n_tags = len(frequent_tags)
        sim_matrix = np.zeros((n_tags, n_tags))
        
        # Calculate similarities using Jaccard index
        for i, tag1 in enumerate(frequent_tags):
            tag1_count = self.tag_occurrences[tag1]
            
            for j, tag2 in enumerate(frequent_tags):
                if i == j:
                    sim_matrix[i, j] = 1.0  # Self-similarity
                    continue
                
                tag2_count = self.tag_occurrences[tag2]
                co_count = self.tag_co_occurrences[tag1][tag2]
                
                # Jaccard similarity: intersection size / union size
                if tag1_count + tag2_count > co_count:  # Avoid division by zero
                    jaccard = co_count / (tag1_count + tag2_count - co_count)
                    sim_matrix[i, j] = jaccard
        
        return frequent_tags, sim_matrix
    
    def get_tag_recommendations_for_lecture(self, lecture_id: str, existing_tags: List[str],
                                          content_tags: List[str], limit: int = 5) -> List[str]:
        """
        Get tag recommendations for a lecture
        
        Args:
            lecture_id: Lecture identifier
            existing_tags: Tags already assigned to the lecture
            content_tags: Tags extracted from content
            limit: Maximum number of recommendations
            
        Returns:
            List of recommended tag names
        """
        # Combine existing tags and content tags
        all_tags = list(set(existing_tags + content_tags))
        
        # Get recommendations based on co-occurrence patterns
        recommendations = self.generate_tag_recommendations(all_tags, limit)
        
        # Extract just the tag names
        return [tag for tag, _ in recommendations]