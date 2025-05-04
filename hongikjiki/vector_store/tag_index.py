"""
Tag Index for Hongik-Jiki Chatbot

Extends vector store with tag-based search and filtering capabilities.
"""
import os
import json
from typing import Dict, List, Tuple, Any, Optional, Set
import logging
from collections import defaultdict

logger = logging.getLogger("HongikJikiChatBot")

class TagIndex:
    """
    Implements tag-based indexing and search for the vector store
    """
    def __init__(self, index_path: str = "data/tag_data/tag_index.json"):
        """
        Initialize the tag index
        
        Args:
            index_path: Path to save/load the index (optional)
        """
        self.index_path = index_path
        
        # Tag -> document mappings
        self.tag_to_docs = defaultdict(set)  # {tag_name: {doc_id1, doc_id2, ...}}
        
        # Document -> tag mappings
        self.doc_to_tags = defaultdict(dict)  # {doc_id: {tag_name: confidence, ...}}
        
        # Load existing index if available
        if os.path.exists(index_path):
            self.load_index()
    
    def add_document(self, doc_id: str, tags: Dict[str, float]) -> None:
        """
        Add a document to the tag index
        
        Args:
            doc_id: Document identifier
            tags: Dict mapping tag names to confidence scores
        """
        # Update document -> tags mapping
        self.doc_to_tags[doc_id] = tags
        
        # Update tag -> documents mappings
        for tag_name in tags:
            self.tag_to_docs[tag_name].add(doc_id)
    
    def remove_document(self, doc_id: str) -> None:
        """
        Remove a document from the tag index
        
        Args:
            doc_id: Document identifier to remove
        """
        # Get tags for this document
        tags = self.doc_to_tags.get(doc_id, {})
        
        # Remove from tag -> documents mappings
        for tag_name in tags:
            if doc_id in self.tag_to_docs[tag_name]:
                self.tag_to_docs[tag_name].remove(doc_id)
                
                # Clean up empty sets
                if not self.tag_to_docs[tag_name]:
                    del self.tag_to_docs[tag_name]
        
        # Remove from document -> tags mapping
        if doc_id in self.doc_to_tags:
            del self.doc_to_tags[doc_id]
    
    def update_document_tags(self, doc_id: str, tags: Dict[str, float]) -> None:
        """
        Update tags for a document
        
        Args:
            doc_id: Document identifier
            tags: New tags dict
        """
        # Remove old entries
        self.remove_document(doc_id)
        
        # Add new entries
        self.add_document(doc_id, tags)
    
    def get_document_tags(self, doc_id: str) -> Dict[str, float]:
        """
        Get tags for a document
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Dict mapping tag names to confidence scores
        """
        return self.doc_to_tags.get(doc_id, {})
    
    def get_documents_by_tag(self, tag_name: str) -> Set[str]:
        """
        Get all documents with a specific tag
        
        Args:
            tag_name: Tag name
            
        Returns:
            Set of document identifiers
        """
        return self.tag_to_docs.get(tag_name, set())
    
    def get_documents_by_tags(self, tags: List[str], require_all: bool = False) -> Set[str]:
        """
        Get documents matching a list of tags
        
        Args:
            tags: List of tag names
            require_all: If True, documents must have all tags
                         If False, documents with any of the tags are returned
            
        Returns:
            Set of document identifiers
        """
        if not tags:
            return set()
            
        if require_all:
            # Intersection - documents must have all tags
            result = self.get_documents_by_tag(tags[0])
            for tag in tags[1:]:
                result = result.intersection(self.get_documents_by_tag(tag))
        else:
            # Union - documents with any of the tags
            result = set()
            for tag in tags:
                result = result.union(self.get_documents_by_tag(tag))
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the tag index
        
        Returns:
            Dict with statistics
        """
        return {
            "total_documents": len(self.doc_to_tags),
            "total_tags": len(self.tag_to_docs),
            "tags_per_document": sum(len(tags) for tags in self.doc_to_tags.values()) / max(1, len(self.doc_to_tags)),
            "documents_per_tag": sum(len(docs) for docs in self.tag_to_docs.values()) / max(1, len(self.tag_to_docs)),
            "top_tags": sorted([(tag, len(docs)) for tag, docs in self.tag_to_docs.items()], 
                               key=lambda x: x[1], reverse=True)[:10]
        }
    
    def save_index(self, path: str = None) -> bool:
        """
        Save the tag index to a file
        
        Args:
            path: Output file path (defaults to self.index_path)
            
        Returns:
            bool: Success status
        """
        path = path or self.index_path
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Convert sets to lists for JSON serialization
            tag_to_docs = {tag: list(docs) for tag, docs in self.tag_to_docs.items()}
            
            data = {
                "tag_to_docs": tag_to_docs,
                "doc_to_tags": dict(self.doc_to_tags)
            }
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
            
            logger.info(f"Tag index saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving tag index: {e}")
            return False
    
    def load_index(self, path: str = None) -> bool:
        """
        Load the tag index from a file
        
        Args:
            path: Input file path (defaults to self.index_path)
            
        Returns:
            bool: Success status
        """
        path = path or self.index_path
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert lists back to sets
            self.tag_to_docs = defaultdict(set)
            for tag, docs in data.get("tag_to_docs", {}).items():
                self.tag_to_docs[tag] = set(docs)
            
            self.doc_to_tags = defaultdict(dict)
            for doc_id, tags in data.get("doc_to_tags", {}).items():
                self.doc_to_tags[doc_id] = tags
            
            logger.info(f"Tag index loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading tag index: {e}")
            return False
    
    def rebuild_from_tag_files(self, tag_dir: str) -> int:
        """
        Rebuild the index from tag files
        
        Args:
            tag_dir: Directory containing tag files
            
        Returns:
            int: Number of documents indexed
        """
        # Reset index
        self.tag_to_docs = defaultdict(set)
        self.doc_to_tags = defaultdict(dict)
        
        count = 0
        
        # Process each tag file
        if os.path.exists(tag_dir):
            for filename in os.listdir(tag_dir):
                if filename.endswith("_tags.json"):
                    try:
                        with open(os.path.join(tag_dir, filename), 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        doc_id = data.get("document_id")
                        tags = data.get("tags", {})
                        
                        if doc_id and tags:
                            self.add_document(doc_id, tags)
                            count += 1
                    except Exception as e:
                        logger.error(f"Error processing tag file {filename}: {e}")
        
        # Save the rebuilt index
        self.save_index()
        
        return count

class TagAwareSearch:
    """
    Implements tag-aware search by combining vector search with tag filtering
    """
    def __init__(self, tag_index: TagIndex):
        """
        Initialize the tag-aware search
        
        Args:
            tag_index: Tag index object
        """
        self.tag_index = tag_index
    
    def filter_results_by_tags(self, results: List[Dict[str, Any]], 
                             tags: List[str], 
                             require_all: bool = False) -> List[Dict[str, Any]]:
        """
        Filter search results by tags
        
        Args:
            results: Search results from vector store
            tags: Tags to filter by
            require_all: Whether to require all tags
            
        Returns:
            Filtered search results
        """
        if not tags:
            return results
            
        # Get documents matching the tags
        matching_docs = self.tag_index.get_documents_by_tags(tags, require_all)
        
        # Filter results
        filtered_results = []
        for result in results:
            doc_id = result.get("id") or result.get("document_id")
            if doc_id in matching_docs:
                filtered_results.append(result)
        
        return filtered_results
    
    def rerank_results_by_tags(self, results: List[Dict[str, Any]], 
                             tags: List[str],
                             tag_boost: float = 0.3) -> List[Dict[str, Any]]:
        """
        Rerank search results based on tag relevance
        
        Args:
            results: Search results from vector store
            tags: Tags to boost
            tag_boost: Weight to give tag matches (0-1)
            
        Returns:
            Reranked search results
        """
        if not tags:
            return results
        
        # Calculate tag-based scores
        reranked_results = []
        
        for result in results:
            doc_id = result.get("id") or result.get("document_id")
            doc_tags = self.tag_index.get_document_tags(doc_id)
            
            # Calculate tag score (% of query tags present in document)
            matching_tags = set(tags).intersection(set(doc_tags.keys()))
            tag_score = len(matching_tags) / len(tags) if tags else 0
            
            # Combine vector similarity score with tag score
            original_score = result.get("score", 0.0)
            combined_score = (1 - tag_boost) * original_score + tag_boost * tag_score
            
            # Create new result with adjusted score
            new_result = result.copy()
            new_result["original_score"] = original_score
            new_result["tag_score"] = tag_score
            new_result["score"] = combined_score
            new_result["matching_tags"] = list(matching_tags)
            
            reranked_results.append(new_result)
        
        # Sort by combined score
        reranked_results.sort(key=lambda x: x["score"], reverse=True)
        
        return reranked_results
    
    def extract_tags_from_query(self, query: str, tag_list: List[str]) -> Tuple[str, List[str]]:
        """
        Extract tag references from a user query
        
        Args:
            query: User query
            tag_list: List of valid tag names
            
        Returns:
            Tuple of (clean_query, extracted_tags)
        """
        # This is a simple implementation
        # A more sophisticated approach could use NLP techniques
        
        extracted_tags = []
        clean_query = query
        
        # Look for tag references in the query
        for tag in tag_list:
            # Check for tag with # prefix
            hashtag = f"#{tag}"
            if hashtag in clean_query:
                extracted_tags.append(tag)
                clean_query = clean_query.replace(hashtag, "")
            
            # Check for tag in brackets
            bracketed = f"[{tag}]"
            if bracketed in clean_query:
                extracted_tags.append(tag)
                clean_query = clean_query.replace(bracketed, "")
        
        # Clean up extra spaces
        clean_query = " ".join(clean_query.split())
        
        return clean_query, extracted_tags