"""
Tagging Tools for Hongik-Jiki Chatbot

Utilities for manual tagging and tag management.
"""
import os
import json
import csv
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from datetime import datetime

from .tag_schema import TagSchema, Tag
from .tag_extractor import TagExtractor
from .tag_analyzer import TagAnalyzer

logger = logging.getLogger("HongikJikiChatBot")

class TaggingSession:
    """
    Manages a manual tagging session for documents
    """
    def __init__(self, tag_schema: TagSchema, tag_extractor: TagExtractor, 
                tag_analyzer: Optional[TagAnalyzer] = None,
                output_dir: str = "data/tag_data/manually_tagged"):
        """
        Initialize a tagging session
        
        Args:
            tag_schema: Tag schema object
            tag_extractor: Tag extractor object
            tag_analyzer: Tag analyzer object (optional)
            output_dir: Directory for saving tagged documents
        """
        self.tag_schema = tag_schema
        self.tag_extractor = tag_extractor
        self.tag_analyzer = tag_analyzer
        self.output_dir = output_dir
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Session state
        self.current_document = None
        self.suggested_tags = {}
        self.manual_tags = {}
        self.session_info = {
            "start_time": datetime.now().isoformat(),
            "documents_tagged": 0
        }
    
    def load_document(self, document_id: str, content: str, 
                    metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Load a document for tagging
        
        Args:
            document_id: Document identifier
            content: Document text content
            metadata: Document metadata (optional)
            
        Returns:
            Dict with document info and suggested tags
        """
        # Save current document if any
        if self.current_document and self.manual_tags:
            self.save_document_tags()
        
        # Load new document
        self.current_document = {
            "id": document_id,
            "content": content,
            "metadata": metadata or {}
        }
        
        # Get existing tags if any
        existing_tags = []
        if metadata and "tags" in metadata:
            if isinstance(metadata["tags"], list):
                existing_tags = metadata["tags"]
            elif isinstance(metadata["tags"], dict):
                existing_tags = list(metadata["tags"].keys())
            else:
                try:
                    # Try to parse as comma-separated string
                    existing_tags = [tag.strip() for tag in str(metadata["tags"]).split(",") if tag.strip()]
                except Exception:
                    pass
        
        # Extract suggested tags
        self.suggested_tags = self.tag_extractor.extract_tags(content, existing_tags)
        
        # Initialize manual tags with existing tags
        if existing_tags:
            if isinstance(metadata.get("tags"), dict):
                # If tags are already a dict with scores, use as-is
                self.manual_tags = metadata["tags"]
            else:
                # Convert list to dict with default confidence 1.0
                self.manual_tags = {tag: 1.0 for tag in existing_tags}
        else:
            self.manual_tags = {}
        
        # Get tag recommendations if analyzer is available
        recommendations = []
        if self.tag_analyzer:
            content_tags = list(self.suggested_tags.keys())
            recommendations = self.tag_analyzer.get_tag_recommendations_for_lecture(
                document_id, existing_tags, content_tags
            )
        
        return {
            "document_id": document_id,
            "suggested_tags": self.suggested_tags,
            "manual_tags": self.manual_tags,
            "recommended_tags": recommendations
        }
    
    def add_tag(self, tag_name: str, confidence: float = 1.0) -> bool:
        """
        Add a tag to the current document
        
        Args:
            tag_name: Tag name to add
            confidence: Confidence score (0-1)
            
        Returns:
            bool: Success status
        """
        if not self.current_document:
            logger.warning("No document loaded for tagging")
            return False
            
        if tag_name not in self.tag_schema.tags:
            logger.warning(f"Tag '{tag_name}' not found in schema")
            return False
            
        self.manual_tags[tag_name] = confidence
        return True
    
    def remove_tag(self, tag_name: str) -> bool:
        """
        Remove a tag from the current document
        
        Args:
            tag_name: Tag name to remove
            
        Returns:
            bool: Success status
        """
        if not self.current_document:
            logger.warning("No document loaded for tagging")
            return False
            
        if tag_name in self.manual_tags:
            del self.manual_tags[tag_name]
            return True
        return False
    
    def accept_suggested_tag(self, tag_name: str) -> bool:
        """
        Accept a suggested tag
        
        Args:
            tag_name: Suggested tag to accept
            
        Returns:
            bool: Success status
        """
        if not self.current_document:
            logger.warning("No document loaded for tagging")
            return False
            
        if tag_name in self.suggested_tags:
            self.manual_tags[tag_name] = self.suggested_tags[tag_name]
            return True
        return False
    
    def accept_all_suggested_tags(self, min_confidence: float = 0.7) -> int:
        """
        Accept all suggested tags above a confidence threshold
        
        Args:
            min_confidence: Minimum confidence threshold
            
        Returns:
            int: Number of tags accepted
        """
        if not self.current_document:
            logger.warning("No document loaded for tagging")
            return 0
            
        count = 0
        for tag, confidence in self.suggested_tags.items():
            if confidence >= min_confidence:
                self.manual_tags[tag] = confidence
                count += 1
        
        return count
    
    def get_current_tags(self) -> Dict[str, float]:
        """
        Get current tags for the document
        
        Returns:
            Dict mapping tag names to confidence scores
        """
        return self.manual_tags.copy()
    
    def save_document_tags(self) -> bool:
        """
        Save tags for the current document
        
        Returns:
            bool: Success status
        """
        if not self.current_document:
            logger.warning("No document loaded for tagging")
            return False
            
        try:
            # Prepare output file path
            document_id = self.current_document["id"]
            file_name = os.path.basename(document_id) if os.path.sep in document_id else document_id
            safe_id = file_name.replace("/", "_").replace("\\", "_")
            output_file = os.path.join(self.output_dir, f"{safe_id}_tags.json")
            
            # Prepare data
            data = {
                "document_id": document_id,
                "file": file_name,  # Add the original filename
                "tags": self.manual_tags,
                "metadata": self.current_document["metadata"],
                "tagging_time": datetime.now().isoformat()
            }
            
            # Save to file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            # Update session stats
            self.session_info["documents_tagged"] += 1
            logger.info(f"Tags saved for document {document_id}")
            
            # Update analyzer if available
            if self.tag_analyzer:
                self.tag_analyzer.process_document_tags(document_id, self.manual_tags)
            
            return True
        except Exception as e:
            logger.error(f"Error saving tags for document {self.current_document['id']}: {e}")
            return False
    
    def export_session_stats(self, output_file: str) -> bool:
        """
        Export session statistics
        
        Args:
            output_file: Output file path
            
        Returns:
            bool: Success status
        """
        try:
            # Update end time
            self.session_info["end_time"] = datetime.now().isoformat()
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Save to file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.session_info, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Session stats exported to {output_file}")
            return True
        except Exception as e:
            logger.error(f"Error exporting session stats: {e}")
            return False

class TaggingBatch:
    """
    Manages batch processing for document tagging
    """
    def __init__(self, tag_schema: TagSchema, tag_extractor: TagExtractor,
                output_dir: str = "data/tag_data/auto_tagged"):
        """
        Initialize a batch tagging processor
        
        Args:
            tag_schema: Tag schema object
            tag_extractor: Tag extractor object
            output_dir: Directory for saving tagged documents
        """
        self.tag_schema = tag_schema
        self.tag_extractor = tag_extractor
        self.output_dir = output_dir
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Batch stats
        self.stats = {
            "total_documents": 0,
            "successfully_tagged": 0,
            "total_tags_applied": 0,
            "tags_per_document": 0,
            "start_time": datetime.now().isoformat()
        }
    
    def process_document(self, document_id: str, content: str, 
                       metadata: Optional[Dict[str, Any]] = None,
                       min_confidence: float = 0.6,
                       save_result: bool = True) -> Dict[str, float]:
        """
        Process a single document
        
        Args:
            document_id: Document identifier
            content: Document text content
            metadata: Document metadata (optional)
            min_confidence: Minimum confidence threshold for tags
            save_result: Whether to save the result
            
        Returns:
            Dict mapping tag names to confidence scores
        """
        # Get existing tags if any
        existing_tags = []
        if metadata and "tags" in metadata:
            if isinstance(metadata["tags"], list):
                existing_tags = metadata["tags"]
            elif isinstance(metadata["tags"], dict):
                existing_tags = list(metadata["tags"].keys())
            else:
                try:
                    # Try to parse as comma-separated string
                    existing_tags = [tag.strip() for tag in str(metadata["tags"]).split(",") if tag.strip()]
                except Exception:
                    pass
        
        # Extract tags
        tags = self.tag_extractor.extract_tags(content, existing_tags)
        
        # Filter by confidence threshold
        filtered_tags = {tag: score for tag, score in tags.items() if score >= min_confidence}
        
        # Update stats
        self.stats["total_documents"] += 1
        if filtered_tags:
            self.stats["successfully_tagged"] += 1
            self.stats["total_tags_applied"] += len(filtered_tags)
        
        # Save result if requested
        if save_result:
            file_name = os.path.basename(document_id) if os.path.sep in document_id else document_id
            self._save_document_tags(document_id, file_name, filtered_tags, metadata, content)
        
        return filtered_tags
    
    def _save_document_tags(self, document_id: str, file_name: str, tags: Dict[str, float],
                          metadata: Optional[Dict[str, Any]] = None,
                          content: str = "") -> None:
        """
        Save tags for a document
        
        Args:
            document_id: Document identifier
            file_name: Original file name
            tags: Dict mapping tag names to confidence scores
            metadata: Document metadata (optional)
        """
        try:
            # Prepare output file path
            safe_id = file_name.replace("/", "_").replace("\\", "_")
            output_file = os.path.join(self.output_dir, f"{safe_id}_tags.json")
            
            # Prepare data
            data = {
                "document_id": document_id,
                "file": file_name,  # Store the original filename
                "tags": tags,
                "metadata": metadata or {},
                "tagging_time": datetime.now().isoformat(),
                "auto_generated": True,
                "page_content": content
            }
            
            # Save to file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"Tags saved for document {document_id}")
        except Exception as e:
            logger.error(f"Error saving tags for document {document_id}: {e}")
    
    def process_batch(self, documents: List[Dict[str, Any]], 
                    min_confidence: float = 0.6) -> Dict[str, Any]:
        """
        Process a batch of documents
        
        Args:
            documents: List of document dicts with id, content, and metadata
            min_confidence: Minimum confidence threshold for tags
            
        Returns:
            Dict with batch processing stats
        """
        for doc in documents:
            doc_id = doc["id"]
            file_name = os.path.basename(doc_id) if os.path.sep in doc_id else doc_id
            
            self.process_document(
                doc_id, doc["content"], doc.get("metadata"),
                min_confidence, save_result=True
            )
        
        # Calculate average tags per document
        if self.stats["successfully_tagged"] > 0:
            self.stats["tags_per_document"] = self.stats["total_tags_applied"] / self.stats["successfully_tagged"]
        
        # Update end time
        self.stats["end_time"] = datetime.now().isoformat()
        
        return self.stats
    
    def export_batch_results(self, output_file: str, include_tag_counts: bool = True) -> bool:
        """
        Export batch processing results
        
        Args:
            output_file: Output file path
            include_tag_counts: Whether to include tag frequency counts
            
        Returns:
            bool: Success status
        """
        try:
            stats = self.stats.copy()
            
            # Add tag counts if requested
            if include_tag_counts:
                tag_counts = self._calculate_tag_counts()
                stats["tag_counts"] = tag_counts
            
            # Calculate average tags per document
            if stats["successfully_tagged"] > 0:
                stats["tags_per_document"] = stats["total_tags_applied"] / stats["successfully_tagged"]
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Save to file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Batch results exported to {output_file}")
            return True
        except Exception as e:
            logger.error(f"Error exporting batch results: {e}")
            return False
    
    def _calculate_tag_counts(self) -> Dict[str, int]:
        """
        Calculate tag frequency counts from processed documents
        
        Returns:
            Dict mapping tag names to occurrence counts
        """
        tag_counts = {}
        
        # Scan output directory for tag files
        for filename in os.listdir(self.output_dir):
            if filename.endswith("_tags.json"):
                try:
                    with open(os.path.join(self.output_dir, filename), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    # Count tags (handle both dict and list formats)
                    tags_data = data.get("tags", {})
                    if isinstance(tags_data, dict):
                        for tag in tags_data:
                            tag_counts[tag] = tag_counts.get(tag, 0) + 1
                    elif isinstance(tags_data, list):
                        for tag in tags_data:
                            tag_counts[tag] = tag_counts.get(tag, 0) + 1
                except Exception as e:
                    logger.warning(f"Error reading tag file {filename}: {e}")
        
        return tag_counts

class TagValidationTool:
    """
    Tool for validating and improving tag quality
    """
    def __init__(self, tag_schema: TagSchema, tag_analyzer: TagAnalyzer):
        """
        Initialize the tag validation tool
        
        Args:
            tag_schema: Tag schema object
            tag_analyzer: Tag analyzer object
        """
        self.tag_schema = tag_schema
        self.tag_analyzer = tag_analyzer
    
    def validate_document_tags(self, document_id: str, content: str, 
                             tags: Union[Dict[str, float], List[str]]) -> Dict[str, Any]:
        """
        Validate tags for a document
        
        Args:
            document_id: Document identifier
            content: Document text content
            tags: Dict mapping tag names to confidence scores or list of tag names
            
        Returns:
            Dict with validation results
        """
        validation_results = {
            "document_id": document_id,
            "valid_tags": [],
            "suspicious_tags": [],
            "missing_tags": [],
            "suggested_additions": [],
            "suggested_removals": []
        }
        
        # Convert tags to dictionary format if it's a list
        if isinstance(tags, list):
            tag_dict = {tag: 1.0 for tag in tags}
        else:
            tag_dict = tags
        
        # 1. Verify all tags exist in schema
        for tag_name in list(tag_dict.keys()):
            if tag_name not in self.tag_schema.tags:
                validation_results["suspicious_tags"].append({
                    "tag": tag_name,
                    "reason": "Tag not found in schema",
                    "confidence": tag_dict[tag_name]
                })
            else:
                validation_results["valid_tags"].append(tag_name)
        
        # 2. Check for potentially missing tags
        # Use tag analyzer to suggest related tags
        for tag_name in validation_results["valid_tags"]:
            related_tags = self.tag_analyzer.get_related_tags(tag_name)
            for related_tag, score in related_tags:
                if related_tag not in tag_dict and score > 0.5:
                    # Check if this tag appears in the content
                    tag_obj = self.tag_schema.get_tag(related_tag)
                    if tag_obj:
                        validation_results["missing_tags"].append({
                            "tag": related_tag,
                            "reason": f"Frequently co-occurs with {tag_name}",
                            "relatedness": score
                        })
        
        # 3. Suggest tags to add or remove
        # Potentially missing high-confidence tags
        for tag_info in validation_results["missing_tags"]:
            validation_results["suggested_additions"].append(tag_info["tag"])
        
        # Potentially incorrect tags (very low confidence or suspicious)
        for tag_info in validation_results["suspicious_tags"]:
            validation_results["suggested_removals"].append(tag_info["tag"])
        
        return validation_results
    
    def batch_validate_tags(self, tag_dir: str, output_file: str) -> Dict[str, Any]:
        """
        Validate tags for a batch of documents
        
        Args:
            tag_dir: Directory containing tag files
            output_file: Output file for validation report
            
        Returns:
            Dict with validation statistics
        """
        if not os.path.exists(tag_dir):
            logger.error(f"Tag directory not found: {tag_dir}")
            return {"error": "Directory not found"}
            
        stats = {
            "total_documents": 0,
            "documents_with_issues": 0,
            "total_valid_tags": 0,
            "total_suspicious_tags": 0,
            "total_missing_tags": 0,
            "documents": []
        }
        
        # Process each tag file
        for filename in os.listdir(tag_dir):
            if not filename.endswith("_tags.json"):
                continue
                
            try:
                # Debug: print filename being processed
                print(f"Processing tag file: {filename}")
                
                with open(os.path.join(tag_dir, filename), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Debug: Print structure of the tags field
                print(f"Tag structure: {type(data.get('tags', {}))} - Content: {data.get('tags', {})}")
                
                document_id = data.get("document_id")
                tags_data = data.get("tags", {})
                
                # Convert tags to proper format
                if isinstance(tags_data, list):
                    tags = {tag: 1.0 for tag in tags_data}
                elif isinstance(tags_data, dict):
                    tags = tags_data
                else:
                    logger.warning(f"Unexpected tag format in {filename}: {type(tags_data)}")
                    tags = {}
                
                # We need content to validate properly
                # If not available in tag file, use a simplified validation
                if "content" not in data:
                    # Simple validation - just check if tags exist in schema
                    validation = {
                        "document_id": document_id,
                        "valid_tags": [],
                        "suspicious_tags": []
                    }
                    
                    for tag_name in tags:
                        if tag_name in self.tag_schema.tags:
                            validation["valid_tags"].append(tag_name)
                        else:
                            validation["suspicious_tags"].append({
                                "tag": tag_name,
                                "reason": "Tag not found in schema",
                                "confidence": tags.get(tag_name, 1.0)
                            })
                    
                    # Add placeholder fields for compatibility
                    validation["missing_tags"] = []
                    validation["suggested_additions"] = []
                    validation["suggested_removals"] = []
                else:
                    # Full validation with content
                    validation = self.validate_document_tags(
                        document_id, data["content"], tags
                    )
                
                # Update statistics
                stats["total_documents"] += 1
                stats["total_valid_tags"] += len(validation["valid_tags"])
                stats["total_suspicious_tags"] += len(validation["suspicious_tags"])
                stats["total_missing_tags"] += len(validation.get("missing_tags", []))
                
                if validation["suspicious_tags"] or validation.get("missing_tags"):
                    stats["documents_with_issues"] += 1
                
                # Add document results
                stats["documents"].append({
                    "document_id": document_id,
                    "issues": len(validation["suspicious_tags"]) + len(validation.get("missing_tags", [])),
                    "valid_tags": validation["valid_tags"],
                    "suspicious_tags": validation["suspicious_tags"],
                    "missing_tags": validation.get("missing_tags", [])
                })
                
            except Exception as e:
                print(f"Error processing tag file {filename}: {e}")
                import traceback
                print(traceback.format_exc())
                logger.error(f"Error processing tag file {filename}: {e}")
        
        # Sort documents by number of issues
        stats["documents"].sort(key=lambda x: x["issues"], reverse=True)
        
        # Save report
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Validation report saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving validation report: {e}")
        
        return stats
    
    def export_tag_validation_report(self, validation_results: Dict[str, Any], 
                                   output_file: str, format: str = "csv") -> bool:
        """
        Export tag validation results to a file
        
        Args:
            validation_results: Validation results from batch_validate_tags
            output_file: Output file path
            format: Output format ("csv" or "json")
            
        Returns:
            bool: Success status
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            if format.lower() == "csv":
                # Export as CSV
                with open(output_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    
                    # Write header
                    writer.writerow([
                        "Document ID", "Valid Tags", "Suspicious Tags", 
                        "Missing Tags", "Issues"
                    ])
                    
                    # Write document rows
                    for doc in validation_results["documents"]:
                        suspicious_tags = []
                        for tag_info in doc["suspicious_tags"]:
                            if isinstance(tag_info, dict) and "tag" in tag_info:
                                suspicious_tags.append(tag_info["tag"])
                            elif isinstance(tag_info, str):
                                suspicious_tags.append(tag_info)
                                
                        missing_tags = []
                        for tag_info in doc.get("missing_tags", []):
                            if isinstance(tag_info, dict) and "tag" in tag_info:
                                missing_tags.append(tag_info["tag"])
                            elif isinstance(tag_info, str):
                                missing_tags.append(tag_info)
                                
                        writer.writerow([
                            doc["document_id"],
                            ", ".join(doc["valid_tags"]),
                            ", ".join(suspicious_tags),
                            ", ".join(missing_tags),
                            doc["issues"]
                        ])
            else:
                # Export as JSON
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(validation_results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Validation report exported to {output_file}")
            return True
        except Exception as e:
            logger.error(f"Error exporting validation report: {e}")
            return False