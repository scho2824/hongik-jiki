"""
Tag Schema for Hongik-Jiki Chatbot

Defines the hierarchical tag structure for categorizing Jungbub teachings.
"""
import os
import yaml
from typing import Dict, List, Optional, Set, Tuple, Any
import logging

logger = logging.getLogger("HongikJikiChatBot")

class Tag:
    """
    Represents a single tag in the taxonomy
    """
    def __init__(self, name: str, category: str, parent: Optional[str] = None,
                 emoji: Optional[str] = None, description: Optional[str] = None,
                 keywords: Optional[List[str]] = None, phrases: Optional[List[str]] = None,
                 patterns: Optional[List[str]] = None):
        self.name = name
        self.category = category
        self.parent = parent
        self.emoji = emoji
        self.description = description
        self.keywords = keywords or []
        self.phrases = phrases or []
        self.patterns = patterns or []
        self.children = []  # Will be populated by TagManager
        self.related_tags = []  # Will be populated by TagManager

    def to_dict(self) -> Dict[str, Any]:
        """Convert tag to dictionary representation"""
        return {
            "name": self.name,
            "category": self.category,
            "parent": self.parent,
            "emoji": self.emoji,
            "description": self.description,
            "keywords": self.keywords,
            "phrases": self.phrases,
            "patterns": self.patterns,
            "children": [child.name for child in self.children],
            "related_tags": self.related_tags
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Tag':
        """Create tag from dictionary"""
        tag = cls(
            name=data["name"],
            category=data["category"],
            parent=data.get("parent"),
            emoji=data.get("emoji"),
            description=data.get("description"),
            keywords=data.get("keywords", []),
            phrases=data.get("phrases", []),
            patterns=data.get("patterns", [])
        )
        tag.related_tags = data.get("related_tags", [])
        # Children will be populated by TagManager
        return tag

class TagSchema:
    """
    Manages the hierarchical tag structure for Jungbub teachings
    """
    def __init__(self, schema_file: Optional[str] = None):
        """
        Initialize the tag schema
        
        Args:
            schema_file: Path to YAML schema definition (optional)
        """
        self.tags = {}  # name -> Tag
        self.categories = {}  # category -> emoji
        
        # Load schema from file if provided
        if schema_file and os.path.exists(schema_file):
            self.load_schema(schema_file)
        else:
            self._init_default_schema()
            
    def _init_default_schema(self):
        """Initialize the default tag taxonomy for Jungbub teachings"""
        # Define categories with emojis
        self.categories = {
            "우주와 진리": "🌌",
            "인간 본성과 삶": "🧍",
            "탐구와 인식": "🔍",
            "실천과 방법": "🧘",
            "사회와 현실": "🌏", 
            "감정 상태": "❤️",
            "삶의 단계": "🕰️"
        }
        
        # 1. 우주와 진리 (Universe & Truth)
        universe_tags = [
            Tag("정법", "우주와 진리"),
            Tag("우주법칙", "우주와 진리"),
            Tag("진리", "우주와 진리"),
            Tag("자연의 섭리", "우주와 진리"),
            Tag("창조와 존재", "우주와 진리"),
            Tag("시간과 공간", "우주와 진리")
        ]
        
        # 2. 인간 본성과 삶 (Human Nature & Life)
        # Parent tags for subcategories
        human_nature_parent_tags = [
            Tag("본성", "인간 본성과 삶"),
            Tag("존재와 죽음", "인간 본성과 삶"),
            Tag("자유와 선택", "인간 본성과 삶")
        ]
        
        # Child tags under '본성'
        nature_child_tags = [
            Tag("인간의 본성", "인간 본성과 삶", parent="본성"),
            Tag("선과 악", "인간 본성과 삶", parent="본성"),
            Tag("욕심과 버림", "인간 본성과 삶", parent="본성"),
            Tag("사랑과 집착", "인간 본성과 삶", parent="본성")
        ]
        
        # Child tags under '존재와 죽음'
        existence_child_tags = [
            Tag("죽음과 삶", "인간 본성과 삶", parent="존재와 죽음"),
            Tag("윤회와 전생", "인간 본성과 삶", parent="존재와 죽음"),
            Tag("존재의 의미", "인간 본성과 삶", parent="존재와 죽음")
        ]
        
        # Child tags under '자유와 선택'
        freedom_child_tags = [
            Tag("자유의지", "인간 본성과 삶", parent="자유와 선택",
                description="인간의 선택과 결정 능력",
                keywords=["자유의지", "결정", "선택", "의지", "자기결정", "스스로 결정"],
                phrases=["선택할 자유", "스스로 결정하는 힘", "인간의 의지", "자신의 삶을 선택한다"]
            ),
            Tag("책임과 결과", "인간 본성과 삶", parent="자유와 선택",
                description="선택과 행위에 따른 결과와 책임",
                keywords=["책임", "결과", "선택의 대가", "행동의 결과", "인과"],
                phrases=["선택에는 책임이 따른다", "행동의 결과를 받아들이다", "인과의 법칙"]
            )
        ]
        
        # 3. 탐구와 인식 (Inquiry & Awareness)
        inquiry_tags = [
            Tag("자기성찰", "탐구와 인식"),
            Tag("의식 성장", "탐구와 인식"),
            Tag("깨달음", "탐구와 인식"),
            Tag("자각과 각성", "탐구와 인식"),
            Tag("믿음과 앎", "탐구와 인식"),
            Tag("무의식과 업", "탐구와 인식")
        ]
        
        # 4. 실천과 방법 (Practice & Method)
        practice_tags = [
            Tag("수행", "실천과 방법"),
            Tag("행공", "실천과 방법"),
            Tag("기도와 명상", "실천과 방법"),
            Tag("생활 실천", "실천과 방법"),
            Tag("정법 공부법", "실천과 방법")
        ]
        
        # 5. 사회와 현실 (Society & Reality)
        # Parent tags for subcategories
        society_parent_tags = [
            Tag("관계", "사회와 현실"),
            Tag("공동체", "사회와 현실"),
            Tag("구조와 제도", "사회와 현실")
        ]
        
        # Child tags under '관계'
        relationship_child_tags = [
            Tag("인간관계", "사회와 현실", parent="관계"),
            Tag("갈등 해결", "사회와 현실", parent="관계"),
            Tag("공감과 배려", "사회와 현실", parent="관계")
        ]
        
        # Child tags under '공동체'
        community_child_tags = [
            Tag("가족과 공동체", "사회와 현실", parent="공동체"),
            Tag("부모-자녀 관계", "사회와 현실", parent="공동체"),
            Tag("부부 관계", "사회와 현실", parent="공동체"),
            Tag("세대 간 갈등", "사회와 현실", parent="공동체")
        ]
        
        # Child tags under '구조와 제도'
        institution_child_tags = [
            Tag("정치", "사회와 현실", parent="구조와 제도"),
            Tag("경제", "사회와 현실", parent="구조와 제도"),
            Tag("리더십", "사회와 현실", parent="구조와 제도"),
            Tag("청년과 교육", "사회와 현실", parent="구조와 제도"),
            Tag("국가와 민족", "사회와 현실", parent="구조와 제도"),
            Tag("변화와 위기", "사회와 현실", parent="구조와 제도")
        ]
        
        # 6. 감정 상태 (Emotional States)
        emotion_tags = [
            Tag("불안", "감정 상태"),
            Tag("분노", "감정 상태"),
            Tag("슬픔", "감정 상태"),
            Tag("외로움", "감정 상태"),
            Tag("무기력", "감정 상태"),
            Tag("후회", "감정 상태"),
            Tag("희망", "감정 상태"),
            Tag("평온", "감정 상태"),
            Tag("기쁨", "감정 상태"),
            Tag("사랑", "감정 상태")
        ]
        
        # 7. 삶의 단계 (Life Stages)
        life_stage_tags = [
            Tag("유년기", "삶의 단계"),
            Tag("청소년기", "삶의 단계"),
            Tag("청년기", "삶의 단계"),
            Tag("중년의 위기", "삶의 단계"),
            Tag("가족 형성기", "삶의 단계"),
            Tag("노년의 지혜", "삶의 단계"),
            Tag("죽음을 준비하는 삶", "삶의 단계")
        ]
        
        # Combine all tags
        all_tags = (
            universe_tags + 
            human_nature_parent_tags + nature_child_tags + existence_child_tags + freedom_child_tags +
            inquiry_tags + 
            practice_tags + 
            society_parent_tags + relationship_child_tags + community_child_tags + institution_child_tags +
            emotion_tags + 
            life_stage_tags
        )
        
        # Add all tags to dictionary
        for tag in all_tags:
            self.tags[tag.name] = tag
            
        # Establish parent-child relationships
        self._build_relationships()
        
    def _build_relationships(self):
        """Build parent-child relationships between tags"""
        # Add children to parent tags
        for tag_name, tag in self.tags.items():
            if tag.parent and tag.parent in self.tags:
                parent_tag = self.tags[tag.parent]
                if tag not in parent_tag.children:
                    parent_tag.children.append(tag)
    
    def add_tag(self, tag: Tag) -> None:
        """
        Add a new tag to the schema
        
        Args:
            tag: Tag object to add
        """
        self.tags[tag.name] = tag
        
        # Update parent-child relationship
        if tag.parent and tag.parent in self.tags:
            parent_tag = self.tags[tag.parent]
            if tag not in parent_tag.children:
                parent_tag.children.append(tag)
                
    def get_tag(self, name: str) -> Optional[Tag]:
        """
        Get a tag by name
        
        Args:
            name: Tag name
            
        Returns:
            Tag object or None if not found
        """
        return self.tags.get(name)
    
    def get_all_tags(self) -> List[Tag]:
        """
        Get all tags in the schema
        
        Returns:
            List of all Tag objects
        """
        return list(self.tags.values())
    
    def get_tags_by_category(self, category: str) -> List[Tag]:
        """
        Get all tags in a specific category
        
        Args:
            category: Category name
            
        Returns:
            List of Tag objects in the category
        """
        return [tag for tag in self.tags.values() if tag.category == category]
    
    def get_child_tags(self, parent_name: str) -> List[Tag]:
        """
        Get all child tags of a parent tag
        
        Args:
            parent_name: Name of the parent tag
            
        Returns:
            List of child Tag objects
        """
        parent = self.get_tag(parent_name)
        if parent:
            return parent.children
        return []
    
    def save_schema(self, file_path: str) -> None:
        """
        Save the tag schema to a YAML file
        
        Args:
            file_path: Output file path
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Prepare data structure
        categories_data = self.categories
        
        tags_data = {}
        for name, tag in self.tags.items():
            tags_data[name] = {
                "category": tag.category,
                "parent": tag.parent,
                "emoji": tag.emoji,
                "description": tag.description,
                "related_tags": tag.related_tags,
                "keywords": tag.keywords,
                "phrases": tag.phrases,
                "patterns": tag.patterns
            }
        
        schema_data = {
            "categories": categories_data,
            "tags": tags_data
        }
        
        # Write to YAML file
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(schema_data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
            logger.info(f"Tag schema saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving tag schema to {file_path}: {e}")
    
    def load_schema(self, file_path: str) -> None:
        """
        Load tag schema from a YAML file
        
        Args:
            file_path: Path to schema file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                schema_data = yaml.safe_load(f)
                
            # Load categories
            self.categories = schema_data.get("categories", {})
            
            # Clear existing tags
            self.tags = {}
            
            # First pass: create all tags
            for name, tag_data in schema_data.get("tags", {}).items():
                tag = Tag(
                    name=name, 
                    category=tag_data["category"],
                    parent=tag_data.get("parent"),
                    emoji=tag_data.get("emoji"),
                    description=tag_data.get("description"),
                    keywords=tag_data.get("keywords", []),
                    phrases=tag_data.get("phrases", []),
                    patterns=tag_data.get("patterns", [])
                )
                self.tags[name] = tag
                
            # Second pass: set related tags and build relationships
            for name, tag_data in schema_data.get("tags", {}).items():
                tag = self.tags[name]
                tag.related_tags = tag_data.get("related_tags", [])
            
            # Build parent-child relationships
            self._build_relationships()
            
            logger.info(f"Tag schema loaded from {file_path}")
        except Exception as e:
            logger.error(f"Error loading tag schema from {file_path}: {e}")
            # Fall back to default schema
            self._init_default_schema()

    @classmethod
    def load_from_file(cls, path: str) -> "TagSchema":
        """
        Create and return a TagSchema instance loaded from a YAML file.

        Args:
            path: Path to the schema file

        Returns:
            TagSchema instance
        """
        instance = cls()
        instance.load_schema(path)
        return instance

    # Alias for backward compatibility
    load_from_yaml = load_from_file  # alias for backward compatibility