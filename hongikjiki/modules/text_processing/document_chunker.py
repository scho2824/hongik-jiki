import hashlib
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

def get_file_hash(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def save_processed_file_metadata(documents: List[Dict[str, Any]], output_path: str = "data/processed_files.json") -> None:
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            processed_data = json.load(f)
    else:
        processed_data = {}

    for doc in documents:
        content = doc["content"]
        metadata = doc["metadata"]
        file_path = metadata.get("source", "unknown.txt")
        file_hash = metadata.get("file_hash", get_file_hash(content))
        chunks_count = metadata.get("total_chunks", 1)

        processed_data[file_path] = {
            "hash": file_hash,
            "processed_time": datetime.now().isoformat(),
            "chunks_count": chunks_count,
            "vector_ids": []  # Placeholder to be updated after vectorization
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

import re
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger("HongikJikiChatBot")

class DocumentChunker:
    """
    문서를 적절한 크기의 청크로 분할하는 클래스
    검색 및 처리를 위한 청킹 로직 구현
    
    개선된 버전:
    - 문맥 보존 강화
    - 의미 단위 기반 분할
    - 중첩 청크 생성으로 검색 성능 향상
    """
    
    def __init__(self, chunk_size: int = 800, overlap: int = 200):
        """
        DocumentChunker 초기화
        
        Args:
            chunk_size: 기본 청크 크기 (문자 수)
            overlap: 기본 중복 영역 크기 (문자 수)
        """
        self.default_chunk_size = chunk_size
        self.default_overlap = overlap
        
        # 문장 구분 패턴 (한국어 종결어미 포함)
        self.sentence_pattern = re.compile(r'([.!?][\s\n]+|[.!?]$|다\.[\s\n]+|다\.$|까\?[\s\n]+|까\?$|니다\.[\s\n]+|니다\.$)')
        
        # 문단 구분 패턴
        self.paragraph_pattern = re.compile(r'\n\s*\n')
        
        # 강의 제목 및 구분자 패턴
        self.title_pattern = re.compile(r'(^|\n)(\[.*?\]|제목:|강의명:|\d+강:|\d+\s*강\s*:)')
    
    def split_documents(self, documents: List[Dict[str, Any]],
                        chunk_size: Optional[int] = None,
                        overlap: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        문서를 청크로 분할하고 메타데이터 유지
        
        Args:
            documents: 문서 딕셔너리 리스트
            chunk_size: 각 청크의 최대 문자 수 (기본값 사용 시 None)
            overlap: 연속된 청크 간의 중복 문자 수 (기본값 사용 시 None)
                
        Returns:
            List[Dict]: 분할된 청크와 메타데이터
        """
        chunk_size = chunk_size or self.default_chunk_size
        overlap = overlap or self.default_overlap
        
        chunks = []

        for doc in documents:
            try:
                content = doc["content"]
                metadata = doc["metadata"]
                # 문서 구조 분석: 항상 먼저 한 번만
                doc_structure = DocumentStructureAnalyzer().analyze_document_structure(content)

                # Fallback: single-paragraph long document -> simple fixed-size splits
                if not doc_structure["has_clear_sections"] and len(doc_structure["paragraphs"]) <= 1 and len(content) > chunk_size:
                    simple_chunks = self.chunk_text(content, chunk_size)
                    for idx, txt in enumerate(simple_chunks):
                        chunk_metadata = metadata.copy()
                        chunk_metadata["chunk_index"] = idx
                        chunk_metadata["chunk_info"] = f"기본 청크 {idx+1} (단순 분할)"
                        chunk_metadata["source_id"] = f"{metadata.get('file_hash','doc')}_chunk_{idx}"
                        chunk_metadata["chunk"] = idx
                        chunks.append({"content": txt, "metadata": chunk_metadata})
                    continue

                # 짧은 콘텐츠는 분할하지 않음
                if len(content) < chunk_size:
                    chunk_metadata = metadata.copy()
                    chunk_metadata["chunk_index"] = 0
                    chunk_metadata["chunk_info"] = f"전체 문서 (1/1)"
                    chunk_metadata["is_short_document"] = True
                    # Assign source_id for short document
                    chunk_metadata["source_id"] = f"{metadata.get('file_hash', 'doc')}_chunk_0"
                    chunk_metadata["chunk"] = 0
                    chunks.append({
                        "content": content,
                        "metadata": chunk_metadata
                    })
                    continue

                # 의미 단위 기반 분할
                semantic_chunks = self._split_by_semantic_units(content, metadata, chunk_size, overlap, doc_structure)
                for i, chunk in enumerate(semantic_chunks):
                    chunk["metadata"]["chunk"] = chunk["metadata"].get("chunk_index", i)
                chunks.extend(semantic_chunks)

                # 중첩 청크 생성 (검색 성능 향상)
                if len(semantic_chunks) > 1:
                    overlap_chunks = self._create_overlapping_chunks(semantic_chunks, metadata)
                    for i, chunk in enumerate(overlap_chunks):
                        chunk["metadata"]["chunk"] = chunk["metadata"].get("chunk_index", len(semantic_chunks) + i)
                    chunks.extend(overlap_chunks)
            except KeyError as e:
                logger.warning(f"[KeyError] Missing key in document: {e}")
                continue
            except ValueError as e:
                logger.warning(f"[ValueError] Invalid value encountered: {e}")
                continue
            except Exception as e:
                logger.warning(f"[Exception] General error processing document: {e}")
                continue

        # 청크 정보 업데이트 및 source_id 할당
        for i, chunk in enumerate(chunks):
            chunk["metadata"]["total_chunks"] = len(chunks)
            if "source_id" not in chunk["metadata"]:
                chunk["metadata"]["source_id"] = f"{chunk['metadata'].get('file_hash', 'doc')}_chunk_{chunk['metadata'].get('chunk_index', i)}"
        return chunks
    
    def _split_by_semantic_units(self, content: str, metadata: Dict[str, Any], 
                               chunk_size: int, overlap: int, 
                               doc_structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        의미 단위 기반 문서 분할
        
        Args:
            content: 문서 내용
            metadata: 문서 메타데이터
            chunk_size: 청크 크기
            overlap: 중복 크기
            doc_structure: 문서 구조 정보
            
        Returns:
            List[Dict]: 의미 단위로 분할된 청크
        """
        chunks = []
        paragraphs = doc_structure["paragraphs"]
        
        # 섹션이 명확한 경우 섹션 기반 분할
        if doc_structure["has_clear_sections"] and len(doc_structure["titles"]) > 1:
            section_chunks = self._split_by_sections(paragraphs, metadata, chunk_size, overlap, doc_structure["titles"])
            # Assign source_id for each chunk in section_chunks
            for idx, chunk in enumerate(section_chunks):
                chunk["metadata"]["source_id"] = f"{metadata.get('file_hash', 'doc')}_chunk_{chunk['metadata'].get('chunk_index', idx)}"
            # Convert any 'chunk' keys to 'content' keys
            for chunk in section_chunks:
                if "chunk" in chunk:
                    chunk["content"] = chunk["chunk"]
                    del chunk["chunk"]
            chunks.extend(section_chunks)
        else:
            # 일반 문서는 문단 및 문장 기반 분할
            current_chunk = ""
            chunk_paragraphs = []
            
            for i, para in enumerate(paragraphs):
                # 현재 문단 추가 시 청크 크기 초과 여부 확인
                para_len = len(para) if para is not None else 0
                if len(current_chunk) + para_len + 2 <= chunk_size:  # +2 for newlines
                    current_chunk += para + "\n\n"
                    chunk_paragraphs.append(i)
                else:
                    # 현재 청크가 있으면 추가
                    if current_chunk:
                        chunk_metadata = metadata.copy()
                        chunk_metadata["chunk_index"] = len(chunks)
                        chunk_metadata["chunk_info"] = f"청크 {len(chunks) + 1} (문단 {', '.join(map(str, [p+1 for p in chunk_paragraphs]))})"
                        chunk_metadata["paragraph_indices"] = chunk_paragraphs
                        chunk_metadata["source_id"] = f"{metadata.get('file_hash', 'doc')}_chunk_{chunk_metadata['chunk_index']}"
                        chunks.append({
                            "content": current_chunk.strip(),
                            "metadata": chunk_metadata
                        })
                    
                    # 중복 영역 처리
                    if overlap > 0 and chunk_paragraphs:
                        # 마지막 몇 개 문단을 중복 포함
                        overlap_size = 0
                        overlap_paragraphs = []
                        
                        for p_idx in reversed(chunk_paragraphs):
                            if p_idx < len(paragraphs):
                                p_content = paragraphs[p_idx]
                                if p_content is not None:
                                    content_length = len(p_content)
                                    if overlap_size + content_length <= overlap:
                                        overlap_paragraphs.insert(0, p_idx)
                                        overlap_size += content_length
                                    else:
                                        break
                        
                        # 중복 영역 포함하여 새 청크 시작
                        current_chunk = ""
                        chunk_paragraphs = []
                        
                        for p_idx in overlap_paragraphs:
                            if p_idx < len(paragraphs):
                                p_content = paragraphs[p_idx]
                                if p_content is not None:
                                    current_chunk += p_content + "\n\n"
                                    chunk_paragraphs.append(p_idx)
                    else:
                        current_chunk = ""
                        chunk_paragraphs = []
                    
                    # 현재 문단 추가
                    if para is not None:
                        current_chunk += para + "\n\n"
                        chunk_paragraphs.append(i)
            
            # 마지막 청크 추가
            if current_chunk:
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = len(chunks)
                chunk_metadata["chunk_info"] = f"청크 {len(chunks) + 1} (문단 {', '.join(map(str, [p+1 for p in chunk_paragraphs]))})"
                chunk_metadata["paragraph_indices"] = chunk_paragraphs
                chunk_metadata["source_id"] = f"{metadata.get('file_hash', 'doc')}_chunk_{chunk_metadata['chunk_index']}"
                chunks.append({
                    "content": current_chunk.strip(),
                    "metadata": chunk_metadata
                })
        
        return chunks
    
    def _split_by_sections(self, paragraphs: List[str], metadata: Dict[str, Any],
                         chunk_size: int, overlap: int, 
                         titles: List[Tuple[int, str]]) -> List[Dict[str, Any]]:
        """
        섹션 기반 문서 분할
        
        Args:
            paragraphs: 문단 리스트
            metadata: 문서 메타데이터
            chunk_size: 청크 크기
            overlap: 중복 크기
            titles: 제목 정보 (인덱스, 제목)
            
        Returns:
            List[Dict]: 섹션 단위로 분할된 청크
        """
        chunks = []
        
        # 섹션 경계 계산
        section_boundaries = []
        for i, (idx, title) in enumerate(titles):
            if i < len(titles) - 1:
                section_boundaries.append((idx, titles[i+1][0] - 1))
            else:
                section_boundaries.append((idx, len(paragraphs) - 1))
        
        # 각 섹션별 처리
        for i, (start_idx, end_idx) in enumerate(section_boundaries):
            section_title = titles[i][1] if i < len(titles) else ""
            
            # 범위 확인 및 유효한 범위로 조정
            start_idx = max(0, min(start_idx, len(paragraphs) - 1))
            end_idx = max(start_idx, min(end_idx, len(paragraphs) - 1))
            
            section_paragraphs = paragraphs[start_idx:end_idx+1]
            
            # 섹션 내용 합치기 - None 체크 추가
            valid_paragraphs = [p for p in section_paragraphs if p is not None]
            section_content = "\n\n".join(valid_paragraphs)
            
            # 섹션이 청크 크기보다 작으면 바로 추가
            if len(section_content) <= chunk_size:
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = len(chunks)
                chunk_metadata["chunk_info"] = f"섹션: {section_title.strip() if section_title else '무제'}"
                chunk_metadata["section_title"] = section_title.strip() if section_title else "무제 섹션"
                chunk_metadata["paragraph_indices"] = list(range(start_idx, end_idx+1))
                chunk_metadata["source_id"] = f"{metadata.get('file_hash', 'doc')}_chunk_{chunk_metadata['chunk_index']}"
                chunks.append({
                    "content": section_content,
                    "metadata": chunk_metadata
                })
            else:
                # 섹션이 크면 문장 단위로 분할
                section_sentences = self._split_text_to_sentences(section_content)
                current_chunk = ""
                current_sentences = []
                
                for j, sentence in enumerate(section_sentences):
                    # None 체크 추가
                    if sentence is None:
                        continue
                        
                    sentence_len = len(sentence)
                    # 현재 문장 추가 시 청크 크기 초과 여부 확인
                    if len(current_chunk) + sentence_len <= chunk_size:
                        current_chunk += sentence
                        current_sentences.append(j)
                    else:
                        # 현재 청크 추가
                        if current_chunk:
                            chunk_metadata = metadata.copy()
                            chunk_metadata["chunk_index"] = len(chunks)
                            chunk_metadata["chunk_info"] = f"섹션: {section_title.strip() if section_title else '무제'} (부분 {len(chunks) + 1})"
                            chunk_metadata["section_title"] = section_title.strip() if section_title else "무제 섹션"
                            chunk_metadata["paragraph_indices"] = list(range(start_idx, end_idx+1))
                            chunk_metadata["is_section_part"] = True
                            chunk_metadata["source_id"] = f"{metadata.get('file_hash', 'doc')}_chunk_{chunk_metadata['chunk_index']}"
                            chunks.append({
                                "content": current_chunk,
                                "metadata": chunk_metadata
                            })
                        
                        # 중복 영역 처리
                        if overlap > 0 and current_sentences:
                            # 마지막 몇 개 문장을 중복 포함
                            overlap_size = 0
                            overlap_sentences = []
                            
                            for s_idx in reversed(current_sentences):
                                if s_idx < len(section_sentences):
                                    s_content = section_sentences[s_idx]
                                    if s_content is not None:
                                        content_length = len(s_content)
                                        if overlap_size + content_length <= overlap:
                                            overlap_sentences.insert(0, s_idx)
                                            overlap_size += content_length
                                        else:
                                            break
                            
                            # 중복 영역 포함하여 새 청크 시작
                            current_chunk = ""
                            current_sentences = []
                            
                            for s_idx in overlap_sentences:
                                if s_idx < len(section_sentences):
                                    s_content = section_sentences[s_idx]
                                    if s_content is not None:
                                        current_chunk += s_content
                                        current_sentences.append(s_idx)
                        else:
                            current_chunk = ""
                            current_sentences = []
                        
                        # 현재 문장 추가
                        if sentence is not None:
                            current_chunk += sentence
                            current_sentences.append(j)
                
                # 마지막 청크 추가
                if current_chunk:
                    chunk_metadata = metadata.copy()
                    chunk_metadata["chunk_index"] = len(chunks)
                    chunk_metadata["chunk_info"] = f"섹션: {section_title.strip() if section_title else '무제'} (부분 {len(chunks) + 1})"
                    chunk_metadata["section_title"] = section_title.strip() if section_title else "무제 섹션"
                    chunk_metadata["paragraph_indices"] = list(range(start_idx, end_idx+1))
                    chunk_metadata["is_section_part"] = True
                    chunk_metadata["source_id"] = f"{metadata.get('file_hash', 'doc')}_chunk_{chunk_metadata['chunk_index']}"
                    chunks.append({
                        "content": current_chunk,
                        "metadata": chunk_metadata
                    })
        
        return chunks
    
    def _split_text_to_sentences(self, text: str) -> List[str]:
        """
        텍스트를 문장 단위로 분할
        
        Args:
            text: 분할할 텍스트
            
        Returns:
            List[str]: 문장 리스트
        """
        if text is None:
            return []
            
        # 문장 구분
        sentences = self.sentence_pattern.split(text)
        
        # 문장 패턴과 실제 문장 합치기
        merged_sentences = []
        for i in range(0, len(sentences), 2):
            if i+1 < len(sentences):
                merged_sentences.append(sentences[i] + sentences[i+1])
            else:
                merged_sentences.append(sentences[i])
        
        return merged_sentences
    
    def _create_overlapping_chunks(self, chunks: List[Dict[str, Any]], 
                                metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        중첩 청크 생성 (검색 성능 향상)
        
        Args:
            chunks: 기본 청크 리스트
            metadata: 원본 메타데이터
            
        Returns:
            List[Dict]: 중첩 청크 리스트
        """
        overlap_chunks = []
        
        # 최소 3개 이상의 청크가 있을 때만 중첩 청크 생성
        if len(chunks) < 3:
            return overlap_chunks
        
        # 인접한 두 청크를 합쳐서 중첩 청크 생성
        for i in range(len(chunks) - 1):
            chunk1_content = chunks[i].get("content")
            chunk2_content = chunks[i+1].get("content")
            
            # None 체크 추가
            if chunk1_content is None or chunk2_content is None:
                continue
                
            # 두 청크 합치기
            combined_content = chunk1_content + "\n\n" + chunk2_content
            
            # 중첩 청크 메타데이터
            overlap_metadata = metadata.copy()
            
            # 안전하게 인덱스 추출
            chunk_index = chunks[i]["metadata"].get("chunk_index", i) if "metadata" in chunks[i] else i
            next_chunk_index = chunks[i+1]["metadata"].get("chunk_index", i+1) if "metadata" in chunks[i+1] else i+1
            
            overlap_metadata["chunk_index"] = len(chunks) + len(overlap_chunks)
            overlap_metadata["chunk_info"] = f"중첩 청크 {chunk_index+1}-{next_chunk_index+1}"
            overlap_metadata["is_overlap_chunk"] = True
            overlap_metadata["original_chunks"] = [chunk_index, next_chunk_index]
            overlap_metadata["source_id"] = f"{metadata.get('file_hash', 'doc')}_overlap_{chunk_index}_{next_chunk_index}"
            
            # 중첩 청크 추가
            overlap_chunks.append({
                "content": combined_content,
                "metadata": overlap_metadata
            })
        
        return overlap_chunks
        
    def chunk_text(self, text: str, size: Optional[int] = None) -> List[str]:
        """
        단일 텍스트를 기본 청크 크기 기준으로 단순 분할 (간이용)
        태깅 또는 테스트용 간단한 청크 생성
        Args:
            text: 분할할 텍스트
            size: 청크 크기 (기본값은 self.default_chunk_size)
        Returns:
            List[str]: 분할된 청크 리스트
        """
        if text is None:
            return []
            
        chunk_size = size or self.default_chunk_size
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size].strip()
            if chunk:
                chunks.append(chunk)
        return chunks


# 문서 구조 분석기 클래스
class DocumentStructureAnalyzer:
    def __init__(self, sentence_pattern=None, paragraph_pattern=None, title_pattern=None):
        self.sentence_pattern = sentence_pattern or re.compile(r'([.!?][\s\n]+|[.!?]$|다\.[\s\n]+|다\.$|까\?[\s\n]+|까\?$|니다\.[\s\n]+|니다\.$)')
        self.paragraph_pattern = paragraph_pattern or re.compile(r'\n\s*\n')
        self.title_pattern = title_pattern or re.compile(r'(^|\n)(\[.*?\]|제목:|강의명:|\d+강:|\d+\s*강\s*:)')

    def analyze_document_structure(self, content: str) -> Dict[str, Any]:
        if content is None:
            return {
                "paragraphs": [],
                "paragraph_count": 0,
                "titles": [],
                "sentences": [],
                "sentence_count": 0,
                "has_clear_sections": False
            }
            
        paragraphs = self.paragraph_pattern.split(content)
        paragraphs = [p.strip() for p in paragraphs if p and p.strip()]
        titles = [(i, p) for i, p in enumerate(paragraphs) if self.title_pattern.search(p)]
        
        sentences = self.sentence_pattern.split(content)
        merged_sentences = []
        for i in range(0, len(sentences), 2):
            if i+1 < len(sentences):
                merged_sentences.append(sentences[i] + sentences[i+1])
            else:
                merged_sentences.append(sentences[i])
                
        return {
            "paragraphs": paragraphs,
            "paragraph_count": len(paragraphs),
            "titles": titles,
            "sentences": merged_sentences,
            "sentence_count": len(merged_sentences),
            "has_clear_sections": len(titles) > 1
        }