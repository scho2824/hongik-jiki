import hashlib
import json
import os
from datetime import datetime
from typing import List, Dict, Any
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
from typing import List, Dict, Any, Tuple

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
                        chunk_size: int = None, 
                        overlap: int = None) -> List[Dict[str, Any]]:
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
                doc_structure = self._analyze_document_structure(content)

                # Fallback: single-paragraph long document -> simple fixed-size splits
                if not doc_structure["has_clear_sections"] and len(doc_structure["paragraphs"]) <= 1 and len(content) > chunk_size:
                    simple_chunks = self.chunk_text(content, chunk_size)
                    for idx, txt in enumerate(simple_chunks):
                        chunk_metadata = metadata.copy()
                        chunk_metadata["chunk_index"] = idx
                        chunk_metadata["chunk_info"] = f"기본 청크 {idx+1} (단순 분할)"
                        chunk_metadata["source_id"] = f"{metadata.get('file_hash','doc')}_chunk_{idx}"
                        chunk_metadata["chunk"] = idx
                        chunks.append({"chunk": txt, "metadata": chunk_metadata})
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
                        "chunk": content,
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
            except Exception as e:
                logger.warning(f"Error processing document {doc.get('metadata', {}).get('source_id', 'unknown')}: {e}")
                continue

        # 청크 정보 업데이트 및 source_id 할당
        for i, chunk in enumerate(chunks):
            chunks[i]["metadata"]["total_chunks"] = len(chunks)
            # Remove chunk_id if present, ensure source_id is used
            # chunks[i]["metadata"]["chunk_id"] = f"{metadata.get('file_hash', 'doc')}_{i}"
            if "source_id" not in chunks[i]["metadata"]:
                chunks[i]["metadata"]["source_id"] = f"{chunks[i]['metadata'].get('file_hash', 'doc')}_chunk_{chunks[i]['metadata'].get('chunk_index', i)}"
            # Preserve content alias for backward compatibility
            chunks[i]["content"] = chunks[i]["chunk"]
        return chunks
    
    def _analyze_document_structure(self, content: str) -> Dict[str, Any]:
        """
        문서 구조 분석
        
        Args:
            content: 문서 내용
            
        Returns:
            Dict: 문서 구조 정보
        """
        # 문단 구분
        paragraphs = self.paragraph_pattern.split(content)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # 제목 및 섹션 식별
        titles = []
        for i, para in enumerate(paragraphs):
            if self.title_pattern.search(para):
                titles.append((i, para))
        
        # 문장 구분
        sentences = self.sentence_pattern.split(content)
        # 문장 패턴과 실제 문장 합치기
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
            # Convert any 'content' keys to 'chunk' keys
            for chunk in section_chunks:
                if "content" in chunk:
                    chunk["chunk"] = chunk["content"]
                    del chunk["content"]
            chunks.extend(section_chunks)
        else:
            # 일반 문서는 문단 및 문장 기반 분할
            current_chunk = ""
            chunk_paragraphs = []
            
            for i, para in enumerate(paragraphs):
                # 현재 문단 추가 시 청크 크기 초과 여부 확인
                if len(current_chunk) + len(para) + 2 <= chunk_size:  # +2 for newlines
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
                            "chunk": current_chunk.strip(),
                            "metadata": chunk_metadata
                        })
                    
                    # 중복 영역 처리
                    if overlap > 0 and chunk_paragraphs:
                        # 마지막 몇 개 문단을 중복 포함
                        overlap_size = 0
                        overlap_paragraphs = []
                        
                        for p_idx in reversed(chunk_paragraphs):
                            p_content = paragraphs[p_idx]
                            if overlap_size + len(p_content) <= overlap:
                                overlap_paragraphs.insert(0, p_idx)
                                overlap_size += len(p_content)
                            else:
                                break
                        
                        # 중복 영역 포함하여 새 청크 시작
                        current_chunk = ""
                        chunk_paragraphs = []
                        
                        for p_idx in overlap_paragraphs:
                            current_chunk += paragraphs[p_idx] + "\n\n"
                            chunk_paragraphs.append(p_idx)
                    else:
                        current_chunk = ""
                        chunk_paragraphs = []
                    
                    # 현재 문단 추가
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
                    "chunk": current_chunk.strip(),
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
            section_title = titles[i][1]
            section_paragraphs = paragraphs[start_idx:end_idx+1]
            
            # 섹션 내용 합치기
            section_content = "\n\n".join(section_paragraphs)
            
            # 섹션이 청크 크기보다 작으면 바로 추가
            if len(section_content) <= chunk_size:
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = len(chunks)
                chunk_metadata["chunk_info"] = f"섹션: {section_title.strip()}"
                chunk_metadata["section_title"] = section_title.strip()
                chunk_metadata["paragraph_indices"] = list(range(start_idx, end_idx+1))
                chunk_metadata["source_id"] = f"{metadata.get('file_hash', 'doc')}_chunk_{chunk_metadata['chunk_index']}"
                chunks.append({
                    "chunk": section_content,
                    "metadata": chunk_metadata
                })
            else:
                # 섹션이 크면 문장 단위로 분할
                section_sentences = self._split_text_to_sentences(section_content)
                current_chunk = ""
                current_sentences = []
                
                for j, sentence in enumerate(section_sentences):
                    # 현재 문장 추가 시 청크 크기 초과 여부 확인
                    if len(current_chunk) + len(sentence) <= chunk_size:
                        current_chunk += sentence
                        current_sentences.append(j)
                    else:
                        # 현재 청크 추가
                        if current_chunk:
                            chunk_metadata = metadata.copy()
                            chunk_metadata["chunk_index"] = len(chunks)
                            chunk_metadata["chunk_info"] = f"섹션: {section_title.strip()} (부분 {len(chunks) + 1})"
                            chunk_metadata["section_title"] = section_title.strip()
                            chunk_metadata["paragraph_indices"] = list(range(start_idx, end_idx+1))
                            chunk_metadata["is_section_part"] = True
                            chunk_metadata["source_id"] = f"{metadata.get('file_hash', 'doc')}_chunk_{chunk_metadata['chunk_index']}"
                            chunks.append({
                                "chunk": current_chunk,
                                "metadata": chunk_metadata
                            })
                        
                        # 중복 영역 처리
                        if overlap > 0 and current_sentences:
                            # 마지막 몇 개 문장을 중복 포함
                            overlap_size = 0
                            overlap_sentences = []
                            
                            for s_idx in reversed(current_sentences):
                                s_content = section_sentences[s_idx]
                                if overlap_size + len(s_content) <= overlap:
                                    overlap_sentences.insert(0, s_idx)
                                    overlap_size += len(s_content)
                                else:
                                    break
                            
                            # 중복 영역 포함하여 새 청크 시작
                            current_chunk = ""
                            current_sentences = []
                            
                            for s_idx in overlap_sentences:
                                current_chunk += section_sentences[s_idx]
                                current_sentences.append(s_idx)
                        else:
                            current_chunk = ""
                            current_sentences = []
                        
                        # 현재 문장 추가
                        current_chunk += sentence
                        current_sentences.append(j)
                
                # 마지막 청크 추가
                if current_chunk:
                    chunk_metadata = metadata.copy()
                    chunk_metadata["chunk_index"] = len(chunks)
                    chunk_metadata["chunk_info"] = f"섹션: {section_title.strip()} (부분 {len(chunks) + 1})"
                    chunk_metadata["section_title"] = section_title.strip()
                    chunk_metadata["paragraph_indices"] = list(range(start_idx, end_idx+1))
                    chunk_metadata["is_section_part"] = True
                    chunk_metadata["source_id"] = f"{metadata.get('file_hash', 'doc')}_chunk_{chunk_metadata['chunk_index']}"
                    chunks.append({
                        "chunk": current_chunk,
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
            chunk1 = chunks[i].get("chunk", chunks[i].get("content"))
            chunk2 = chunks[i+1].get("chunk", chunks[i+1].get("content"))
            
            # 두 청크 합치기
            combined_content = chunk1 + "\n\n" + chunk2
            
            # 중첩 청크 메타데이터
            overlap_metadata = metadata.copy()
            overlap_metadata["chunk_index"] = len(chunks) + len(overlap_chunks)
            overlap_metadata["chunk_info"] = f"중첩 청크 {i+1}-{i+2}"
            overlap_metadata["is_overlap_chunk"] = True
            overlap_metadata["original_chunks"] = [i, i+1]
            overlap_metadata["source_id"] = f"{metadata.get('file_hash', 'doc')}_overlap_{i}_{i+1}"
            
            # 중첩 청크 추가
            overlap_chunks.append({
                "chunk": combined_content,
                "metadata": overlap_metadata
            })
        
        return overlap_chunks
    def chunk_text(self, text: str, size: int = None) -> List[str]:
        """
        단일 텍스트를 기본 청크 크기 기준으로 단순 분할 (간이용)
        태깅 또는 테스트용 간단한 청크 생성
        Args:
            text: 분할할 텍스트
            size: 청크 크기 (기본값은 self.default_chunk_size)
        Returns:
            List[str]: 분할된 청크 리스트
        """
        chunk_size = size or self.default_chunk_size
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size].strip()
            if chunk:
                chunks.append(chunk)
        return chunks