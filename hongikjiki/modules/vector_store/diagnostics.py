# hongikjiki/modules/vector_store/diagnostics.py
import time
import logging
import json
from pathlib import Path
import numpy as np
from typing import List, Dict, Any, Optional

from hongikjiki.modules.vector_store.chroma_store import ChromaVectorStore
from hongikjiki.modules.vector_store.embeddings import get_embeddings

logger = logging.getLogger("VectorStoreDiagnostics")

class VectorStoreDiagnostics:
    """
    벡터 저장소 성능 진단 및 모니터링 도구
    """
    
    def __init__(self, vector_store: ChromaVectorStore):
        """
        진단 도구 초기화
        
        Args:
            vector_store: 진단할 벡터 저장소 인스턴스
        """
        self.vector_store = vector_store
        
    def run_benchmarks(self, test_queries: Optional[List[str]] = None, k: int = 3) -> Dict[str, Any]:

        """
        벡터 저장소 성능 벤치마크 실행
        
        Args:
            test_queries: 테스트할 쿼리 목록
            k: 각 쿼리당 검색할 결과 수
            
        Returns:
            Dict: 벤치마크 결과
        """
        if test_queries is None:
            test_queries = [
                "정법이란 무엇인가요?",
                "홍익인간의 의미는 무엇인가요?",
                "감정을 다스리는 방법은 무엇인가요?",
                "마음 수행은 어떻게 해야 하나요?",
                "정법에서 말하는 사랑이란?"
            ]
        
        results = {
            "total_documents": self.vector_store.count(),
            "queries": [],
            "avg_search_time": 0,
            "search_times": [],
            "avg_results_per_query": 0
        }
        
        total_results = 0
        
        for query in test_queries:
            start_time = time.time()
            search_results = self.vector_store.search(query, k=k)
            end_time = time.time()
            
            search_time = end_time - start_time
            results["search_times"].append(search_time)
            
            result_count = len(search_results) if search_results else 0
            total_results += result_count
            
            # 쿼리별 결과 저장
            query_result = {
                "query": query,
                "search_time": search_time,
                "result_count": result_count,
                "results": []
            }
            
            # 상위 결과의 요약 정보 저장
            for i, result in enumerate(search_results[:k]):
                score = result.get("score", 0)
                content_preview = result.get("content", "")[:100] + "..." if len(result.get("content", "")) > 100 else result.get("content", "")
                query_result["results"].append({
                    "index": i+1,
                    "score": score,
                    "content_preview": content_preview
                })
            
            results["queries"].append(query_result)
        
        # 평균 계산
        results["avg_search_time"] = sum(results["search_times"]) / len(test_queries) if test_queries else 0
        results["avg_results_per_query"] = total_results / len(test_queries) if test_queries else 0
        
        # 검색 시간 통계
        search_times = np.array(results["search_times"])
        results["min_search_time"] = float(np.min(search_times)) if len(search_times) > 0 else 0
        results["max_search_time"] = float(np.max(search_times)) if len(search_times) > 0 else 0
        results["std_search_time"] = float(np.std(search_times)) if len(search_times) > 0 else 0
        
        return results
    
    def analyze_tags(self) -> Dict[str, Any]:
        """
        태그 분포 및 사용 현황 분석
        
        Returns:
            Dict: 태그 분석 결과
        """
        all_docs = self.vector_store.get_all_documents()
        metadatas = all_docs.get("metadatas", [])
        
        tag_counts = {}
        docs_with_tags = 0
        
        for meta in metadatas:
            if meta and "tags" in meta:
                docs_with_tags += 1
                tags = meta["tags"]
                
                # 태그가 문자열인 경우 처리
                if isinstance(tags, str):
                    tags = [tag.strip() for tag in tags.split(",")]
                
                # 태그 카운트 업데이트
                if isinstance(tags, list):
                    for tag in tags:
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # 태그 분석 결과
        return {
            "total_documents": len(metadatas),
            "documents_with_tags": docs_with_tags,
            "tagging_rate": docs_with_tags / len(metadatas) if metadatas else 0,
            "unique_tags": len(tag_counts),
            "top_tags": sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            "tag_distribution": tag_counts
        }
    
    def check_duplicates(self) -> Dict[str, Any]:
        """
        중복 문서 검사
        
        Returns:
            Dict: 중복 문서 분석 결과
        """
        all_docs = self.vector_store.get_all_documents()
        documents = all_docs.get("documents", [])
        metadatas = all_docs.get("metadatas", [])
        
        # 문서 내용 기반 중복 검사
        content_hashes = {}
        duplicate_count = 0
        duplicate_groups = []
        
        for i, doc in enumerate(documents):
            if not doc:
                continue
                
            # 문서 콘텐츠의 처음 100자만 해시화 (유사 중복 감지용)
            content_hash = hash(doc[:100])
            
            if content_hash in content_hashes:
                duplicate_count += 1
                content_hashes[content_hash].append(i)
                
                # 첫 중복 발견 시 그룹 생성
                if len(content_hashes[content_hash]) == 2:
                    duplicate_groups.append(content_hashes[content_hash])
            else:
                content_hashes[content_hash] = [i]
        
        # source_id 기반 추가 검사
        source_ids = {}
        duplicate_source_ids = 0
        
        for i, meta in enumerate(metadatas):
            if meta and "source_id" in meta:
                source_id = meta["source_id"]
                
                if source_id in source_ids:
                    duplicate_source_ids += 1
                    source_ids[source_id].append(i)
                else:
                    source_ids[source_id] = [i]
        
        # 중복 문서 그룹 중 일부만 샘플로 추출
        duplicate_examples = []
        for group in duplicate_groups[:3]:  # 최대 3개 그룹만 예시로 표시
            dup_docs = []
            for idx in group:
                if idx < len(documents):
                    content_preview = documents[idx][:100] + "..." if len(documents[idx]) > 100 else documents[idx]
                    dup_docs.append({
                        "index": idx,
                        "preview": content_preview
                    })
            duplicate_examples.append(dup_docs)
        
        return {
            "total_documents": len(documents),
            "content_duplicates": duplicate_count,
            "source_id_duplicates": duplicate_source_ids,
            "duplicate_examples": duplicate_examples
        }
    
    def save_report(self, report: Dict[str, Any], output_path: str) -> None:
        """
        진단 보고서 저장
        
        Args:
            report: 저장할 보고서 데이터
            output_path: 출력 파일 경로
        """
        # 보고서에 타임스탬프 추가
        report["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # 파일로 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"벡터 저장소 진단 보고서가 저장되었습니다: {output_path}")
    
    def run_full_diagnostics(self, output_path: str = "vector_store_report.json") -> Dict[str, Any]:
        """
        전체 진단 실행 및 보고서 생성
        
        Args:
            output_path: 보고서 저장 경로
            
        Returns:
            Dict: 진단 보고서
        """
        report = {
            "benchmarks": self.run_benchmarks(),
            "tag_analysis": self.analyze_tags(),
            "duplicate_analysis": self.check_duplicates()
        }
        
        # 보고서 저장
        self.save_report(report, output_path)
        
        return report

# 모듈 단독 실행 시 진단 실행
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 벡터 저장소 초기화
    vector_store = ChromaVectorStore(
        collection_name="hongikjiki_jungbub",
        persist_directory="./data/vector_store",
        embeddings=get_embeddings("openai", model="text-embedding-3-small")
    )
    
    # 진단 도구 초기화 및 실행
    diagnostics = VectorStoreDiagnostics(vector_store)
    diagnostics.run_full_diagnostics()