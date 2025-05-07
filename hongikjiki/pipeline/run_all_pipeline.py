#!/usr/bin/env python3
"""
홍익지기 챗봇 Two-Stage 파이프라인 실행 스크립트

스테이지 1: 문서 처리 (문서 로드, 청크화, 태깅)
스테이지 2: 벡터 저장소 구축 (청크 병합, QA 생성, 벡터화)
"""
import os
import sys
import logging
import subprocess
import time
import json
from datetime import datetime, timedelta

# 로깅 설정
os.makedirs("logs", exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f"logs/pipeline_{timestamp}.log"

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[
                       logging.FileHandler(log_filename),
                       logging.StreamHandler()
                   ])
logger = logging.getLogger("HongikJikiPipeline")

def run_command(command, desc):
    """명령어 실행 및 결과 기록"""
    logger.info(f"==== {desc} 시작 ====")
    start_time = time.time()
    
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            shell=True
        )
        
        # 실시간 출력 캡처
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                logger.info(output.strip())
        
        # 오류 확인
        stderr = process.stderr.read()
        if stderr:
            logger.error(stderr)
            
        # 종료 코드 확인
        return_code = process.poll()
        if return_code != 0:
            logger.error(f"{desc} 실패 (종료 코드: {return_code})")
            return False
            
        elapsed = time.time() - start_time
        logger.info(f"==== {desc} 완료 (소요 시간: {timedelta(seconds=int(elapsed))}) ====")
        return True
        
    except Exception as e:
        logger.error(f"{desc} 오류: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def prepare_directories():
    """필요한 디렉토리 구조 준비"""
    directories = [
        "data/jungbub_teachings",         # 원본 문서
        "data/processed_originals",       # 처리 완료된 원본
        "data/tag_data/input_chunks/processed_chunks", # 처리된 청크
        "data/tag_data/auto_tagged",      # 태깅된 문서
        "data/qa",                        # QA 생성 결과
        "data/vector_store",              # 벡터 저장소
        "data/processed",                 # 처리된 데이터셋
        "logs"                            # 로그 파일
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"디렉토리 확인: {directory}")

def count_documents(directory):
    """디렉토리 내 문서 파일 수 계산"""
    count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in ['.txt', '.rtf', '.pdf', '.docx', '.md']:
                count += 1
    return count

def stage1_document_processing():
    """
    스테이지 1: 문서 처리
    - 문서 로드 및 청크화
    - 태깅
    """
    logger.info("===== 스테이지 1: 문서 처리 시작 =====")
    stage_start = time.time()
    
    # 문서 처리 전 상태 확인
    orig_count = count_documents("data/jungbub_teachings")
    processed_count = count_documents("data/processed_originals")
    logger.info(f"처리 전 상태: 원본 문서 {orig_count}개, 처리 완료 문서 {processed_count}개")
    
    # 1-1: 문서 청킹 파이프라인 실행
    stage1_1_success = run_command(
        "python3 hongikjiki/pipeline/run_chunking_pipeline.py",
        "1-1: 문서 로드 및 청크화"
    )
    
    if not stage1_1_success:
        logger.error("스테이지 1-1 실패로 파이프라인 중단")
        return False
    
    # 1-2: 문서 태깅
    stage1_2_success = run_command(
        "python3 hongikjiki/pipeline/run_tagging.py",
        "1-2: 문서 태깅"
    )
    
    if not stage1_2_success:
        logger.error("스테이지 1-2 실패로 파이프라인 중단")
        return False
    
    # 처리 후 상태 확인
    processed_count_after = count_documents("data/processed_originals")
    logger.info(f"처리 후 상태: 처리 완료 문서 {processed_count_after}개 (새로 처리됨: {processed_count_after - processed_count}개)")
    
    stage_elapsed = time.time() - stage_start
    logger.info(f"===== 스테이지 1: 문서 처리 완료 (소요 시간: {timedelta(seconds=int(stage_elapsed))}) =====")
    
    return True

def stage2_vector_building():
    """
    스테이지 2: 벡터 저장소 구축
    - 청크 병합
    - QA 생성
    - 벡터 저장소 구축
    """
    logger.info("===== 스테이지 2: 벡터 저장소 구축 시작 =====")
    stage_start = time.time()
    
    # 2-1: 청크 병합
    stage2_1_success = run_command(
        "python3 hongikjiki/utils/merge_chunks.py --input-dir data/tag_data/auto_tagged --output-file data/processed/jungbub_dataset.json",
        "2-1: 청크 병합"
    )
    
    if not stage2_1_success:
        logger.error("스테이지 2-1 실패로 파이프라인 중단")
        return False
    
    # 2-2: QA 생성
    stage2_2_success = run_command(
        "python3 hongikjiki/qa_generation/generate_qa.py --input_file data/processed/jungbub_dataset.json --output_file data/qa/jungbub_qa_dataset.json",
        "2-2: QA 생성"
    )
    
    if not stage2_2_success:
        logger.error("스테이지 2-2 실패로 파이프라인 중단")
        return False
    
    # 2-3: 벡터 저장소 구축
    stage2_3_success = run_command(
        "python3 hongikjiki/scripts/build_vector_store.py --qa_file data/qa/jungbub_qa_dataset.json --persist_dir data/vector_store --collection_name hongikjiki_jungbub",
        "2-3: 벡터 저장소 구축"
    )
    
    if not stage2_3_success:
        logger.error("스테이지 2-3 실패로 파이프라인 중단")
        return False
    
    stage_elapsed = time.time() - stage_start
    logger.info(f"===== 스테이지 2: 벡터 저장소 구축 완료 (소요 시간: {timedelta(seconds=int(stage_elapsed))}) =====")
    
    return True

def main():
    """전체 파이프라인 실행"""
    logger.info("===== 홍익지기 Two-Stage 파이프라인 시작 =====")
    start_time = time.time()
    
    # 디렉토리 구조 준비
    prepare_directories()
    
    # 스테이지 1: 문서 처리
    if not stage1_document_processing():
        return False
    
    # 스테이지 2: 벡터 저장소 구축
    if not stage2_vector_building():
        return False
    
    elapsed = time.time() - start_time
    logger.info(f"===== 홍익지기 파이프라인 완료 (총 소요 시간: {timedelta(seconds=int(elapsed))}) =====")
    logger.info(f"로그 파일: {log_filename}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)