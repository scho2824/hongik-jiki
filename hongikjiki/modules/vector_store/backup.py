# hongikjiki/modules/vector_store/backup.py
import os
import shutil
import logging
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger("VectorStoreBackup")

class VectorStoreBackup:
    """
    벡터 저장소 백업 및 복구 기능
    """
    
    def __init__(self, 
                persist_directory: str = "./data/vector_store",
                backup_directory: str = "./data/backups/vector_store"):
        """
        백업 도구 초기화
        
        Args:
            persist_directory: 벡터 저장소 경로
            backup_directory: 백업 저장 경로
        """
        self.persist_directory = Path(persist_directory)
        self.backup_directory = Path(backup_directory)
        
        # 백업 디렉토리 생성
        os.makedirs(self.backup_directory, exist_ok=True)
    
    def create_backup(self, backup_name: Optional[str] = None) -> Dict[str, Any]:
        """
        현재 벡터 저장소 백업 생성
        
        Args:
            backup_name: 백업 이름 (지정하지 않으면 타임스탬프 사용)
            
        Returns:
            Dict: 백업 정보
        """
        # 벡터 저장소 경로 확인
        if not self.persist_directory.exists():
            logger.error(f"벡터 저장소 경로가 존재하지 않습니다: {self.persist_directory}")
            return {"success": False, "error": "벡터 저장소 경로가 존재하지 않습니다."}
        
        # 백업 이름 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = backup_name or f"backup_{timestamp}"
        
        # 백업 경로 설정
        backup_path = self.backup_directory / backup_name
        
        try:
            # 디렉토리 복사
            shutil.copytree(self.persist_directory, backup_path)
            
            # 백업 메타데이터 생성
            metadata = {
                "backup_name": backup_name,
                "source_directory": str(self.persist_directory),
                "backup_directory": str(backup_path),
                "timestamp": timestamp,
                "created_at": datetime.now().isoformat()
            }
            
            # 메타데이터 저장
            metadata_path = backup_path / "backup_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"벡터 저장소 백업 생성 완료: {backup_path}")
            return {
                "success": True,
                "backup_name": backup_name,
                "backup_path": str(backup_path),
                "timestamp": timestamp
            }
            
        except Exception as e:
            logger.error(f"백업 생성 중 오류 발생: {e}")
            return {"success": False, "error": str(e)}
    
    def list_backups(self) -> Dict[str, Any]:
        """
        생성된 백업 목록 조회
        
        Returns:
            Dict: 백업 목록 정보
        """
        backups = []
        
        try:
            for item in self.backup_directory.iterdir():
                if item.is_dir():
                    # 백업 메타데이터 확인
                    metadata_path = item.joinpath("backup_metadata.json")
                    if metadata_path.exists():
                        try:
                            with open(metadata_path, 'r', encoding='utf-8') as f:
                                metadata = json.load(f)
                                backups.append(metadata)
                        except Exception:
                            # 메타데이터 파일이 손상된 경우 기본 정보만 추가
                            backups.append({
                                "backup_name": item.name,
                                "backup_directory": str(item),
                                "created_at": None
                            })
                    else:
                        # 메타데이터 파일이 없는 경우 기본 정보만 추가
                        backups.append({
                            "backup_name": item.name,
                            "backup_directory": str(item),
                            "created_at": None
                        })
            
            # 생성일자 기준 내림차순 정렬 (최신 백업이 맨 위로)
            backups.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            
            return {
                "success": True,
                "total_backups": len(backups),
                "backups": backups
            }
        
        except Exception as e:
            logger.error(f"백업 목록 조회 중 오류 발생: {e}")
            return {"success": False, "error": str(e)}
    
    def restore_backup(self, backup_name: str) -> Dict[str, Any]:
        """
        지정된 백업에서 벡터 저장소 복원
        
        Args:
            backup_name: 복원할 백업 이름
            
        Returns:
            Dict: 복원 결과 정보
        """
        backup_path = self.backup_directory / backup_name
        
        # 백업 경로 확인
        if not backup_path.exists():
            logger.error(f"백업 경로가 존재하지 않습니다: {backup_path}")
            return {"success": False, "error": "지정된 백업이 존재하지 않습니다."}
        
        try:
            # 현재 벡터 저장소 백업 (복원 실패 시 복구용)
            temp_backup_name = f"pre_restore_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            temp_backup_path = self.backup_directory / temp_backup_name
            
            if self.persist_directory.exists():
                shutil.copytree(self.persist_directory, temp_backup_path)
                logger.info(f"복원 전 현재 상태 백업 완료: {temp_backup_path}")
                
                # 기존 벡터 저장소 삭제
                shutil.rmtree(self.persist_directory)
            
            # 백업에서 복원
            shutil.copytree(backup_path, self.persist_directory)
            
            logger.info(f"백업 '{backup_name}'에서 벡터 저장소 복원 완료")
            return {
                "success": True,
                "backup_name": backup_name,
                "restored_at": datetime.now().isoformat(),
                "temp_backup": temp_backup_name
            }
            
        except Exception as e:
            logger.error(f"백업 복원 중 오류 발생: {e}")
            return {"success": False, "error": str(e)}
    
    def delete_backup(self, backup_name: str) -> Dict[str, Any]:
        """
        지정된 백업 삭제
        
        Args:
            backup_name: 삭제할 백업 이름
            
        Returns:
            Dict: 삭제 결과 정보
        """
        backup_path = self.backup_directory / backup_name
        
        # 백업 경로 확인
        if not backup_path.exists():
            logger.error(f"백업 경로가 존재하지 않습니다: {backup_path}")
            return {"success": False, "error": "지정된 백업이 존재하지 않습니다."}
        
        try:
            # 백업 디렉토리 삭제
            shutil.rmtree(backup_path)
            
            logger.info(f"백업 '{backup_name}' 삭제 완료")
            return {
                "success": True,
                "backup_name": backup_name,
                "deleted_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"백업 삭제 중 오류 발생: {e}")
            return {"success": False, "error": str(e)}

# 모듈 단독 실행 시 백업 생성
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    backup_tool = VectorStoreBackup()
    result = backup_tool.create_backup()
    
    if result["success"]:
        logger.info(f"벡터 저장소 백업이 성공적으로 생성되었습니다: {result['backup_path']}")
    else:
        logger.error(f"벡터 저장소 백업 생성 실패: {result.get('error', '알 수 없는 오류')}")