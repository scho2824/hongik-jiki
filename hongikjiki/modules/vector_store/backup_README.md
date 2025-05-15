# 벡터 저장소 백업 도구 (Vector Store Backup)

이 도구는 Hongik-Jiki 챗봇의 벡터 저장소 데이터를 안전하게 백업하고 복원하기 위한 유틸리티입니다. 데이터 손실을 방지하고, 필요 시 이전 상태로 빠르게 복구할 수 있도록 지원합니다.

## 주요 기능

- **백업 생성**: 현재 벡터 저장소의 스냅샷 생성
- **백업 목록 조회**: 생성된 모든 백업 확인
- **백업 복원**: 지정된 백업에서 벡터 저장소 복원
- **백업 삭제**: 불필요한 백업 제거

## 설치 방법

이 모듈은 Hongik-Jiki 프로젝트의 일부로, 프로젝트 클론 후 사용할 수 있습니다.

```bash
# 프로젝트 루트 디렉토리에서 설치
pip install -e .
```

## 사용 방법

### 명령줄에서 실행

```bash
# 백업 생성
python -m hongikjiki.modules.vector_store.backup create

# 백업 목록 조회
python -m hongikjiki.modules.vector_store.backup list

# 백업 복원
python -m hongikjiki.modules.vector_store.backup restore backup_20230101_120000

# 백업 삭제
python -m hongikjiki.modules.vector_store.backup delete backup_20230101_120000
```

### 코드에서 사용

```python
from hongikjiki.modules.vector_store.backup import VectorStoreBackup

# 백업 도구 초기화
backup_tool = VectorStoreBackup(
    persist_directory="./data/vector_store",
    backup_directory="./data/backups/vector_store"
)

# 백업 생성
result = backup_tool.create_backup("weekly_backup")
if result["success"]:
    print(f"백업이 성공적으로 생성되었습니다: {result['backup_path']}")

# 백업 목록 조회
backups = backup_tool.list_backups()
for backup in backups["backups"]:
    print(f"백업 이름: {backup['backup_name']}, 생성 시간: {backup['created_at']}")

# 백업 복원
restore_result = backup_tool.restore_backup("weekly_backup")
if restore_result["success"]:
    print("백업에서 성공적으로 복원되었습니다.")

# 백업 삭제
delete_result = backup_tool.delete_backup("old_backup")
if delete_result["success"]:
    print(f"백업 '{delete_result['backup_name']}'이 삭제되었습니다.")
```

## 자동 백업 스케줄링

### Linux/Mac에서 Cron 작업으로 설정

```bash
# 매일 자정에 백업 생성
0 0 * * * cd /path/to/hongik-jiki && python -m hongikjiki.modules.vector_store.backup create daily_$(date +\%Y\%m\%d)

# 매주 일요일 자정에 백업 생성
0 0 * * 0 cd /path/to/hongik-jiki && python -m hongikjiki.modules.vector_store.backup create weekly_$(date +\%Y\%m\%d)
```

### Windows에서 작업 스케줄러로 설정

1. 작업 스케줄러 열기
2. '작업 만들기' 선택
3. 트리거 탭에서 일정 설정 (매일 또는 매주)
4. 동작 탭에서 프로그램 실행 선택 후 다음 설정:
   - 프로그램/스크립트: `python`
   - 인수 추가: `-m hongikjiki.modules.vector_store.backup create scheduled_backup`
   - 시작 위치: Hongik-Jiki 프로젝트 경로

## 백업 관리 전략

### 백업 순환 정책 예시

```python
import datetime
from hongikjiki.modules.vector_store.backup import VectorStoreBackup

# 백업 도구 초기화
backup_tool = VectorStoreBackup()

# 백업 목록 가져오기
backup_list = backup_tool.list_backups()

# 백업 순환 정책 적용
# - 일일 백업: 7일 보관
# - 주간 백업: 4주 보관
# - 월간 백업: 12개월 보관
today = datetime.datetime.now()
for backup in backup_list["backups"]:
    if not backup.get("created_at"):
        continue
    
    created_at = datetime.datetime.fromisoformat(backup["created_at"])
    backup_age = (today - created_at).days
    
    if "daily_" in backup["backup_name"] and backup_age > 7:
        # 7일 이상 된 일일 백업 삭제
        backup_tool.delete_backup(backup["backup_name"])
    elif "weekly_" in backup["backup_name"] and backup_age > 28:
        # 4주 이상 된 주간 백업 삭제
        backup_tool.delete_backup(backup["backup_name"])
    elif "monthly_" in backup["backup_name"] and backup_age > 365:
        # 1년 이상 된 월간 백업 삭제
        backup_tool.delete_backup(backup["backup_name"])
```

## 복원 후 검증

복원 후 벡터 저장소의 정상 작동 여부를 검증할 수 있습니다.

```python
from hongikjiki.modules.vector_store.chroma_store import ChromaVectorStore
from hongikjiki.modules.vector_store.embeddings import get_embeddings
from hongikjiki.modules.vector_store.backup import VectorStoreBackup

# 백업 복원
backup_tool = VectorStoreBackup()
restore_result = backup_tool.restore_backup("my_backup")

if restore_result["success"]:
    # 복원된 벡터 저장소 초기화
    vector_store = ChromaVectorStore(
        collection_name="hongikjiki_jungbub",
        persist_directory="./data/vector_store",
        embeddings=get_embeddings("openai", model="text-embedding-3-small")
    )
    
    # 검증 쿼리 실행
    results = vector_store.search("정법이란 무엇인가요?", k=3)
    
    # 검색 결과 확인
    if results:
        print("벡터 저장소가 성공적으로 복원되었습니다.")
        print(f"검색 결과 {len(results)}개 반환됨")
    else:
        print("복원된 벡터 저장소에 문제가 있을 수 있습니다.")
```

## 주의사항

- 대용량 벡터 저장소의 백업은 디스크 공간을 많이 차지할 수 있습니다.
- 백업 및 복원 작업은 실행 중에 다른 작업을 중단시킬 수 있습니다.
- 복원 작업 전에는 현재 상태가 자동으로 백업됩니다.
- 백업 파일 경로에 충분한 디스크 공간이 있는지 확인하세요.

## 라이선스

이 도구는 Hongik-Jiki 프로젝트의 일부로 제공됩니다.