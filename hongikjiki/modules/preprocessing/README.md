# Preprocessing Module

이 모듈은 자막 파일(.txt / .json)로부터 전처리된 텍스트 데이터를 추출하고,  
통일된 형식의 JSON 데이터셋을 생성하는 기능을 제공합니다.

---

## 📁 주요 파일

| 파일명 | 설명 |
|--------|------|
| `preprocess_subtitles.py` | 자막 파일 읽기, 텍스트 정규화, 간단한 태그 추출 및 저장 |

---

## ⚙️ 주요 기능

- 자막 폴더 내 `.txt`, `.json` 파일 읽기
- 문자열 정규화 (`normalize_text`)
- 간단한 키워드 기반 태그 추출 (`extract_tags`)
- 통일된 JSON 포맷으로 저장 (`save_dataset`)

---

## 🔄 사용 예시 (CLI)

```bash
python preprocess_subtitles.py \
  --input_dir ./data/raw_subtitles \
  --output_file ./data/processed_dataset.json