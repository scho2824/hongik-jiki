import json
import matplotlib.pyplot as plt
from collections import Counter

# 청크 데이터 로드
with open("data/processed/jungbub_dataset.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# 청크 길이 계산
lengths = [len(chunk.get("content", "").strip()) for chunk in chunks]

# 통계 출력
print(f"총 청크 수: {len(lengths)}")
print(f"최소 길이: {min(lengths)}")
print(f"최대 길이: {max(lengths)}")
print(f"평균 길이: {sum(lengths)/len(lengths):.1f}")
print(f"30자 미만 청크 수: {sum(1 for l in lengths if l < 30)}")
print(f"50자 미만 청크 수: {sum(1 for l in lengths if l < 50)}")
print(f"100자 미만 청크 수: {sum(1 for l in lengths if l < 100)}")

# 길이 분포 (구간별로 나누어 카운트)
bins = [0, 10, 20, 30, 50, 100, 200, 500, 1000, 5000, 10000]
hist_counts = Counter()
for length in lengths:
    for i in range(len(bins)-1):
        if bins[i] <= length < bins[i+1]:
            hist_counts[f"{bins[i]}-{bins[i+1]}"] += 1
            break
    if length >= bins[-1]:
        hist_counts[f"{bins[-1]}+"] += 1

print("\n청크 길이 분포:")
for bin_range, count in sorted(hist_counts.items()):
    print(f"{bin_range}: {count}개 ({count/len(lengths)*100:.1f}%)")

# matplotlib이 설치되어 있다면 그래프도 생성
try:
    plt.figure(figsize=(12, 6))
    plt.hist(lengths, bins=bins, alpha=0.7)
    plt.xlabel('청크 길이 (자)')
    plt.ylabel('청크 수')
    plt.title('청크 길이 분포')
    plt.axvline(x=30, color='r', linestyle='--', label='min_length (30)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('chunk_length_histogram.png')
    plt.close()
    print("\n히스토그램이 chunk_length_histogram.png에 저장되었습니다.")
except Exception as e:
    print(f"그래프 생성 실패: {e}")
    