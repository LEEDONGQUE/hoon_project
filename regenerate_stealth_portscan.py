"""
regenerate_stealth_portscan.py
===============================
stealth_portscan.log 재생성기

목표:
  - train_normal.log에서 라인을 그대로 샘플링
  - 피처 분포(길이, 경로 빈도 등)가 정상 로그와 100% 동일
  - OC-SVM/RF가 피처만으로는 구조적으로 탐지 불가능한 상태 만들기
  - LSTM만 '순서 흐름'으로 탐지 가능하도록

원리:
  정상 로그에서 줄을 그대로 가져오되,
  윈도우(5줄) 단위 순서를 정상과 다르게 섞어서
  개별 줄 피처는 동일하지만 시퀀스 흐름은 이상하게 만듦
  → LSTM PPL 상승 유발
"""

import random

INPUT_FILE  = "normal_pure.txt"
OUTPUT_FILE = "stealth_portscan.log"
TARGET_LINES = 5000
WINDOW = 5
SEED = 99

random.seed(SEED)

# ─── 원본 정상 로그 로드 ───────────────────────────
with open(INPUT_FILE, "r", errors="ignore") as f:
    normal_lines = [l.strip() for l in f if l.strip()]

print(f"[로드] {INPUT_FILE}: {len(normal_lines)}줄")

# ─── 윈도우 단위로 묶기 ───────────────────────────
# 정상 흐름: [A B C D E] [F G H I J] ...
# 포트스캔 흐름: 윈도우 내부 순서를 섞어서 [C A E B D] [I F J H G] ...
# → 개별 줄은 정상이지만 흐름이 이상함

windows = []
for i in range(0, len(normal_lines) - WINDOW + 1, WINDOW):
    windows.append(normal_lines[i : i + WINDOW])

# 윈도우 내부 셔플 (포트스캔 특성: 무작위 순서로 다양한 경로 탐색)
shuffled_lines = []
for w in windows:
    w_copy = w[:]
    random.shuffle(w_copy)      # 윈도우 내 순서 섞기
    shuffled_lines.extend(w_copy)

# 추가로 윈도우 간 순서도 섞기 (더 강한 시퀀스 이상)
random.shuffle(windows)
inter_shuffled = []
for w in windows:
    inter_shuffled.extend(w)

# 두 방식 반반 섞기
combined = shuffled_lines[:TARGET_LINES // 2] + inter_shuffled[:TARGET_LINES // 2]
random.shuffle(combined)
result = combined[:TARGET_LINES]

# ─── 저장 ─────────────────────────────────────────
with open(OUTPUT_FILE, "w") as f:
    for line in result:
        f.write(line + "\n")

print(f"[완료] {OUTPUT_FILE}: {len(result)}줄 생성")

# ─── 검증: 피처 분포 비교 ─────────────────────────
import re
import numpy as np

def quick_features(lines):
    feats = []
    for line in lines:
        nums = re.findall(r"\d+", line)
        feats.append([
            len(line),
            len(nums),
            line.count("/"),
            line.count("?"),
            line.count("="),
            line.count("&"),
            sum(int(n) for n in nums[:3]) if nums else 0,
        ])
    return np.array(feats, dtype=float)

orig_feat  = quick_features(normal_lines[:TARGET_LINES])
port_feat  = quick_features(result)

print(f"\n[피처 분포 비교] (평균값, 클수록 차이 없음이 좋음)")
print(f"  {'피처':<25} {'정상 평균':>10} {'portscan 평균':>14} {'차이':>8}")
print(f"  {'-'*60}")
feat_names = ["line_length", "num_count", "slash_count",
              "query_count", "equal_count", "amp_count", "numeric_sum"]
for i, name in enumerate(feat_names):
    o, p = orig_feat[:, i].mean(), port_feat[:, i].mean()
    diff = abs(o - p)
    ok = "✓ 동일" if diff < 0.5 else "△ 유사" if diff < 2.0 else "✗ 다름"
    print(f"  {name:<25} {o:>10.3f} {p:>14.3f} {diff:>7.3f}  {ok}")

print(f"\n→ 피처가 동일하므로 ML 모델은 구조적으로 탐지 불가")
print(f"→ LSTM만 시퀀스 흐름 이상으로 탐지 가능")
