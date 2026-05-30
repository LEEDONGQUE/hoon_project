"""
generate_normal_noise.py
========================
정상 노이즈 로그 생성기 (normal_noise.log)

포함 패턴:
  [A] 비밀번호 3번 실패 → 4번째 성공  (brute force처럼 보이는 정상)
  [B] 다양한 경로의 404 에러           (설정 오류/오타로 인한 정상 노이즈)
  [C] 일반 정상 브라우징               (배경 트래픽, 노이즈에 섞어 자연스럽게)

목표:
  - iForest/OC-SVM : 단편 피처(404 빈도, 에러 수)만 보고 공격으로 오탐 유도
  - LSTM            : 5줄 윈도우로 앞뒤 맥락(결국 로그인 성공) 파악 → 정상 통과
"""

import random
import os
from datetime import datetime, timedelta

# =========================================================
# 설정
# =========================================================
OUTPUT_FILE  = "normal_noise.log"
TOTAL_LINES  = 3000      # 총 생성 줄 수 (test_normal.log 분량과 맞춤)

# 패턴 비율
RATIO_LOGIN_NOISE = 0.35  # [A] 로그인 실패→성공 시퀀스
RATIO_404_NOISE   = 0.40  # [B] 404 에러 (다양한 경로)
RATIO_NORMAL_BG   = 0.25  # [C] 일반 정상 브라우징

# 시작 시각
BASE_TIME = datetime(2026, 5, 27, 9, 0, 0)

# =========================================================
# 데이터 풀
# =========================================================
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)",
    "curl/8.1.2",
]

IPS = [
    "192.168.1.10", "192.168.1.23", "192.168.1.45",
    "10.0.0.5",     "10.0.0.12",    "10.0.0.88",
    "172.16.0.3",   "172.16.0.19",
]

# [B] 404를 유발하는 다양한 경로 (오타/존재하지 않는 경로)
TYPO_PATHS_404 = [
    "/hom", "/indx.html", "/login.phpp", "/admn", "/dashbord",
    "/api/v1/usr", "/setings", "/profle", "/favicon.ico",
    "/robots.txt", "/sitemap.xm", "/.env", "/config.bak",
    "/wp-login.php", "/phpmyadmin", "/admin/login",
    "/static/main.cs", "/img/logo.pn", "/api/hlth",
    "/v2/endpont", "/.git/config", "/backup.sql",
]

# [C] 정상 경로
NORMAL_PATHS = [
    ("GET",  "/",              200, 512),
    ("GET",  "/dashboard",     200, 1024),
    ("GET",  "/profile",       200, 768),
    ("GET",  "/api/v1/user",   200, 256),
    ("GET",  "/static/app.js", 200, 4096),
    ("GET",  "/static/app.css",200, 2048),
    ("GET",  "/posts",         200, 1536),
    ("GET",  "/posts/42",      200, 890),
    ("POST", "/api/comment",   201, 128),
    ("GET",  "/logout",        302, 64),
    ("GET",  "/search?q=hello",200, 640),
    ("GET",  "/notifications", 200, 320),
]


# =========================================================
# 헬퍼
# =========================================================
_current_time = BASE_TIME

def next_time(min_sec=0, max_sec=3):
    global _current_time
    _current_time += timedelta(seconds=random.uniform(min_sec, max_sec))
    return _current_time.strftime("%d/%b/%Y:%H:%M:%S +0000")

def rand_ip():
    return random.choice(IPS)

def rand_ua():
    return random.choice(USER_AGENTS)

def fmt(ip, ts, method, path, status, size):
    return f'{ip} - - [{ts}] "{method} {path} HTTP/1.1" {status} {size} "-" "{rand_ua()}"'


# =========================================================
# 패턴 생성 함수
# =========================================================

def gen_login_noise_block():
    """
    [A] 비밀번호 N번(2~4번) 실패 → 마지막에 성공
    LSTM 5줄 윈도우 안에 '실패들 + 성공'이 포착되도록 구성
    → 맥락상 정상 사용자의 오타
    """
    lines = []
    ip = rand_ip()
    fail_count = random.randint(2, 4)

    for _ in range(fail_count):
        ts = next_time(0.5, 2.0)
        lines.append(fmt(ip, ts, "POST", "/login", 401, 128))

    # 성공
    ts = next_time(1.0, 3.0)
    lines.append(fmt(ip, ts, "POST", "/login", 200, 256))

    # 성공 후 정상 페이지 이동 1~2회 (문맥 완성)
    for _ in range(random.randint(1, 2)):
        method, path, status, size = random.choice(NORMAL_PATHS)
        ts = next_time(0.5, 2.0)
        lines.append(fmt(ip, ts, method, path, status, size))

    return lines


def gen_404_noise_block():
    """
    [B] 잘못된 경로 요청 (오타/존재하지 않는 리소스)
    - 경로가 매번 다름 (기존 노이즈 파일의 동일 반복 문제 해결)
    - 2~5줄씩 묶어서 발생 (한 사용자가 여러 번 잘못 접근)
    """
    lines = []
    ip = rand_ip()
    count = random.randint(2, 5)
    paths = random.sample(TYPO_PATHS_404, min(count, len(TYPO_PATHS_404)))

    for path in paths:
        ts = next_time(0.1, 1.5)
        lines.append(fmt(ip, ts, "GET", path, 404, random.randint(200, 512)))

    return lines


def gen_normal_bg_block():
    """
    [C] 일반 정상 브라우징 (배경 트래픽)
    """
    lines = []
    ip = rand_ip()
    count = random.randint(2, 5)

    for _ in range(count):
        method, path, status, size = random.choice(NORMAL_PATHS)
        ts = next_time(0.3, 2.5)
        lines.append(fmt(ip, ts, method, path, status, size))

    return lines


# =========================================================
# 메인 생성 루프
# =========================================================

def main():
    all_lines = []

    n_login = int(TOTAL_LINES * RATIO_LOGIN_NOISE)
    n_404   = int(TOTAL_LINES * RATIO_404_NOISE)
    n_bg    = int(TOTAL_LINES * RATIO_NORMAL_BG)

    # 블록 단위로 생성
    blocks = []

    # [A] 로그인 실패→성공 블록
    generated = 0
    while generated < n_login:
        block = gen_login_noise_block()
        blocks.append(("LOGIN_NOISE", block))
        generated += len(block)

    # [B] 404 노이즈 블록
    generated = 0
    while generated < n_404:
        block = gen_404_noise_block()
        blocks.append(("404_NOISE", block))
        generated += len(block)

    # [C] 정상 배경 블록
    generated = 0
    while generated < n_bg:
        block = gen_normal_bg_block()
        blocks.append(("NORMAL_BG", block))
        generated += len(block)

    # 블록 섞기 (시간순으로 자연스럽게 섞임)
    random.shuffle(blocks)

    for _, block in blocks:
        all_lines.extend(block)

    # 총 줄 수 맞추기
    all_lines = all_lines[:TOTAL_LINES]

    # 저장
    with open(OUTPUT_FILE, "w") as f:
        for line in all_lines:
            f.write(line + "\n")

    # =========================================================
    # 통계 출력
    # =========================================================
    login_lines = sum(len(b) for t, b in blocks if t == "LOGIN_NOISE")
    noise_404   = sum(len(b) for t, b in blocks if t == "404_NOISE")
    bg_lines    = sum(len(b) for t, b in blocks if t == "NORMAL_BG")

    print(f"\n[생성 완료] {OUTPUT_FILE}")
    print(f"{'─'*40}")
    print(f"  총 줄 수          : {len(all_lines)}")
    print(f"  [A] 로그인 노이즈 : {min(login_lines, n_login)}줄  (비번 실패→성공 시퀀스)")
    print(f"  [B] 404 노이즈    : {min(noise_404, n_404)}줄  (다양한 오타 경로)")
    print(f"  [C] 정상 배경     : {min(bg_lines, n_bg)}줄  (일반 브라우징)")
    print(f"{'─'*40}")
    print(f"\n  → iForest/OC-SVM 예상: 404 빈도수 높아서 FPR 상승")
    print(f"  → LSTM 예상          : 5줄 윈도우로 실패→성공 맥락 파악, 낮은 FPR 유지")

    # 샘플 미리보기
    print(f"\n[샘플 출력 (앞 15줄)]")
    print(f"{'─'*80}")
    for line in all_lines[:15]:
        print(line)


if __name__ == "__main__":
    random.seed(42)
    main()
