import re

raw_file_path = "attack_raw.txt"
attack_out_path = "attack_pure.txt"
normal_out_path = "normal_pure.txt"

# 지능형 포트스캔(NSE), 무차별대입, SQLi, DDoS 단골 키워드 싹 긁어모으기
attack_keywords = [
    "ApacheBench", "sqlmap", "hydra", "Nmap", 
    "union", "admin", "etc/passwd", 
    "HNAP1", "nmaplowercheck", "testproxy", "w00tw00t"
]
normal_keywords = ["python-requests"]

attack_count = 0
normal_count = 0

print("정제 작업을 시작합니다...")

with open(raw_file_path, "r", encoding="utf-8", errors="ignore") as f,\
     open(attack_out_path, "w", encoding="utf-8") as fa,\
     open(normal_out_path, "w", encoding="utf-8") as fn:
    
    for line in f:
        # 1. 지능형 및 일반 공격 로그 분류
        if any(kw in line for kw in attack_keywords):
            fa.write(line)
            attack_count += 1
        # 2. 동규님 파이썬 크롤러 -> 정상 로그로 분류
        elif any(kw in line for kw in normal_keywords):
            fn.write(line)
            normal_count += 1

print("\n✨ 정제 완료!")
print(f" - 순수 공격 로그 (attack_pure.txt): {attack_count:,} 줄")
print(f" - 순수 정상 로그 (normal_pure.txt): {normal_count:,} 줄")
