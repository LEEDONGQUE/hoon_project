"""
gemma.py
========
LSTM이 탐지한 이상 로그를 로컬 Gemma에 전달하여
공격 유형 분석 및 대응 가이드를 생성

실행 방법:
  python3 gemma.py

사전 조건:
  ollama pull gemma2:2b
  ollama serve (백그라운드 실행)
"""

import re
import json
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================
# 설정
# =========================================================
NORMAL_LOG_PATH = "normal_pure.txt"
ATTACK_LOG_PATH = "sqli.log"        # 또는 stealth_portscan.log
WINDOW_SIZE     = 5
TRAIN_RATIO     = 0.80
SIGMA_K         = 3
OLLAMA_URL      = "http://localhost:11434/api/generate"
GEMMA_MODEL     = "gemma2:2b"

# =========================================================
# 1. 로그 파서 (원본 로그 → Log ID + 원본 텍스트 보존)
# =========================================================
_LOG_RE = re.compile(
    r'"(GET|POST|PUT|DELETE|HEAD|OPTIONS|PATCH|PROPFIND)\s+(\S+)\s+HTTP/[\d.]+"\s+(\d{3})'
)

def normalize_path(path, keep_query=False):
    if not keep_query:
        path = path.split("?")[0]
    else:
        if "?" in path:
            base, query = path.split("?", 1)
            path = base + "?" + query[:80]
    parts = path.split("/")
    parts = ["<NUM>" if re.fullmatch(r"\d+", p) else p for p in parts]
    return "/".join(parts)

def parse_logs(filepath, template_map, status_filter=None, keep_query=False):
    """
    로그 파일 파싱
    반환: (ids 배열, 원본 로그 라인 리스트)
    """
    ids   = []
    lines = []
    with open(filepath, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = _LOG_RE.search(line)
            if not m:
                continue
            status = m.group(3)
            if status_filter and status not in status_filter:
                continue
            method = m.group(1)
            path   = normalize_path(m.group(2), keep_query=keep_query)
            key    = (method, path, status)
            if key not in template_map:
                template_map[key] = len(template_map) + 1
            ids.append(template_map[key])
            lines.append(line)   # 원본 텍스트 보존
    return np.array(ids, dtype=np.int32), lines

# =========================================================
# 2. LSTM 모델 (동규 코드 구조 동일)
# =========================================================
class LSTMLM(nn.Module):
    def __init__(self, vocab_size, embed_size=64, hidden_size=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm  = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc    = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.lstm(x)
        return self.fc(out)

def create_sequences(logs, w):
    if len(logs) <= w:
        return torch.LongTensor([]), torch.LongTensor([])
    X, y = [], []
    for i in range(len(logs) - w):
        X.append(logs[i:i+w])
        y.append(logs[i+w])
    return torch.LongTensor(np.array(X)), torch.LongTensor(np.array(y))

def compute_ppl(model, X, y):
    if len(X) == 0:
        return np.array([])
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        out  = model(X.to(device))
        pred = out[:, -1, :]
        loss = F.cross_entropy(pred, y.to(device), reduction='none')
        return torch.exp(loss).cpu().numpy()

def train_model(model, X, y, epochs=20, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=lr)
    crit   = nn.CrossEntropyLoss()
    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        out  = model(X.to(device))
        pred = out[:, -1, :]
        loss = crit(pred, y.to(device))
        loss.backward()
        opt.step()
        if (ep+1) % 5 == 0:
            print(f"    epoch {ep+1}/{epochs}  loss={loss.item():.4f}")
    return model

# =========================================================
# 3. Gemma 호출 (ollama API)
# =========================================================
def ask_gemma(anomaly_log_lines):
    """
    원본 로그 라인을 Gemma에 전달하여 대응 가이드 생성
    anomaly_log_lines: 원본 Apache 로그 텍스트 리스트
    """
    log_text = "\n".join(anomaly_log_lines)

    prompt = f"""You are a cybersecurity expert analyzing web server logs.

[Anomaly Detected - Server Log]
{log_text}

Please analyze and provide:
1. Attack Type (e.g., SQL Injection, Path Traversal, Brute Force, Port Scan)
2. Threat Level (Low / Medium / High / Critical)
3. Attacker's Intent
4. Immediate Response Actions
5. Prevention Measures

Answer in Korean."""

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": GEMMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 512,
                }
            },
            timeout=120
        )
        if resp.status_code == 200:
            return resp.json().get("response", "응답 없음")
        else:
            return f"Gemma 오류: {resp.status_code}"
    except requests.exceptions.ConnectionError:
        return "ollama 서버가 실행 중이지 않습니다. 'ollama serve' 실행 후 다시 시도하세요."

# =========================================================
# 4. 메인
# =========================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  LSTM 이상 탐지 + Gemma 대응 가이드")
    print("=" * 60)

    # ── 데이터 로드 ──────────────────────────────────────
    print("\n[데이터 로드]")
    template_map = {}
    normal_ids, normal_lines = parse_logs(
        NORMAL_LOG_PATH, template_map, status_filter={"200"}
    )
    attack_ids, attack_lines = parse_logs(
        ATTACK_LOG_PATH, template_map, keep_query=True
    )
    vocab_size = len(template_map)
    print(f"  정상 로그  : {len(normal_ids)}줄  (vocab: {vocab_size})")
    print(f"  공격 로그  : {len(attack_ids)}줄")

    # ── 정규화 ───────────────────────────────────────────
    max_id       = max(normal_ids.max(), attack_ids.max())
    normal_norm  = normal_ids / max_id
    attack_norm  = attack_ids / max_id

    # ── LSTM 학습 ────────────────────────────────────────
    print("\n[LSTM 학습] 정상 로그만 사용")
    train_size = int(len(normal_norm) * TRAIN_RATIO)
    val_size   = len(normal_norm) - train_size

    X_train, y_train = create_sequences(normal_norm[:train_size],  WINDOW_SIZE)
    X_val,   y_val   = create_sequences(normal_norm[train_size:],  WINDOW_SIZE)
    X_atk,   y_atk   = create_sequences(attack_norm,               WINDOW_SIZE)

    model = LSTMLM(vocab_size + 1)
    model = train_model(model, X_train, y_train, epochs=20)

    # ── 임계치 (μ + 3σ) ──────────────────────────────────
    val_ppls  = compute_ppl(model, X_val, y_val)
    mu        = val_ppls.mean()
    sigma     = val_ppls.std()
    threshold = mu + SIGMA_K * sigma
    print(f"\n[임계치] μ={mu:.4f}  σ={sigma:.4f}  임계치={threshold:.4f}")

    # ── 이상 탐지 ────────────────────────────────────────
    print("\n[이상 탐지]")
    atk_ppls    = compute_ppl(model, X_atk, y_atk)
    anomaly_idx = np.where(atk_ppls > threshold)[0]
    tpr         = len(anomaly_idx) / len(atk_ppls) * 100

    print(f"  공격 시퀀스 : {len(atk_ppls)}")
    print(f"  탐지 수     : {len(anomaly_idx)}")
    print(f"  TPR         : {tpr:.1f}%")

    if len(anomaly_idx) == 0:
        print("\n탐지된 이상 없음.")
        exit()

    # ── 원본 로그 추출 ───────────────────────────────────
    # 이상 탐지된 시퀀스의 마지막 줄 인덱스 = anomaly_idx + WINDOW_SIZE
    print("\n[탐지된 원본 로그]")
    sample_lines = []
    for idx in anomaly_idx[:3]:   # 대표 3개
        log_idx = int(idx) + WINDOW_SIZE
        if log_idx < len(attack_lines):
            line = attack_lines[log_idx]
            sample_lines.append(line)
            print(f"  PPL={atk_ppls[idx]:.1f}  {line[:80]}")

    # ── Gemma 분석 ───────────────────────────────────────
    print("\n[Gemma 분석 중...]")
    print("(ollama가 실행 중이어야 합니다: ollama serve)\n")

    response = ask_gemma(sample_lines)

    print("=" * 60)
    print("  Gemma 대응 가이드")
    print("=" * 60)
    print(response)
    print("=" * 60)
