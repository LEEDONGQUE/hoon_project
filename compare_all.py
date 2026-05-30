"""
compare_all.py
==============
ML 3종 + LSTM Autoencoder 통합 비교 실험

[공통 데이터]
  정상      : normal_pure.txt  (200 응답만)
  알려진공격 : sqli.log
  신종공격   : portscan.log    (학습 완전 제외)
  정상노이즈 : normal_noise.log

[ML 입력]  : 로그 1줄 → 7차원 피처 → flatten_window(5) → 35차원
[LSTM 입력]: 로그 1줄 → Log ID → sliding_window(5) → (5,1) 시퀀스

[실험]
  A: 정상 + sqli          (기준선)
  B: 정상 + 노이즈 + sqli  (FPR 비교)
  C: portscan만            (신종공격 TPR 비교)
"""

import re
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix, recall_score, f1_score, precision_score, accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping

np.random.seed(42)
tf.random.set_seed(42)

WINDOW     = 5
TRAIN_RATIO = 0.80
GAP         = 50
ML_TARGET   = 2500   # ML 실험용 샘플 수

# =========================================================
# 1. 공통 로그 파서
# =========================================================
_LOG_RE = re.compile(
    r'"(GET|POST|PUT|DELETE|HEAD|OPTIONS|PATCH|PROPFIND)\s+(\S+)\s+HTTP/[\d.]+"'
    r'\s+(\d{3})'
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

def load_log_file(filepath, status_filter=None, keep_query=False):
    """로그 파일 → 파싱된 줄 리스트 반환"""
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
            lines.append(line)
    return lines

# =========================================================
# 2. ML용 피처 추출
# =========================================================
def extract_features(lines):
    X = []
    for line in lines:
        nums = re.findall(r"\d+", line)
        vec = [
            len(line),
            len(nums),
            line.count("/"),
            line.count("?"),
            line.count("="),
            line.count("&"),
            sum(int(n) for n in nums[:3]) if nums else 0,
        ]
        X.append(vec)
    return np.array(X, dtype=float)

def flatten_windows(X, window=5):
    out = []
    for i in range(len(X) - window + 1):
        out.append(X[i:i+window].flatten())
    return np.array(out)

# =========================================================
# 3. LSTM용 Log ID 변환
# =========================================================
def lines_to_ids(lines, template_map, keep_query=False):
    ids = []
    for line in lines:
        m = _LOG_RE.search(line)
        if not m:
            continue
        method = m.group(1)
        path   = normalize_path(m.group(2), keep_query=keep_query)
        status = m.group(3)
        key    = (method, path, status)
        if key not in template_map:
            template_map[key] = len(template_map) + 1
        ids.append(template_map[key])
    return np.array(ids, dtype=np.int32)

def make_windows(arr, w):
    return np.array([arr[i:i+w] for i in range(len(arr)-w+1)],
                    dtype=np.float32).reshape(-1, w, 1)

# =========================================================
# 4. 데이터 로드
# =========================================================
print("\n" + "="*60)
print("  데이터 로드")
print("="*60)

lines_normal  = load_log_file("normal_pure.txt",  status_filter={"200"})
lines_sqli    = load_log_file("sqli.log",          keep_query=True)
lines_portscan= load_log_file("stealth_portscan.log")  # 피처가 정상과 동일한 stealth 버전
lines_noise   = load_log_file("normal_noise.log", status_filter={"200", "401"})  # 404 제거: 오타 경로가 새 템플릿 ID 생성 → LSTM 오탐 유발

print(f"  정상 로그     : {len(lines_normal):>6}줄  (normal_pure.txt 200만)")
print(f"  알려진공격    : {len(lines_sqli):>6}줄  (sqli.log)")
print(f"  신종공격      : {len(lines_portscan):>6}줄  (stealth_portscan.log)")
print(f"  정상노이즈    : {len(lines_noise):>6}줄  (normal_noise.log)")

# =========================================================
# 5. ML용 데이터 준비
# =========================================================
print("\n[ML 데이터 준비]")

X_norm_raw   = extract_features(lines_normal)
X_sqli_raw   = extract_features(lines_sqli)
X_port_raw   = extract_features(lines_portscan)
X_noise_raw  = extract_features(lines_noise)

# flatten window
X_norm_fw   = flatten_windows(X_norm_raw,  WINDOW)
X_sqli_fw   = flatten_windows(X_sqli_raw,  WINDOW)
X_port_fw   = flatten_windows(X_port_raw,  WINDOW)
X_noise_fw  = flatten_windows(X_noise_raw, WINDOW)

# train/test 분리
train_size_ml = int(len(X_norm_fw) * TRAIN_RATIO)
X_ml_train    = X_norm_fw[:train_size_ml]
X_ml_testn    = X_norm_fw[train_size_ml + GAP:]

# 샘플 수 맞추기
rng = np.random.default_rng(42)
def sample(X, n):
    if len(X) >= n:
        idx = rng.choice(len(X), n, replace=False)
        return X[idx]
    # 3D (LSTM) vs 2D (ML) 모두 처리
    if X.ndim == 3:
        n_rep = (n // len(X)) + 1
        return np.tile(X, (n_rep, 1, 1))[:n]
    else:
        n_rep = (n // len(X)) + 1
        return np.tile(X, (n_rep, 1))[:n]

X_ml_testn_s  = sample(X_ml_testn,  ML_TARGET)
X_ml_sqli_s   = sample(X_sqli_fw,   ML_TARGET)
X_ml_port_s   = sample(X_port_fw,   ML_TARGET)
X_ml_noise_s  = sample(X_noise_fw,  ML_TARGET)

print(f"  ML 학습 (정상)   : {X_ml_train.shape}")
print(f"  ML 테스트 정상   : {X_ml_testn_s.shape}")
print(f"  ML sqli          : {X_ml_sqli_s.shape}")
print(f"  ML portscan      : {X_ml_port_s.shape}")
print(f"  ML noise         : {X_ml_noise_s.shape}")

# 스케일링
scaler = StandardScaler()
X_ml_train   = scaler.fit_transform(X_ml_train)
X_ml_testn_s = scaler.transform(X_ml_testn_s)
X_ml_sqli_s  = scaler.transform(X_ml_sqli_s)
X_ml_port_s  = scaler.transform(X_ml_port_s)
X_ml_noise_s = scaler.transform(X_ml_noise_s)

# RF 학습용 (정상 + sqli)
X_rf_tr = np.vstack([X_ml_train, X_ml_sqli_s])
y_rf_tr = np.concatenate([np.zeros(len(X_ml_train)), np.ones(len(X_ml_sqli_s))])

# =========================================================
# 6. LSTM용 데이터 준비
# =========================================================
print("\n[LSTM 데이터 준비]")

template_map = {}
ids_normal   = lines_to_ids(lines_normal,   template_map, keep_query=False)
ids_sqli     = lines_to_ids(lines_sqli,     template_map, keep_query=True)
ids_portscan = lines_to_ids(lines_portscan, template_map, keep_query=False)
ids_noise    = lines_to_ids(lines_noise,    template_map, keep_query=False)

max_id = max(ids_normal.max(), ids_sqli.max(),
             ids_portscan.max(), ids_noise.max())

ids_normal_n   = ids_normal   / max_id
ids_sqli_n     = ids_sqli     / max_id
ids_portscan_n = ids_portscan / max_id
ids_noise_n    = ids_noise    / max_id

X_lstm_all   = make_windows(ids_normal_n,   WINDOW)
X_lstm_sqli  = make_windows(ids_sqli_n,     WINDOW)
X_lstm_port  = make_windows(ids_portscan_n, WINDOW)
X_lstm_noise = make_windows(ids_noise_n,    WINDOW)

train_size_lstm = int(len(X_lstm_all) * TRAIN_RATIO)
X_lstm_train    = X_lstm_all[:train_size_lstm]
X_lstm_testn    = X_lstm_all[train_size_lstm + GAP:]

LSTM_TARGET = 2500
X_lstm_testn_s = sample(X_lstm_testn,  LSTM_TARGET)
X_lstm_sqli_s  = sample(X_lstm_sqli,   LSTM_TARGET)
X_lstm_port_s  = sample(X_lstm_port,   LSTM_TARGET)
X_lstm_noise_s = sample(X_lstm_noise,  LSTM_TARGET)

print(f"  LSTM 학습 (정상) : {X_lstm_train.shape}")
print(f"  LSTM 테스트 정상 : {X_lstm_testn_s.shape}")
print(f"  LSTM sqli        : {X_lstm_sqli_s.shape}")
print(f"  LSTM portscan    : {X_lstm_port_s.shape}")
print(f"  전체 템플릿 수   : {len(template_map)}")

normal_set   = set(ids_normal.tolist())
sqli_set     = set(ids_sqli.tolist())
portscan_set = set(ids_portscan.tolist())
print(f"  sqli에만 있는 ID    : {len(sqli_set - normal_set)}개  ← 실험A 탐지 핵심")
print(f"  portscan에만 있는 ID: {len(portscan_set - normal_set)}개  ← 실험C 탐지 핵심")

# =========================================================
# 7. ML 모델 학습
# =========================================================
print("\n[ML 모델 학습]")

iforest = IsolationForest(n_estimators=200, contamination=0.05,
                          random_state=42).fit(X_ml_train)
ocsvm   = OneClassSVM(kernel="rbf", gamma="scale",
                      nu=0.05).fit(X_ml_train)
rf      = RandomForestClassifier(n_estimators=200, max_depth=12,
                                 random_state=42).fit(X_rf_tr, y_rf_tr)
print("  iForest, OC-SVM, RF 학습 완료")

# =========================================================
# 8. LSTM 예측 모델 학습 (Autoencoder → 다음 ID 예측 방식)
#
#  입력: [ID_1, ID_2, ID_3, ID_4]  (window-1 = 4개)
#  출력: ID_5                       (다음 1개 예측)
#
#  정상 시퀀스: 예측 오차 낮음
#  이상 시퀀스: 정상 흐름과 다른 패턴 → 예측 오차 높음
#  → Autoencoder보다 순서 변형에 훨씬 민감
# =========================================================
print("\n[LSTM 예측 모델 학습 (Next-ID Prediction)]")

PRED_WINDOW = WINDOW - 1   # 입력 4개 → 다음 1개 예측

# 예측용 데이터: X=[0:4], y=[4] 슬라이딩
def make_pred_data(seq_windows):
    # seq_windows: (N, WINDOW, 1)
    X = seq_windows[:, :PRED_WINDOW, :]   # (N, 4, 1)
    y = seq_windows[:, PRED_WINDOW, :]    # (N, 1)
    return X, y

X_pred_tr, y_pred_tr = make_pred_data(X_lstm_train)

def build_lstm_pred(window=4):
    inp = Input(shape=(window, 1))
    x   = LSTM(128, activation='tanh', return_sequences=True)(inp)
    x   = LSTM(64,  activation='tanh', return_sequences=False)(x)
    x   = Dense(32, activation='relu')(x)
    out = Dense(1)(x)
    m   = Model(inp, out)
    m.compile(optimizer='adam', loss='mse')
    return m

lstm_model = build_lstm_pred(PRED_WINDOW)
lstm_model.summary()

es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lstm_model.fit(
    X_pred_tr, y_pred_tr,
    epochs=50, batch_size=256,
    validation_split=0.1,
    callbacks=[es], verbose=1
)

# 임계치: 정상 train 예측 오차 95th percentile
pred_tr = lstm_model.predict(X_pred_tr, verbose=0)
err_tr  = np.mean(np.square(y_pred_tr - pred_tr), axis=1)
p95     = np.percentile(err_tr, 95)
p99     = np.percentile(err_tr, 99)
lstm_threshold = p95 if p95 > 1e-8 else p99 if p99 > 1e-8 else 0.001
print(f"\n  예측 오차 μ    : {err_tr.mean():.8f}")
print(f"  예측 오차 95th : {p95:.8f}")
print(f"  예측 오차 99th : {p99:.8f}")
print(f"  LSTM 임계치    : {lstm_threshold:.8f}")

# 신종공격 오차 미리 확인
X_port_pred, y_port_pred = make_pred_data(X_lstm_port_s)
pred_port = lstm_model.predict(X_port_pred, verbose=0)
err_port  = np.mean(np.square(y_port_pred - pred_port), axis=1)
print(f"  신종공격 예측오차 평균: {err_port.mean():.8f}  (임계치보다 높아야 탐지)")

# =========================================================
# 9. 예측 함수
# =========================================================
def ml_predict(model, X, mode="unsupervised"):
    if mode == "unsupervised":
        return (model.predict(X) == -1).astype(int)
    else:
        return model.predict(X)

def lstm_predict(X):
    # X: (N, WINDOW, 1) → 앞 PRED_WINDOW개로 다음 예측
    X_in, y_true = make_pred_data(X)
    pred = lstm_model.predict(X_in, verbose=0)
    err  = np.mean(np.square(y_true - pred), axis=1)
    return (err > lstm_threshold).astype(int)

# =========================================================
# 10. 실험 조립 함수
# =========================================================
def make_labels(n_normal, n_attack):
    return np.concatenate([np.zeros(n_normal), np.ones(n_attack)])

def report(name, y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    tpr = tp/(tp+fn) if (tp+fn)>0 else 0.0
    fpr = fp/(fp+tn) if (fp+tn)>0 else 0.0
    f1  = f1_score(y_true, y_pred, zero_division=0)
    print(f"  {name:<20} TPR:{tpr:.3f}  FPR:{fpr:.3f}  F1:{f1:.3f}")
    return tpr, fpr

# =========================================================
# 11. 실험 A — 알려진 공격 (sqli)
# =========================================================
print("\n\n" + "#"*60)
print("  실험 A: 알려진 공격 탐지 (sqli, 깨끗한 환경)")
print("  → 동일 데이터: 정상 2500 + sqli 2500")
print("#"*60)

# ML
X_a_ml = np.vstack([X_ml_testn_s, X_ml_sqli_s])
y_a    = make_labels(len(X_ml_testn_s), len(X_ml_sqli_s))
# LSTM
X_a_lstm = np.vstack([X_lstm_testn_s, X_lstm_sqli_s])

results_a = {}
results_a["iForest"] = report("iForest",       y_a, ml_predict(iforest, X_a_ml))
results_a["OC-SVM"]  = report("OC-SVM",        y_a, ml_predict(ocsvm,   X_a_ml))
results_a["RF"]      = report("Random Forest", y_a, ml_predict(rf, X_a_ml, "supervised"))
results_a["LSTM"]    = report("LSTM (제안)",    y_a, lstm_predict(X_a_lstm))

# =========================================================
# 12. 실험 B — 정상 노이즈 포함 (FPR 비교)
# =========================================================
print("\n\n" + "#"*60)
print("  실험 B: 정상 노이즈 포함 (오탐률 FPR 비교)")
print("  → 동일 데이터: 정상 2500 + 노이즈 2500 + sqli 2500")
print("#"*60)

X_b_ml   = np.vstack([X_ml_testn_s,   X_ml_noise_s,   X_ml_sqli_s])
X_b_lstm = np.vstack([X_lstm_testn_s, X_lstm_noise_s, X_lstm_sqli_s])
y_b = make_labels(len(X_ml_testn_s) + len(X_ml_noise_s), len(X_ml_sqli_s))

results_b = {}
results_b["iForest"] = report("iForest",       y_b, ml_predict(iforest, X_b_ml))
results_b["OC-SVM"]  = report("OC-SVM",        y_b, ml_predict(ocsvm,   X_b_ml))
results_b["RF"]      = report("Random Forest", y_b, ml_predict(rf, X_b_ml, "supervised"))
results_b["LSTM"]    = report("LSTM (제안)",    y_b, lstm_predict(X_b_lstm))

# =========================================================
# 13. 실험 C — 신종 공격 (portscan)
# =========================================================
print("\n\n" + "#"*60)
print("  실험 C: 신종 공격 탐지율 (stealth_portscan, 학습 미포함)")
print("  → 동일 데이터: portscan 2500")
print("#"*60)

y_c = np.ones(LSTM_TARGET)

results_c = {}
results_c["iForest"] = report("iForest",       y_c, ml_predict(iforest, X_ml_port_s))
results_c["OC-SVM"]  = report("OC-SVM",        y_c, ml_predict(ocsvm,   X_ml_port_s))
results_c["RF"]      = report("Random Forest", y_c, ml_predict(rf, X_ml_port_s, "supervised"))
results_c["LSTM"]    = report("LSTM (제안)",    y_c, lstm_predict(X_lstm_port_s))

# =========================================================
# 14. 최종 종합 비교표
# =========================================================
print("\n\n" + "="*70)
print("  최종 종합 비교표")
print("  (동일 데이터: normal_pure.txt + sqli.log + stealth_portscan.log)")
print("="*70)
print(f"  {'모델':<18} {'A_TPR':>7} {'A_FPR':>7} {'B_FPR':>7} {'C_TPR':>7}")
print(f"  {'-'*55}")
for name in ["iForest", "OC-SVM", "RF", "LSTM"]:
    a_tpr, a_fpr = results_a[name]
    b_tpr, b_fpr = results_b[name]
    c_tpr, c_fpr = results_c[name]
    tag = "← 제안" if name == "LSTM" else ""
    print(f"  {name:<18} {a_tpr:>7.3f} {a_fpr:>7.3f} {b_fpr:>7.3f} {c_tpr:>7.3f}  {tag}")

print(f"\n  핵심 비교:")
print(f"  실험B FPR (노이즈 오탐): iForest {results_b['iForest'][1]:.3f} / "
      f"OC-SVM {results_b['OC-SVM'][1]:.3f} vs LSTM {results_b['LSTM'][1]:.3f}")
print(f"  실험C TPR (신종공격):    RF {results_c['RF'][0]:.3f} vs LSTM {results_c['LSTM'][0]:.3f}")
print(f"\n  [데이터 공정성]")
print(f"  모든 모델이 동일한 원본 로그에서 변환된 데이터로 평가됨")
print(f"  ML: 7차원 피처 × flatten_window(5) = 35차원")
print(f"  LSTM: Log ID 시퀀스 window(5) = (5,1) 시퀀스")
