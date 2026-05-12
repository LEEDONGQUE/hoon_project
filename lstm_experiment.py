import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
import os
import csv
# ==========================================
# 1. 모델 및 핵심 함수 정의
# ==========================================

class LSTMLM(nn.Module):
    def __init__(self, vocab_size, embed_size=64, hidden_size=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        # batch_first=True: 입력 데이터의 형태가 (Batch, Sequence, Feature)임을 명시
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x):
        x = self.embed(x)
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

def create_sequences(logs, window_size):
    """1차원 배열을 받아 지정된 window_size에 맞게 슬라이딩 윈도우(2D) 형태로 자름"""
    if len(logs) <= window_size:
        return torch.LongTensor([]), torch.LongTensor([])
        
    X, y = [], []
    for i in range(len(logs) - window_size):
        X.append(logs[i:i + window_size])
        y.append(logs[i + window_size])
    return torch.LongTensor(np.array(X)), torch.LongTensor(np.array(y))

def compute_ppl_for_batch(model, X_batch, y_batch):
    """배열 전체의 PPL(Perplexity)을 계산하여 반환"""
    if len(X_batch) == 0:
        return np.array([])
        
    model.eval()
    device = next(model.parameters()).device
    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)
    
    with torch.no_grad():
        output = model(X_batch)
        prediction = output[:, -1, :] # 마지막 시점의 예측값만 사용
        loss = F.cross_entropy(prediction, y_batch, reduction='none')
        ppl = torch.exp(loss)
    return ppl.cpu().numpy()

def train_model(model, X_train, y_train, epochs=5, lr=0.001):
    """LSTM 모델 학습"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train)
        prediction = output[:, -1, :]
        loss = criterion(prediction, y_train)
        loss.backward()
        optimizer.step()
    return model

# ==========================================
# 2. 메인 실험 파이프라인 (이중 FOR문)
# ==========================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 다중 변수 기반 LSTM 이상 탐지 성능 평가 실험 시작")
    print("="*70)

    # 🌟 [경로 설정] lee 폴더의 절대 경로를 입력합니다.
    # 만약 딥러닝 PC의 구조가 다르면 '/home/student/log_project/lee/' 처럼 수정해주세요!
    BASE_DIR = '/home/midori/lee/' 
    
    id_map_path = os.path.join(BASE_DIR, 'id_map.json')
    normal_data_path = os.path.join(BASE_DIR, 'normal_numeric_logs_1d.npy')
    attack_data_path = os.path.join(BASE_DIR, 'attack_numeric_logs_1d.npy')

    # 1. 단어장(Vocab) 사이즈 로드
    try:
        with open(id_map_path, 'r', encoding='utf-8') as f:
            id_map = json.load(f)
        vocab_size = len(id_map)
        print(f"[*] 기준 단어장 로드 완료 (Vocab Size: {vocab_size})")
    except FileNotFoundError:
        print(f"[!] 에러: {id_map_path} 파일이 없습니다.")
        exit()

    # 2. 1차원 전처리 데이터 로드
    try:
        # 정상 로그 로드 및 분할 (학습 풀 18,000 / 정상 테스트 2,000)
        all_normal_logs = np.load(normal_data_path)
        raw_train_pool = all_normal_logs[:-2000] 
        raw_test_normal = all_normal_logs[-2000:] 
        
        # 공격 로그 로드 (공격 테스트 200)
        raw_test_attack = np.load(attack_data_path)
        
        print(f"[*] 데이터 로드 완료!")
        print(f"    - 최대 학습 가용 로그: {len(raw_train_pool)}줄")
        print(f"    - 정상 테스트 로그: {len(raw_test_normal)}줄")
        print(f"    - 공격 테스트 로그: {len(raw_test_attack)}줄")
    except FileNotFoundError as e:
        print(f"[!] 데이터 파일을 찾을 수 없습니다: {e}")
        exit()

    # 3. 실험 변수 세팅
    data_sizes = [3000, 9000, 13000, 18000] # 외부 FOR문
    window_sizes = [3, 5, 7, 10]            # 내부 FOR문
    val_ratio = 0.20                        # 검증 데이터 비율 (20%)
    
    # 💡 팁: 코드가 잘 도는지 테스트할 때는 5, 실제 논문용 데이터를 뽑을 때는 20~30으로 올리세요!
    epochs_per_exp = 5                      

    results_summary = [] # 최종 결과 표 출력을 위한 리스트

    print("\n" + "-"*70)
    print("⏳ 모델 학습 및 평가 진행 중... (시간이 다소 소요됩니다)")
    print("-"*70)

    # 🔄 [외부 FOR문]: 데이터 양 증가
    for size in data_sizes:
        print(f"\n▶ 데이터 크기: {size}줄 실험 시작")
        
        current_data = raw_train_pool[:size]
        split_idx = int(len(current_data) * (1 - val_ratio))
        train_raw = current_data[:split_idx]
        val_raw = current_data[split_idx:]
        
        # 🔄 [내부 FOR문]: 윈도우 크기 증가
        for w in window_sizes:
            start_time = time.time()
            
            # 🌟 [초기화] 이전 실험의 가중치(기억) 완벽 삭제
            model = LSTMLM(vocab_size)
            
            # 🌟 [동적 자르기] 현재 루프의 w값에 맞춰 실시간으로 시퀀스화
            X_train, y_train = create_sequences(train_raw, w)
            X_val, y_val = create_sequences(val_raw, w)
            X_test_normal, y_test_normal = create_sequences(raw_test_normal, w)
            X_test_attack, y_test_attack = create_sequences(raw_test_attack, w)
            
            # 학습 진행
            model = train_model(model, X_train, y_train, epochs=epochs_per_exp)
            
            # 임계치 산정 (m + 3*sigma) - 검증 데이터(Val) 기준
            val_ppls = compute_ppl_for_batch(model, X_val, y_val)
            if len(val_ppls) > 0:
                m = np.mean(val_ppls)
                sigma = np.std(val_ppls)
                threshold = m + (3 * sigma)
            else:
                threshold = 0.0

            # 🧪 테스트 1: 정상 로그 (정상을 정상으로 판단 -> PPL <= 임계치)
            normal_ppls = compute_ppl_for_batch(model, X_test_normal, y_test_normal)
            if len(normal_ppls) > 0:
                true_negatives = np.sum(normal_ppls <= threshold)
                tnr_rate = (true_negatives / len(normal_ppls)) * 100
            else:
                true_negatives, tnr_rate = 0, 0.0

            # 🧪 테스트 2: 공격 로그 (공격을 공격으로 판단 -> PPL > 임계치)
            attack_ppls = compute_ppl_for_batch(model, X_test_attack, y_test_attack)
            if len(attack_ppls) > 0:
                true_positives = np.sum(attack_ppls > threshold)
                tpr_rate = (true_positives / len(attack_ppls)) * 100
            else:
                true_positives, tpr_rate = 0, 0.0
                
            elapsed_time = time.time() - start_time
            print(f"  └ Win Size {w:2d} 완료 ({elapsed_time:.1f}초) | 임계치: {threshold:.2f} | TNR(정상탐지): {tnr_rate:5.1f}% | TPR(공격탐지): {tpr_rate:5.1f}%")
            
            # 결과 저장
            results_summary.append({
                'size': size, 'window': w, 'threshold': threshold,
                'tn': true_negatives, 'tn_total': len(normal_ppls), 'tn_rate': tnr_rate,
                'tp': true_positives, 'tp_total': len(attack_ppls), 'tp_rate': tpr_rate
            })

    # ==========================================
    # 3. 최종 결과 요약 표 출력 (논문 첨부용)
    # ==========================================
    print("\n" + "="*90)
    print(f"{'Data Size':<10} | {'Win Size':<8} | {'Threshold':<10} | {'Normal Accuracy (TNR)':<25} | {'Attack Detect Rate (TPR)':<25}")
    print("-" * 90)
    for res in results_summary:
        tn_str = f"{res['tn']}/{res['tn_total']} ({res['tn_rate']:.1f}%)"
        tp_str = f"{res['tp']}/{res['tp_total']} ({res['tp_rate']:.1f}%)"
        print(f"{res['size']:<10} | {res['window']:<8} | {res['threshold']:<10.2f} | {tn_str:<25} | {tp_str:<25}")
    print("="*90)
    print("✅ 모든 다중 변수 평가 실험 완료!")

    csv_file_path = os.path.join(BASE_DIR, 'lstm_experiment_results.csv')
    
    # encoding='utf-8-sig'를 쓰면 엑셀에서 열 때 한글이 깨지지 않습니다!
    with open(csv_file_path, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        
        # 엑셀 첫 번째 줄(헤더) 작성
        writer.writerow(['데이터 크기(Data Size)', '윈도우 크기(Window Size)', '임계치(Threshold)', '정상 탐지율(TNR %)', '공격 탐지율(TPR %)', '정상 맞춘 개수', '공격 맞춘 개수'])
        
        # 실제 데이터 줄줄이 쓰기
        for res in results_summary:
            writer.writerow([
                res['size'], 
                res['window'], 
                f"{res['threshold']:.4f}",
                f"{res['tn_rate']:.1f}", 
                f"{res['tp_rate']:.1f}",
                f"{res['tn']} / {res['tn_total']}", 
                f"{res['tp']} / {res['tp_total']}"
            ])
            
    print(f"📁 엑셀로 열어볼 수 있는 결과 파일이 저장되었습니다: {csv_file_path}")
