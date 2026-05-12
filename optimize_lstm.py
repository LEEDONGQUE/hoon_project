import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import csv
import time

# ==========================================
# 0. 이전 코드와 동일한 모델 및 함수 정의 (생략/복사 필요)
# (LSTMLM 클래스, create_sequences, compute_ppl 등)
# ==========================================

def main():
    # 데이터 로드
    normal_data = np.load('/home/student/log_project/normal_numeric_logs_1d.npy')
    attack_data = np.load('/home/student/log_project/attack_numeric_logs_1d.npy')
    with open('/home/student/log_project/id_map.json', 'r') as f:
        vocab_size = len(json.load(f))
        
    # 고정 변수
    TRAIN_SIZE = 13000
    WINDOW_SIZE = 5

    # 🌟 탐색할 변수 후보군 (원하시는 값으로 수정 가능)
    val_sizes = [1000, 2000, 3000]       # 검증 데이터 크기
    epochs_list = [5, 10, 20]            # 학습 횟수
    lr_list = [0.001, 0.0005, 0.0001]    # 학습률

    results = []
    
    # 3중 for문 시작
    for v_size in val_sizes:
        for ep in epochs_list:
            for lr in lr_list:
                print(f"[*] 테스트 시작: Val_Size={v_size}, Epochs={ep}, LR={lr}")
                start_time = time.time()
                
                # 1. 데이터 분할
                train_slice = normal_data[:TRAIN_SIZE]
                val_slice = normal_data[TRAIN_SIZE : TRAIN_SIZE + v_size]
                test_normal_slice = normal_data[TRAIN_SIZE + v_size :] # 남은 정상 로그 전부
                
                # 2. 시퀀스 생성 (Window = 5)
                X_train, y_train = create_sequences(train_slice, WINDOW_SIZE)
                X_val, y_val = create_sequences(val_slice, WINDOW_SIZE)
                X_test_n, y_test_n = create_sequences(test_normal_slice, WINDOW_SIZE)
                X_test_a, y_test_a = create_sequences(attack_data, WINDOW_SIZE)
                
                # 3. 모델 초기화 (매 루프마다 완전히 새로운 모델로 리셋!)
                model = LSTMLM(vocab_size)
                
                # 4. 모델 학습 (train_model 함수는 기존 코드 사용)
                # (주의: train_model이 학습된 모델을 반환하도록 설정되어 있어야 함)
                model = train_model(model, X_train, y_train, epochs=ep, lr=lr)
                
                # 5. 검증 데이터로 PPL 및 임계치(m+3sigma) 계산
                val_ppls = compute_ppl_for_batch(model, X_val, y_val) # 기존 함수명에 맞춤
                m, sigma = np.mean(val_ppls), np.std(val_ppls)
                threshold = m + (3 * sigma)
                
                # 6. 정상 테스트 데이터 평가 (오탐율 FPR)
                test_n_ppls = compute_ppl_for_batch(model, X_test_n, y_test_n)
                fp = np.sum(test_n_ppls > threshold)
                fpr = (fp / len(test_n_ppls)) * 100
                
                # 7. 공격 테스트 데이터 평가 (정탐율 TPR)
                test_a_ppls = compute_ppl_for_batch(model, X_test_a, y_test_a)
                tp = np.sum(test_a_ppls > threshold)
                tpr = (tp / len(test_a_ppls)) * 100
                
                run_time = time.time() - start_time
                print(f"  -> 결과: TPR(정탐) {tpr:.2f}% | FPR(오탐) {fpr:.2f}% | 걸린시간: {run_time:.1f}초")
                
                # 결과 기록
                results.append({
                    'Train_Size': TRAIN_SIZE, 'Window_Size': WINDOW_SIZE,
                    'Val_Size': v_size, 'Epochs': ep, 'LearningRate': lr,
                    'Threshold': threshold, 'TPR(%)': tpr, 'FPR(%)': fpr
                })

    # CSV로 결과 저장
    csv_file = '/home/student/log_project/hyperparam_optimization_results.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[+] 모든 실험 완료! 결과가 {csv_file} 에 저장되었습니다.")

if __name__ == "__main__":
    main()

