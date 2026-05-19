# normal_numeric_logs_1d.npy, combined_20k_logs_drained.txt, final_ml_test.py는 그대로 유지. attack~.npy는 실험시 폐기
# 3가지 모델에 대한 통합 코드 -> i-forest, oc-svm은 비지도 / random forest는 지도학습

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split

# 1. 데이터 로드 [cite: 147-148]
normal = np.load('normal_numeric_logs_1d.npy')
attack = np.load('attack_numeric_logs_1d.npy')

# 2. 데이터 분할 및 차원 조정 (ML 모델용 2D array)
# 정상 로그 10,000개를 학습에 사용 [cite: 111-140]
train_size = 10000 
X_train_normal = normal[:train_size].reshape(-1, 1)
X_test_normal = normal[train_size:].reshape(-1, 1)

# 제로데이 테스트를 위해 공격 로그 분할 (50%는 '알려진 공격', 50%는 '신종 공격') 
X_atk_known, X_atk_unknown = train_test_split(attack, test_size=0.5, random_state=42)
X_atk_known = X_atk_known.reshape(-1, 1)
X_atk_unknown = X_atk_unknown.reshape(-1, 1)

# 3. 비지도 학습 모델 테스트 (i-Forest, OC-SVM) [cite: 30-31]
print("--- [비지도 학습] 탐지율 측정 시작 ---")
models = {
    "Isolation Forest": IsolationForest(contamination=0.01, random_state=42),
    "One-Class SVM": OneClassSVM(nu=0.01, kernel='rbf')
}

for name, model in models.items():
    model.fit(X_train_normal)
    tnr = np.mean(model.predict(X_test_normal) == 1) * 100 # 정상 탐지율
    tpr = np.mean(model.predict(X_atk_unknown) == -1) * 100 # 신종 공격 탐지율
    print(f"[{name}] 정상 탐지율(TNR): {tnr:.2f}% | 신종 공격 탐지율(TPR): {tpr:.2f}%")

# 4. 지도 학습 모델 테스트 (Random Forest) 
print("\n--- [지도 학습] Random Forest 탐지율 측정 시작 ---")
# 학습 데이터: 정상 로그 + '알려진' 공격 일부만 포함
X_rf_train = np.vstack([X_train_normal, X_atk_known])
y_rf_train = np.concatenate([np.zeros(len(X_train_normal)), np.ones(len(X_atk_known))])

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_rf_train, y_rf_train)

# 결과 산출
tnr_rf = np.mean(rf.predict(X_test_normal) == 0) * 100
tpr_rf_known = np.mean(rf.predict(X_atk_known) == 1) * 100
tpr_rf_unknown = np.mean(rf.predict(X_atk_unknown) == 1) * 100

print(f"[RF] 정상 탐지율: {tnr_rf:.2f}%")
print(f"[RF] 알려진 공격 탐지율: {tpr_rf_known:.2f}%")
print(f"[RF] 신종 공격(Zero-day) 탐지율: {tpr_rf_unknown:.2f}%")
