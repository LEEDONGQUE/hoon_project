import pandas as pd
import numpy as np
import torch
import json

# 이거는 전처리된 20k_drained_logs.txt 파일을 읽어서 LSTM이 읽기 좋게 시퀀스 데이터로 변환하는 코드
# 1. 시퀀스 생성 함수 (태훈님이 올린 코드)
def create_sequences(logs, window_size):
    sequences = []
    targets = []
    for i in range(len(logs) - window_size):
        sequences.append(logs[i:i + window_size])
        targets.append(logs[i + window_size])
    return np.array(sequences), np.array(targets)

def main():
    # 2. 데이터 로드
    file_path = 'combined_20k_logs_drained.txt'
    print(f"[*] {file_path} 로딩 중...")
    
    # Drain 결과 파일 읽기 (보통 EventId 컬럼에 E1, E2 등이 들어있음)
    df = pd.read_csv(file_path)
    
    if 'EventId' not in df.columns:
        print("[!] 에러: 'EventId' 컬럼을 찾을 수 없습니다. 파일의 컬럼명을 확인해주세요.")
        return

    event_ids = df['EventId'].tolist()

    # 3. 문자열 ID를 숫자로 매핑 (E1 -> 0, E2 -> 1 ...)
    unique_ids = sorted(list(set(event_ids)))
    id_map = {id_str: i for i, id_str in enumerate(unique_ids)}
    
    # 나중에 Gemma 리포트용으로 쓸 역매핑 저장
    with open('id_map.json', 'w') as f:
        json.dump(id_map, f)
    
    numeric_logs = [id_map[log] for log in event_ids]
    print(f"[*] 총 로그 라인 수: {len(numeric_logs)}")
    print(f"[*] 고유 템플릿 개수(Vocab Size): {len(unique_ids)}")

    # 4. 슬라이딩 윈도우 생성
    window_size = 5 # 과거 5개를 보고 다음 1개를 예측
    X, y = create_sequences(numeric_logs, window_size)
    
    print(f"[*] 시퀀스 생성 완료!")
    print(f"[*] 입력 데이터(X) 모양: {X.shape}") # (데이터개수, window_size)
    print(f"[*] 정답 데이터(y) 모양: {y.shape}")
    
    # 5. 결과 저장 (나중에 LSTM 학습 코드가 불러다 쓸 수 있게)
    np.save('X_train.npy', X)
    np.save('y_train.npy', y)
    print("[+] 전처리 완료! 'X_train.npy'와 'y_train.npy'가 생성되었습니다.")

if __name__ == "__main__":
    main()
