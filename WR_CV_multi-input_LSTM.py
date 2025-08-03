import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.callbacks import EarlyStopping

from preprocessing.rvar_generator import generate_rvar_from_csv
from preprocessing.stationarity_transform import apply_stationarity_transformation
from preprocessing.make_lstm_dataset import create_lagged_dataset
from models.lstm_model import build_multi_input_lstm1, build_lstm

# --- 1. 데이터 준비 ---
# 예측 대상을 RVAR로 설정합니다.
df = generate_rvar_from_csv("data/merged_data.csv").dropna()

# stationarity_transform에서 예측 대상인 RVAR은 변환하지 않도록 제외합니다.
transformed_df, _ = apply_stationarity_transformation(df, exclude_cols=['Date', 'log_return', 'RVAR'])

# 변환된 컬럼들과 원본 데이터프레임을 합칩니다.
# 먼저, 원본 df에서 변환이 일어난 컬럼들을 삭제합니다.
df_untransformed = df.drop(columns=transformed_df.columns)
# 인덱스를 기준으로 변환된 데이터프레임과 합칩니다.
final_df = pd.merge(df_untransformed, transformed_df, left_index=True, right_index=True)

# Group 4 피처 생성 (RVAR 기반)
final_df['RVAR_lag_1'] = final_df['RVAR'].shift(1)
final_df['RVAR_MA5'] = final_df['RVAR'].rolling(window=5).mean().shift(1)
final_df['RVAR_MA22'] = final_df['RVAR'].rolling(window=22).mean().shift(1)
final_df.dropna(inplace=True)
final_df.reset_index(drop=True, inplace=True)


# --- 2. 변수 그룹 정의 및 데이터셋 생성 ---
group1 = ['코스피', '코스닥', '회사채(3년, AA-)', '회사채(3년, BBB-)', '국고채(1년)', '국고채(3년)', '국고채(5년)', '국고채(10년)', '국고채(20년)', '국고채(30년)', '통안증권(91일)', '통안증권(1년)', '통안증권(2년)', 'CD(91일)', '콜금리(1일, 전체거래)', 'KORIBOR(3개월)', 'KORIBOR(6개월)', 'KORIBOR(12개월)', 'DGS3MO', 'DGS1', 'DGS5', 'DGS30', 'Dow Jones Index', 'Nasdaq Index', 'S&P500 Index', 'CBOE Volatility Index (VIX)', 'WTI Index', 'Dollar Index', 'USD-KRW Exchange Rate', '금값', '코스피200']
group2 = ['기관 합계', '개인', '외국인 합계', '위탁매매 미수금', '위탁매매 미수금 대비 실제 반대매매금액', '미수금 대비 반대매매비중(%)']
group3 = ['투자자예탁금(장내파생상품  거래예수금제외)', '장내파생상품 거래 예수금', '대고객 환매 조건부 채권(RP) 매도잔고']
group4 = ['RVAR_lag_1', 'RVAR_MA5', 'RVAR_MA22']
groups = [group1, group2, group3, group4]

X_groups_full = [create_lagged_dataset(final_df, group, 'RVAR', window=22)[0] for group in groups]
y_full = create_lagged_dataset(final_df, group1, 'RVAR', window=22)[1]

# --- 3. 롤링 윈도우 교차 검증 설정 ---
initial_train_size = int(len(y_full) * 0.8) # 초기 학습 데이터 비율
test_horizon = 3 # 한 번에 예측할 스텝 수

# 결과를 저장할 딕셔너리
all_predictions = {
    'true': [],
    'multi_mse': []
}

# 각 모델의 성능 지표를 저장할 리스트
metrics = {
    'multi_mse': {'mse': [], 'mae': []}
}

print("Starting Rolling Window Cross-Validation for Multi-Input LSTM (MSE)...")

for i in range(initial_train_size, len(y_full) - test_horizon + 1):
    train_end_idx = i
    test_start_idx = i
    test_end_idx = i + test_horizon

    # 현재 윈도우의 데이터 분할
    X_groups_train_fold = [X[:train_end_idx] for X in X_groups_full]
    X_groups_test_fold = [X[test_start_idx:test_end_idx] for X in X_groups_full]
    y_train_fold = y_full[:train_end_idx]
    y_test_fold = y_full[test_start_idx:test_end_idx]

    # # Standard LSTM을 위한 데이터 결합
    # X_train_std_fold = np.concatenate(X_groups_train_fold, axis=2)
    # X_test_std_fold = np.concatenate(X_groups_test_fold, axis=2)

    # 실제 값 저장
    all_predictions['true'].extend(y_test_fold.flatten())

    # --- 4. 모델 학습 및 예측 ---
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Multi-Input LSTM (MSE)
    model_multi_mse = build_multi_input_lstm1([X.shape[1:] for X in X_groups_train_fold], loss_function='mse')
    model_multi_mse.fit(X_groups_train_fold, y_train_fold, epochs=500, batch_size=64, validation_split=0.3, callbacks=[early_stopping], verbose=1)
    y_pred_multi_mse = model_multi_mse.predict(X_groups_test_fold)
    all_predictions['multi_mse'].extend(y_pred_multi_mse.flatten())
    metrics['multi_mse']['mse'].append(mean_squared_error(y_test_fold, y_pred_multi_mse))
    metrics['multi_mse']['mae'].append(mean_absolute_error(y_test_fold, y_pred_multi_mse))

    # # Standard LSTM (MSE)
    # model_std_mse = build_lstm(X_train_std_fold.shape[1:], loss_function='mse')
    # model_std_mse.fit(X_train_std_fold, y_train_fold, epochs=500, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)
    # y_pred_std_mse = model_std_mse.predict(X_test_std_fold)
    # all_predictions['std_mse'].extend(y_pred_std_mse.flatten())
    # metrics['std_mse']['mse'].append(mean_squared_error(y_test_fold, y_pred_std_mse))
    # metrics['std_mse']['mae'].append(mean_absolute_error(y_test_fold, y_pred_std_mse))

    # # Multi-Input LSTM (MAE)
    # model_multi_mae = build_multi_input_lstm1([X.shape[1:] for X in X_groups_train_fold], loss_function='mae')
    # model_multi_mae.fit(X_groups_train_fold, y_train_fold, epochs=500, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)
    # y_pred_multi_mae = model_multi_mae.predict(X_groups_test_fold)
    # all_predictions['multi_mae'].extend(y_pred_multi_mae.flatten())
    # metrics['multi_mae']['mse'].append(mean_squared_error(y_test_fold, y_pred_multi_mae))
    # metrics['multi_mae']['mae'].append(mean_absolute_error(y_test_fold, y_pred_multi_mae))

    # # Standard LSTM (MAE)
    # model_std_mae = build_lstm(X_train_std_fold.shape[1:], loss_function='mae')
    # model_std_mae.fit(X_train_std_fold, y_train_fold, epochs=500, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)
    # y_pred_std_mae = model_std_mae.predict(X_test_std_fold)
    # all_predictions['std_mae'].extend(y_pred_std_mae.flatten())
    # metrics['std_mae']['mse'].append(mean_squared_error(y_test_fold, y_pred_std_mae))
    # metrics['std_mae']['mae'].append(mean_absolute_error(y_test_fold, y_pred_std_mae))

    print(f"Fold {i - initial_train_size + 1} completed.")

print("Rolling Window Cross-Validation Completed.")

# --- 5. 최종 성능 지표 출력 ---
print("\n--- Average Performance Metrics (Rolling Window) ---")
avg_mse = np.mean(metrics['multi_mse']['mse'])
avg_mae = np.mean(metrics['multi_mse']['mae'])
print(f"Multi-Input LSTM (MSE): Average MSE = {avg_mse:.4f}, Average MAE = {avg_mae:.4f}")

# --- 6. 예측 결과 시각화 ---
plt.figure(figsize=(14, 7))
plt.plot(all_predictions['true'], label='True KOSPI', color='black', linewidth=2)
plt.plot(all_predictions['multi_mse'], label='Multi-Input LSTM (MSE) Prediction', color='red', linestyle='-')
plt.title('KOSPI Prediction with Multi-Input LSTM (MSE) - Rolling Window')
plt.xlabel('Time Step')
plt.ylabel('KOSPI')
plt.legend()
plt.tight_layout()
plt.savefig('prediction_multi_input_mse_rolling_window.png', dpi=500)
plt.show()
