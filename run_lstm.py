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
df = generate_rvar_from_csv("data/VKOSPI_pred_data.csv")
transformed_df, _ = apply_stationarity_transformation(df)
final_df = pd.concat([df[['Date', 'RVAR']], transformed_df], axis=1).dropna().reset_index(drop=True)

final_df['vkospi_lag_1'] = final_df['VKOSPI'].shift(1)
final_df['vkospi_MA5'] = final_df['VKOSPI'].rolling(window=5).mean().shift(1)
final_df['vkospi_MA22'] = final_df['VKOSPI'].rolling(window=22).mean().shift(1)
final_df.dropna(inplace=True)
final_df.reset_index(drop=True, inplace=True)

# --- 2. 변수 그룹 정의 및 데이터셋 생성 ---
group1 = ['코스피', '코스닥', '회사채(3년, AA-)', '회사채(3년, BBB-)', '국고채(1년)', '국고채(3년)', '국고채(5년)', '국고채(10년)', '국고채(20년)', '국고채(30년)', '통안증권(91일)', '통안증권(1년)', '통안증권(2년)', 'CD(91일)', '콜금리(1일, 전체거래)', 'KORIBOR(3개월)', 'KORIBOR(6개월)', 'KORIBOR(12개월)', 'DGS3MO', 'DGS1', 'DGS5', 'DGS30', 'Dow Jones Index', 'Nasdaq Index', 'S&P500 Index', 'CBOE Volatility Index (VIX)', 'WTI Index', 'Dollar Index', 'USD-KRW Exchange Rate', '금값', '코스피200']
group2 = ['기관 합계', '개인', '외국인 합계', '위탁매매 미수금', '위탁매매 미수금 대비 실제 반대매매금액', '미수금 대비 반대매매비중(%)']
group3 = ['투자자예탁금(장내파생상품  거래예수금제외)', '장내파생상품 거래 예수금', '대고객 환매 조건부 채권(RP) 매도잔고']
group4 = ['vkospi_lag_1', 'vkospi_MA5', 'vkospi_MA22']
groups = [group1, group2, group3, group4]

X_groups_full = [create_lagged_dataset(final_df, group, 'RVAR', window=4)[0] for group in groups]
y_full = create_lagged_dataset(final_df, group1, 'RVAR', window=4)[1]

# --- 3. 롤링 윈도우 교차 검증 설정 ---
initial_train_size = int(len(y_full) * 0.7) # 초기 학습 데이터 비율
test_horizon = 1 # 한 번에 예측할 스텝 수

# 결과를 저장할 딕셔너리
all_predictions = {
    'true': [],
    'multi_mse': [],
    'std_mse': [],
    'multi_mae': [],
    'std_mae': []
}

# 각 모델의 성능 지표를 저장할 리스트
metrics = {
    'multi_mse': {'mse': [], 'mae': []},
    'std_mse': {'mse': [], 'mae': []},
    'multi_mae': {'mse': [], 'mae': []},
    'std_mae': {'mse': [], 'mae': []}
}

print("Starting Rolling Window Cross-Validation...")

for i in range(initial_train_size, len(y_full) - test_horizon + 1):
    train_end_idx = i
    test_start_idx = i
    test_end_idx = i + test_horizon

    # 현재 윈도우의 데이터 분할
    X_groups_train_fold = [X[:train_end_idx] for X in X_groups_full]
    X_groups_test_fold = [X[test_start_idx:test_end_idx] for X in X_groups_full]
    y_train_fold = y_full[:train_end_idx]
    y_test_fold = y_full[test_start_idx:test_end_idx]

    # Standard LSTM을 위한 데이터 결합
    X_train_std_fold = np.concatenate(X_groups_train_fold, axis=2)
    X_test_std_fold = np.concatenate(X_groups_test_fold, axis=2)

    # 실제 값 저장
    all_predictions['true'].extend(y_test_fold.flatten())

    # --- 4. 모델 학습 및 예측 ---
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    # Multi-Input LSTM (MSE)
    model_multi_mse = build_multi_input_lstm1([X.shape[1:] for X in X_groups_train_fold], loss_function='mse')
    model_multi_mse.fit(X_groups_train_fold, y_train_fold, epochs=500, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)
    y_pred_multi_mse = model_multi_mse.predict(X_groups_test_fold)
    all_predictions['multi_mse'].extend(y_pred_multi_mse.flatten())
    metrics['multi_mse']['mse'].append(mean_squared_error(y_test_fold, y_pred_multi_mse))
    metrics['multi_mse']['mae'].append(mean_absolute_error(y_test_fold, y_pred_multi_mse))

    # Standard LSTM (MSE)
    model_std_mse = build_lstm(X_train_std_fold.shape[1:], loss_function='mse')
    model_std_mse.fit(X_train_std_fold, y_train_fold, epochs=500, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)
    y_pred_std_mse = model_std_mse.predict(X_test_std_fold)
    all_predictions['std_mse'].extend(y_pred_std_mse.flatten())
    metrics['std_mse']['mse'].append(mean_squared_error(y_test_fold, y_pred_std_mse))
    metrics['std_mse']['mae'].append(mean_absolute_error(y_test_fold, y_pred_std_mse))

    # Multi-Input LSTM (MAE)
    model_multi_mae = build_multi_input_lstm1([X.shape[1:] for X in X_groups_train_fold], loss_function='mae')
    model_multi_mae.fit(X_groups_train_fold, y_train_fold, epochs=500, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)
    y_pred_multi_mae = model_multi_mae.predict(X_groups_test_fold)
    all_predictions['multi_mae'].extend(y_pred_multi_mae.flatten())
    metrics['multi_mae']['mse'].append(mean_squared_error(y_test_fold, y_pred_multi_mae))
    metrics['multi_mae']['mae'].append(mean_absolute_error(y_test_fold, y_pred_multi_mae))

    # Standard LSTM (MAE)
    model_std_mae = build_lstm(X_train_std_fold.shape[1:], loss_function='mae')
    model_std_mae.fit(X_train_std_fold, y_train_fold, epochs=500, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)
    y_pred_std_mae = model_std_mae.predict(X_test_std_fold)
    all_predictions['std_mae'].extend(y_pred_std_mae.flatten())
    metrics['std_mae']['mse'].append(mean_squared_error(y_test_fold, y_pred_std_mae))
    metrics['std_mae']['mae'].append(mean_absolute_error(y_test_fold, y_pred_std_mae))

    print(f"Fold {i - initial_train_size + 1} completed.")

print("Rolling Window Cross-Validation Completed.")

# --- 5. 최종 성능 지표 출력 ---
print("\n--- Average Performance Metrics (Rolling Window) ---")
for model_name, model_metrics in metrics.items():
    avg_mse = np.mean(model_metrics['mse'])
    avg_mae = np.mean(model_metrics['mae'])
    print(f"{model_name.replace('_', ' ').title()}: Average MSE = {avg_mse:.4f}, Average MAE = {avg_mae:.4f}")

# --- 6. 예측 결과 시각화 ---
# MSE 모델 예측 결과 시각화
plt.figure(figsize=(14, 7))
plt.plot(all_predictions['true'], label='True RVAR', color='black', linewidth=2)
plt.plot(all_predictions['multi_mse'], label='Multi-Input LSTM (MSE) Prediction', color='red', linestyle='-')
plt.plot(all_predictions['std_mse'], label='Standard LSTM (MSE) Prediction', color='blue', linestyle='--')
plt.title('RVAR Prediction Comparison (MSE Loss Models)')
plt.xlabel('Time Step')
plt.ylabel('RVAR')
plt.legend()
plt.tight_layout()
plt.savefig('prediction_comparison_mse_rolling_window.png', dpi =500)
plt.show()

# MAE 모델 예측 결과 시각화
plt.figure(figsize=(14, 7))
plt.plot(all_predictions['true'], label='True RVAR', color='black', linewidth=2)
plt.plot(all_predictions['multi_mae'], label='Multi-Input LSTM (MAE) Prediction', color='green', linestyle='-')
plt.plot(all_predictions['std_mae'], label='Standard LSTM (MAE) Prediction', color='purple', linestyle='--')
plt.title('RVAR Prediction Comparison (MAE Loss Models)')
plt.xlabel('Time Step')
plt.ylabel('RVAR')
plt.legend()
plt.tight_layout()
plt.savefig('prediction_comparison_mae_rolling_window.png', dpi =500)
plt.show()
