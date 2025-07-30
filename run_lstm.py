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

X_groups = [create_lagged_dataset(final_df, group, 'RVAR', window=4)[0] for group in groups]
_, y = create_lagged_dataset(final_df, group1, 'RVAR', window=4)

# --- 3. 데이터 분할 ---
train_size = int(0.8 * len(y))
X_groups_train = [X[:train_size] for X in X_groups]
X_groups_test = [X[train_size:] for X in X_groups]
y_train, y_test = y[:train_size], y[train_size:]

early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# --- 4. Multi-Input LSTM 모델 학습 및 평가 ---
print("--- Training Multi-Input LSTM Model ---")
multi_input_shapes = [X.shape[1:] for X in X_groups_train]
model_multi = build_multi_input_lstm1(multi_input_shapes)

history_multi = model_multi.fit(
    X_groups_train, y_train,
    epochs=500,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

y_pred_multi = model_multi.predict(X_groups_test)
print("Multi-Input LSTM MSE:", mean_squared_error(y_test, y_pred_multi))
print("Multi-Input LSTM MAE:", mean_absolute_error(y_test, y_pred_multi))
print("-" * 40)


# --- 5. Standard LSTM 모델 학습 및 평가 ---
print("--- Training Standard LSTM Model ---")
# 일반 LSTM을 위해 데이터셋 결합
X_train_std = np.concatenate(X_groups_train, axis=2)
X_test_std = np.concatenate(X_groups_test, axis=2)

model_std = build_lstm(X_train_std.shape[1:])

history_std = model_std.fit(
    X_train_std, y_train,
    epochs=500,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

y_pred_std = model_std.predict(X_test_std)
print("Standard LSTM MSE:", mean_squared_error(y_test, y_pred_std))
print("Standard LSTM MAE:", mean_absolute_error(y_test, y_pred_std))
print("-" * 40)


# --- 6. 학습 과정 및 예측 결과 시각화 ---
# Loss 시각화
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(history_multi.history['loss'], label='Multi-Input Train Loss')
plt.plot(history_multi.history['val_loss'], label='Multi-Input Val Loss')
plt.plot(history_std.history['loss'], label='Standard Train Loss', linestyle='--')
plt.plot(history_std.history['val_loss'], label='Standard Val Loss', linestyle='--')
plt.title('Model Training & Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 예측 결과 시각화
plt.subplot(1, 2, 2)
plt.plot(y_test, label='True RVAR', color='black', linewidth=2)
plt.plot(y_pred_multi, label='Multi-Input LSTM Prediction', color='red', linestyle='-')
plt.plot(y_pred_std, label='Standard LSTM Prediction', color='blue', linestyle='--')
plt.title('RVAR Prediction Comparison')
plt.xlabel('Time')
plt.ylabel('RVAR')
plt.legend()

plt.tight_layout()
plt.savefig('Std_LSTM_VS_Multi-Input_LSTM_training_results.png')
plt.show()
