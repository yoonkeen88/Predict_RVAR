import pandas as pd
from preprocessing.rvar_generator import generate_rvar_from_csv
from preprocessing.stationarity_transform import apply_stationarity_transformation
from preprocessing.make_lstm_dataset import create_lagged_dataset
from models.lstm_model import build_multi_input_lstm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np

# 1. RVAR 및 log_return 생성
df = generate_rvar_from_csv("data/VKOSPI_pred_data.csv")

# 2. 정상시계열 변환
transformed_df, transformations = apply_stationarity_transformation(df)

# 3. 최종 데이터프레임 구성
final_df = pd.concat([df[['Date', 'RVAR']], transformed_df], axis=1).dropna().reset_index(drop=True)

# 4. RVAR 시차 및 이동평균 변수 생성 (논문 기반)
final_df['RVAR_lag_1'] = final_df['RVAR'].shift(1)
final_df['RVAR_MA5'] = final_df['RVAR'].rolling(window=5).mean().shift(1)
final_df['RVAR_MA22'] = final_df['RVAR'].rolling(window=22).mean().shift(1)
final_df.dropna(inplace=True)
final_df.reset_index(drop=True, inplace=True)


# 5. 변수 그룹 정의 (논문 기반)
group1 = ['코스피', '코스닥', '회사채(3년, AA-)', '회사채(3년, BBB-)', '국고채(1년)', '국고채(3년)', '국고채(5년)', '국고채(10년)', '국고채(20년)', '국고채(30년)', '통안증권(91일)', '통안증권(1년)', '통안증권(2년)', 'CD(91일)', '콜금리(1일, 전체거래)', 'KORIBOR(3개월)', 'KORIBOR(6개월)', 'KORIBOR(12개월)', 'DGS3MO', 'DGS1', 'DGS5', 'DGS30', 'Dow Jones Index', 'Nasdaq Index', 'S&P500 Index', 'CBOE Volatility Index (VIX)', 'WTI Index', 'Dollar Index', 'USD-KRW Exchange Rate', '금값', '코스피200']
group2 = ['기관 합계', '개인', '외국인 합계', '위탁매매 미수금', '위탁매매 미수금 대비 실제 반대매매금액', '미수금 대비 반대매매비중(%)']
group3 = ['투자자예탁금(장내파생상품  거래예수금제외)', '장내파생상품 거래 예수금', '대고객 환매 조건부 채권(RP) 매도잔고']
group4 = ['RVAR_lag_1', 'RVAR_MA5', 'RVAR_MA22']

groups = [group1, group2, group3, group4]

# 5. 각 그룹별 시차 데이터 생성
X_groups = []
for group in groups:
    X, y = create_lagged_dataset(final_df, group, 'RVAR', window=4)
    X_groups.append(X)

# y는 동일하므로 한 번만 추출
_, y = create_lagged_dataset(final_df, group1, 'RVAR', window=4)

# 6. 데이터 분할
X_groups_train = [X[:int(0.8*len(X))] for X in X_groups]
X_groups_test = [X[int(0.8*len(X)):] for X in X_groups]
y_train, y_test = y[:int(0.8*len(y))], y[int(0.8*len(y)):]

# 7. 모델 학습
input_shapes = [X.shape[1:] for X in X_groups]
model = build_multi_input_lstm(input_shapes)
from tensorflow.keras.callbacks import EarlyStopping

# ... (기존 코드) ...

# 7. 모델 학습
input_shapes = [X.shape[1:] for X in X_groups]
model = build_multi_input_lstm(input_shapes)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(X_groups_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# 학습 과정 시각화
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.title("Training and Validation Loss")
plt.show()

# 8. 예측 및 평가
y_pred = model.predict(X_groups_test)

from sklearn.metrics import mean_squared_error, mean_absolute_error
print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

# 9. 시각화
plt.plot(y_test, label='True')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.title("LSTM Estimation of RVAR")
plt.show()