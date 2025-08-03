# 📈 Time Series RVAR Prediction Project (Multi-Model Ready)

이 프로젝트는 실현변동성(RVAR: Realized Variance)을 예측하기 위해 다양한 시계열 모델 (LSTM, HAR, XGBoost, RandomForest 등)을 적용하기 위한 **공통 전처리 기반의 예측 시스템**입니다.  
모든 모델이 동일한 입력 데이터를 사용하도록 **정상 시계열 변환 로직을 통일**하였으며, 현재 이 저장소는 `LSTM` 모델에 중점을 두고 있습니다.

---

## 📁 디렉토리 구조 및 파일 설명

```
.
├── data/
│   └── VKOSPI_pred_data.csv         # 원본 입력 데이터 (일별 경제/시장 지표 포함)
├── preprocessing/
│   ├── stationarity_transform.py    # 정상성 검정 및 자동 변환 모듈
│   ├── rvar_generator.py            # 로그 수익률 및 RVAR 계산기
│   └── make_lstm_dataset.py         # LSTM용 시차 데이터셋 생성기
├── models/
│   └── lstm_model.py                # LSTM 모델 구조 정의 (일반 및 Multi-Input)
├── README.md                        # 본 문서
├── run_lstm.py                      # 데이터 전처리 및 모델 학습-평가, 롤링 윈도우 교차 검증
├── conda_requirements.txt           # 본 프로젝트 실행을 위한 라이브러리 정보
├── GEMINI.md                        # Gemini CLI 세션 기록 및 컨텍스트
```

---

## 정상 시계열 변환 로직

모든 설명 변수는 다음과 같은 순서로 자동 변환됩니다:

1. **로그 변환**: 모든 값이 양수이고, log 후 정상성이 확인되면 사용
2. **로그 후 차분**: log 변환이 비정상인 경우 1차 차분 수행
3. **단순 차분**: 로그 불가능하거나 위 방법이 실패할 경우 적용
4. **정상성 검정**: ADF(p < 0.05) + KPSS(p > 0.05) 조건 만족 시 통과

> 모든 변수는 동일한 로직을 적용하여 **모델 간 결과 비교의 공정성**을 확보합니다.

사용된 함수는 `preprocessing/stationarity_transform.py`에 정의되어 있으며, 
출력은 `transformed_df`, 변환 방식 사전은 `variable_transformations`로 전달됩니다.

---

## RVAR 계산 방식

`preprocessing/rvar_generator.py`를 통해 다음 작업을 수행합니다:

- KOSPI 종가를 기준으로 RVAR 계산
- `RVAR = Σ log(S_{t+i} / S_t)^2, i = {1,2,...22}` 로 22일 실현변동성 추출
- 이후 예측 대상 y로 사용 (`RVAR.shift(-21)`)

---

## 모델 상세 설명

### 1. 데이터 구조

모델은 **Multi-Input LSTM**과 **Standard LSTM** 두 가지 구조를 비교하며, 두 모델 모두 동일한 데이터셋을 기반으로 하지만 입력 형태가 다릅니다.

#### **예측 대상 (Target Variable)**

*   **`RVAR` (Realized Volatility, 실현 변동성)**: 모델이 최종적으로 예측하고자 하는 값입니다. 현재 `rvar_generator.py`에서는 코스피(KOSPI) 지수의 과거 22일간의 일별 로그 수익률 제곱합으로 계산되고 있습니다.

#### **입력 변수 (Input Features)**

입력 변수들은 총 4개의 그룹으로 나뉘어 **Multi-Input LSTM** 모델의 각기 다른 입력으로 사용됩니다. **Standard LSTM**은 이 4개 그룹을 하나로 합쳐서 사용합니다.

*   **데이터 전처리**: 모든 입력 변수들은 모델 학습 전에 `stationarity_transform.py`를 통해 **정상성(Stationarity)**을 갖도록 변환됩니다. 이는 시계열 데이터의 안정성을 확보하여 모델 성능을 높이기 위함이며, 주로 로그 변환, 차분(differencing) 등이 적용됩니다.

*   **Group 1: 거시 경제 및 시장 지표 (총 31개)**
    *   국내외 주가 지수 (`코스피`, `코스닥`, `Dow Jones` 등)
    *   채권 금리 (`국고채`, `회사채` 등)
    *   변동성 및 위험 지표 (`VIX`, `Dollar Index` 등)
    *   원자재 및 환율 (`WTI`, `USD-KRW Exchange Rate` 등)

*   **Group 2: 수급 주체 관련 지표 (총 6개)**
    *   투자 주체별 순매수 (`기관`, `개인`, `외국인`)
    *   시장 신용 관련 데이터 (`위탁매매 미수금` 등)

*   **Group 3: 유동성 관련 지표 (총 3개)**
    *   `투자자예탁금`, `RP 매도잔고` 등 시장의 전반적인 자금 흐름을 나타내는 지표

*   **Group 4: VKOSPI 기반 파생 변수 (총 3개)**
    *   **`vkospi_lag_1`**: VKOSPI 지수의 1일 전 값
    *   **`vkospi_MA5`**: VKOSPI 지수의 과거 5일 이동평균
    *   **`vkospi_MA22`**: VKOSPI 지수의 과거 22일 이동평균

### 2. 모델 구조 및 예측 방식

#### **시차 (Time Lag / Window)**

*   모델은 **과거 4일**의 데이터를 사용하여 미래를 예측합니다.
*   `make_lstm_dataset.py`의 `create_lagged_dataset` 함수에서 `window=4`로 설정되어, 모든 입력 변수 그룹을 4일치 묶음(sequence)으로 만듭니다.

#### **예측 시점 (Prediction Horizon)**

*   모델은 **1일 후**의 `RVAR` 값을 예측합니다.
*   즉, **과거 4일간의 데이터(t-3, t-2, t-1, t)를 사용하여 다음 날(t+1)의 RVAR 값을 예측**하는 구조입니다.

---

## 롤링 윈도우 교차 검증 (Rolling Window Cross-Validation)

`run_lstm.py`에서는 시계열 데이터에 적합한 롤링 윈도우 교차 검증 방식을 사용하여 모델의 성능을 평가합니다.

- **작동 방식**: 초기 학습 윈도우를 설정하고, 모델을 학습한 후 다음 `n` 스텝을 예측합니다. 이후 학습 윈도우와 테스트 윈도우를 `n` 스텝만큼 미래로 이동하며 이 과정을 반복합니다.
- **평가 지표**: 각 윈도우에서의 MSE(Mean Squared Error)와 MAE(Mean Absolute Error)를 기록하고, 모든 윈도우의 평균 성능을 최종 지표로 사용합니다.
- **적용 모델**: 현재 Multi-Input LSTM (MSE/MAE) 및 Standard LSTM (MSE/MAE) 총 4가지 모델에 대해 롤링 윈도우 교차 검증을 수행합니다.

---

## 타 모델 (HAR, XGBoost, RandomForest 등) 적용 시

- `transformed_df` 또는 `final_df`에서 동일하게 전처리된 설명변수를 사용할 수 있습니다.
- `make_har_features.py`, `make_rf_features.py` 등 별도 모듈을 만들고 해당 전처리 결과를 기반으로 피처를 구성하세요.
- 정상 시계열 로직(`stationarity_transform.py`)을 사용하세요.

- 해당 모듈에 필요한 정보는 conda_requirements.txt 를 이용하세요
```
conda create --name new_env --file conda_requirements.txt
```
---

## 🔗 향후 계획

- HAR / XGBoost / RandomForest 학습 추가
- 모델별 결과 비교 및 시각화 개선

---

**문의**: Yoonkeen / LSTM 담당