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

- KOSPI 종가를 기준으로 `log_return = log(S_t / S_{t-1})` 계산
- `RVAR = Σ (log_return_{t+1}² ~ log_return_{t+22}²)` 로 22일 실현변동성 추출
- 이후 예측 대상 y로 사용 (`RVAR.shift(-21)`)

---

## LSTM 입력 데이터 구조

`preprocessing/make_lstm_dataset.py`에서 다음과 같은 형태로 변환됩니다:

- **Multi-Input LSTM**: 의미 그룹별 입력 데이터 (4개 그룹 → 4개 Input)
  - 금융지표 / 수급 / 자금흐름 / RVAR 관련 입력 등
  - 각 그룹별 window: `t, t-1, t-2, t-3` (window=4)
  - 최종 형태:
    ```python
    x1: (n, 4, d1)
    x2: (n, 4, d2)
    x3: (n, 4, d3)
    x4: (n, 4, d4)
    y : (n, 1)
    ```
- **Standard LSTM**: 모든 입력 그룹을 하나의 시퀀스로 결합하여 단일 입력으로 사용합니다.

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