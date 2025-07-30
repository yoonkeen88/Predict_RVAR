import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss

def transform_to_stationary(df, column, signif=0.05):
    series = df[column].copy()
    if (series > 0).all():
        log_series = np.log(series)
        adf_p = adfuller(log_series.dropna(), autolag='AIC')[1]
        try:
            kpss_p = kpss(log_series.dropna(), regression='c', nlags='auto')[1]
        except:
            kpss_p = 0.01
        if adf_p < signif and kpss_p > signif:
            return log_series, 'log'
        else:
            log_diff_series = log_series.diff()
            adf_p2 = adfuller(log_diff_series.dropna(), autolag='AIC')[1]
            try:
                kpss_p2 = kpss(log_diff_series.dropna(), regression='c', nlags='auto')[1]
            except:
                kpss_p2 = 0.01
            if adf_p2 < signif and kpss_p2 > signif:
                return log_diff_series, 'log_diff'
    diff_series = series.diff()
    adf_p3 = adfuller(diff_series.dropna(), autolag='AIC')[1]
    try:
        kpss_p3 = kpss(diff_series.dropna(), regression='c', nlags='auto')[1]
    except:
        kpss_p3 = 0.01
    if adf_p3 < signif and kpss_p3 > signif:
        return diff_series, 'diff'
    return series, 'none'

def apply_stationarity_transformation(df, exclude_cols=['Date', 'log_return', 'RVAR']):
    transformed_df = pd.DataFrame()
    transformations = {}
    for col in df.columns:
        if col in exclude_cols:
            continue
        try:
            ts, method = transform_to_stationary(df, col)
            transformed_df[col] = ts
            transformations[col] = method
        except:
            transformed_df[col] = df[col]
            transformations[col] = 'error'
    return transformed_df, transformations