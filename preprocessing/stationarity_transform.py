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

def apply_stationarity_transformation(df, exclude_cols=['Date', 'log_return', 'RVAR'], output_csv_path=None):
    """
    Applies stationarity transformations to the columns of a DataFrame and optionally saves the details to a CSV file.

    Args:
        df (pd.DataFrame): The input DataFrame.
        exclude_cols (list, optional): A list of columns to exclude from transformation. Defaults to ['Date', 'log_return', 'RVAR'].
        output_csv_path (str, optional): If provided, the path to save the CSV file with transformation details.

    Returns:
        tuple: A tuple containing:
            - transformed_df (pd.DataFrame): The DataFrame with stationary columns.
            - transformations (dict): A dictionary mapping column names to the applied transformation method.
    """
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

    if output_csv_path:
        transform_details = []
        for column, method in transformations.items():
            diff_count = 1 if 'diff' in method else 0
            transform_details.append({
                'column': column,
                'transformation': method,
                'diff_count': diff_count
            })
        
        transform_df = pd.DataFrame(transform_details)
        transform_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

    return transformed_df, transformations