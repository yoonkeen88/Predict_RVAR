import pandas as pd
import numpy as np

def generate_rvar_from_csv(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    for col in df.columns[1:]:
        df[col] = df[col].astype(str).str.replace(',', '').str.strip()
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['log_return'] = np.log(df['코스피'] / df['코스피'].shift(1))
    df['RVAR'] = df['log_return'].rolling(window=22).apply(lambda x: np.sum(x**2), raw=True).shift(-21)

    return df.dropna().reset_index(drop=True)