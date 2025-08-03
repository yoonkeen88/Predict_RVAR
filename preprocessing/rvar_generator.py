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

    rvar_components = []
    for i in range(1, 23):
        log_return_i = np.log(df['코스피'] / df['코스피'].shift(i))
        squared_log_return_i = log_return_i**2
        rvar_components.append(squared_log_return_i)
    
    rvar_df = pd.concat(rvar_components, axis=1)
    df['RVAR'] = rvar_df.sum(axis=1)

    return df.dropna().reset_index(drop=True)

print("RVAR generation function is ready to use.")
ddf = generate_rvar_from_csv('data/VKOSPI_pred_data.csv')
print(ddf[['RVAR']].head(30))