import numpy as np

def create_lagged_dataset(df, features, target, window=4):
    X = []
    y = []
    for i in range(window, len(df)):
        X.append(df[features].iloc[i-window:i].values)
        y.append(df[target].iloc[i])
    return np.array(X), np.array(y)