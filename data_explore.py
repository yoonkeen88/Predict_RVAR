import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

df = pd.read_csv('data/merged_data.csv')
print(df.isnull().sum())
print(df.describe())
print(df.head())
print(df.shape)