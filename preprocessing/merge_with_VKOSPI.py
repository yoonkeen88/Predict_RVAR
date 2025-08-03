import pandas as pd

# Define file paths
kospi_volatility_path = '/Users/angwang-yun/Desktop/Project/RVAR_pred/data/KOSPI Volatility 과거 데이터.csv'
vkospi_pred_path = '/Users/angwang-yun/Desktop/Project/RVAR_pred/data/VKOSPI_pred_data.csv'
output_path = '/Users/angwang-yun/Desktop/Project/RVAR_pred/data/merged_data.csv'

# Load the datasets
kospi_volatility = pd.read_csv(kospi_volatility_path)
vkospi_pred = pd.read_csv(vkospi_pred_path)

# Rename columns for consistency
kospi_volatility.rename(columns={'날짜': 'Date', '종가': 'VKOSPI'}, inplace=True)

# Convert 'Date' columns to datetime objects
kospi_volatility['Date'] = pd.to_datetime(kospi_volatility['Date'])
vkospi_pred['Date'] = pd.to_datetime(vkospi_pred['Date'])

# Merge the two dataframes on the 'Date' column
merged_df = pd.merge(vkospi_pred, kospi_volatility[['Date', 'VKOSPI']], on='Date', how='left')

# Save the merged dataframe to a new CSV file
merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"Merged data saved to {output_path}")
