import pandas as pd
import numpy as np
from stationarity_transform import apply_stationarity_transformation

# Create a sample DataFrame for testing

df = pd.read_csv('data/VKOSPI_pred_data.csv')

# Apply the stationarity transformation
transformed_df, transformations = apply_stationarity_transformation(
    df,
    exclude_cols=['Date', 'log_return', 'RVAR'],
    output_csv_path='transformation_details.csv'
)

# Print the results
print("--- Original DataFrame ---")
print(df.head())
print("\n--- Transformed DataFrame ---")
print(transformed_df.head())
print("\n--- Transformations Applied ---")
print(transformations)
print("\nTransformation details saved to 'transformation_details.csv'")
