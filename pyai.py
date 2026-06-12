import os
import pandas as pd

# Define base directories
data_dir = r"C:\Users\FarhadBillimoria\OneDrive - Aurora Energy Research\Documents\Personal\Endog_demand_AI\pyai\data\sample"
script_dir = r"C:\Users\FarhadBillimoria\OneDrive - Aurora Energy Research\Documents\Personal\Endog_demand_AI\pyai\scripts"

# File names (update if needed)
file1 = "P4d24xlarge.csv"
file2 = "inf2xlarge.csv"

# Full paths
file1_path = os.path.join(data_dir, file1)
file2_path = os.path.join(data_dir, file2)

# Load data
df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)

# Basic checks
print("File 1 shape:", df1.shape)
print("File 2 shape:", df2.shape)

print("\nFile 1 preview:")
print(df1.head())

print("\nFile 2 preview:")
print(df2.head())

