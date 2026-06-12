import sys
from pathlib import Path
import pandas as pd


BASE_DIR = Path(
    r"C:\Users\FarhadBillimoria\OneDrive - Aurora Energy Research\Documents\Personal\Endog_demand_AI\pyai"
)

INSTANCES_DIR = BASE_DIR / "data" / "original" / "instances"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "instances"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Load files
aws_df = pd.read_csv(INSTANCES_DIR / "Amazon EC2 Instance Comparison.csv")
azure_df = pd.read_csv(INSTANCES_DIR / "Azure VM Comparison.csv")
gcp_df = pd.read_csv(INSTANCES_DIR / "GCP Compute Engine Comparison.csv")


def keep_nonzero_gpu(val):
    try:
        return float(val) != 0
    except:
        return True


def not_empty(series):
    return series.notna() & (series.astype(str).str.strip() != "")


# --- Apply filters ---
aws_filtered = aws_df[aws_df["GPUs"].apply(keep_nonzero_gpu)]
azure_filtered = azure_df[azure_df["GPUs"].apply(keep_nonzero_gpu)]
gcp_filtered = gcp_df[gcp_df["GPUs"].apply(keep_nonzero_gpu)]

#aws_filtered = aws_filtered[not_empty(aws_filtered["On Demand"])]
#azure_filtered = azure_filtered[not_empty(azure_filtered["Linux On Demand cost"])]
#gcp_filtered = gcp_filtered[not_empty(gcp_filtered["Linux On Demand cost"])]


# --- Save intermediate outputs ---
aws_filtered.to_csv(OUTPUT_DIR / "aws_gpu_filtered.csv", index=False)
azure_filtered.to_csv(OUTPUT_DIR / "azure_gpu_filtered.csv", index=False)
gcp_filtered.to_csv(OUTPUT_DIR / "gcp_gpu_filtered.csv", index=False)


print("Saved intermediate files to:", OUTPUT_DIR)
print("AWS:", aws_filtered.shape)
print("Azure:", azure_filtered.shape)
print("GCP:", gcp_filtered.shape)