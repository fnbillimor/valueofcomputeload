from pathlib import Path

import pandas as pd

from ec2_power_common import ORIGINAL_INSTANCES_DIR, output_path


AWS_FILE = ORIGINAL_INSTANCES_DIR / "Amazon EC2 Instance Comparison.csv"
AZURE_FILE = ORIGINAL_INSTANCES_DIR / "Azure VM Comparison.csv"
GCP_FILE = ORIGINAL_INSTANCES_DIR / "GCP Compute Engine Comparison.csv"



def keep_nonzero_gpu(val) -> bool:
    try:
        return float(str(val).split()[0].replace('X', '').replace('x', '')) != 0
    except Exception:
        s = str(val)
        import re
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)", s)
        return float(m.group(1)) != 0 if m else False



def main():
    aws_df = pd.read_csv(AWS_FILE)
    azure_df = pd.read_csv(AZURE_FILE)
    gcp_df = pd.read_csv(GCP_FILE)

    aws_filtered = aws_df[aws_df["GPUs"].apply(keep_nonzero_gpu)].copy()
    azure_filtered = azure_df[azure_df["GPUs"].apply(keep_nonzero_gpu)].copy()
    gcp_filtered = gcp_df[gcp_df["GPUs"].apply(keep_nonzero_gpu)].copy()

    aws_file = output_path("aws_gpu_filtered.csv")
    azure_file = output_path("azure_gpu_filtered.csv")
    gcp_file = output_path("gcp_gpu_filtered.csv")

    aws_filtered.to_csv(aws_file, index=False)
    azure_filtered.to_csv(azure_file, index=False)
    gcp_filtered.to_csv(gcp_file, index=False)

    print(f"Saved → {aws_file}")
    print(f"Saved → {azure_file}")
    print(f"Saved → {gcp_file}")
    print(f"AWS rows: {len(aws_filtered)}")
    print(f"Azure rows: {len(azure_filtered)}")
    print(f"GCP rows: {len(gcp_filtered)}")


if __name__ == "__main__":
    main()
