from pathlib import Path
import re
import pandas as pd


# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(
    r"C:\Users\FarhadBillimoria\OneDrive - Aurora Energy Research\Documents\Personal\Endog_demand_AI\pyai"
)

INPUT_DIR = BASE_DIR / "data" / "processed" / "instances"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "instances"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = INPUT_DIR / "aws_gpu_filtered_with_vantage_prices.csv"
OUTPUT_FILE = OUTPUT_DIR / "aws_gpu_power_and_value.csv"
GPU_REVIEW_FILE = OUTPUT_DIR / "ec2_gpu_tdp_review.csv"


# ============================================================
# Assumptions
# ============================================================
GPU_UTILIZATION = 0.90
CPU_UTILIZATION = 0.50

# Placeholder until CPU model/TDP mapping is added
CPU_TDP_W_PER_VCPU = 0.0


# ============================================================
# GPU normalization
# ============================================================
GPU_NORMALIZATION_MAP = {
    "NVIDIA A100": "A100",
    "NVIDIA A10G": "A10G",
    "NVIDIA B200": "B200",
    "NVIDIA B300": "B300",
    "NVIDIA H100": "H100",
    "NVIDIA H200": "H200",
    "NVIDIA L4": "L4",
    "NVIDIA L40S": "L40S",
    "NVIDIA T4 Tensor Core": "T4",
    "NVIDIA T4G Tensor Core": "T4G",
    "NVIDIA Tesla K80": "K80",
    "NVIDIA Tesla M60": "M60",
    "NVIDIA Tesla V100": "V100",
}


# ============================================================
# GPU TDP map (W)
# ============================================================
GPU_TDP_W = {
    "T4": 70,
    "T4G": 70,
    "L4": 72,
    "L40S": 350,
    "A10": 150,
    "A10G": 300,
    "K80": 300,
    "M60": 300,
    "V100": 300,   # SXM2
    "A100": 400,   # SXM
    "H100": 700,   # SXM
    "H200": 700,   # SXM
    "B200": 1000,
    "B300": 1100,
    "GB200": 1200,
}


# ============================================================
# Helpers
# ============================================================
def load_ec2_file():
    if not INPUT_FILE.exists():
        raise FileNotFoundError("Input file not found: {}".format(INPUT_FILE))
    return pd.read_csv(INPUT_FILE)


def find_column(df, candidates, required=True):
    for col in candidates:
        if col in df.columns:
            return col

    if required:
        raise ValueError("Could not find any of these columns: {}".format(candidates))

    return None


def parse_price(val):
    """
    Convert mixed hourly price strings to float.

    Examples:
    - '$32.77 hourly' -> 32.77
    - '32.77 hourly' -> 32.77
    - 32.77 -> 32.77
    - '' / NaN -> pd.NA
    """
    if pd.isna(val):
        return pd.NA

    s = str(val).strip()
    if s == "":
        return pd.NA

    s = s.replace(",", "")
    s = s.replace("$", "")
    s = s.lower().replace("hourly", "").strip()

    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", s)
    if match:
        return float(match.group(1))

    return pd.NA


def parse_numeric(val):
    """
    Extract first numeric value from a field.
    """
    if pd.isna(val):
        return pd.NA

    s = str(val).strip()
    if s == "":
        return pd.NA

    s = s.replace(",", "")
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", s)
    if match:
        return float(match.group(1))

    return pd.NA


def normalize_gpu_model(val):
    if pd.isna(val):
        return pd.NA

    s = str(val).strip()
    if s == "":
        return pd.NA

    if s in GPU_NORMALIZATION_MAP:
        return GPU_NORMALIZATION_MAP[s]

    s = s.replace("NVIDIA", "").replace("Tesla", "").strip()
    s = s.replace("Tensor Core", "").strip()

    return s if s != "" else pd.NA


def get_unique_gpu_models(df):
    gpu_col = find_column(df, ["GPU model", "GPU Model", "Gpu model"])
    models = (
        df[gpu_col]
        .apply(normalize_gpu_model)
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )
    return models


# ============================================================
# Standardization steps
# ============================================================
def standardize_price(df):
    df = df.copy()

    price_col = find_column(
        df,
        ["On Demand Filled", "price_usd_per_hour", "On Demand Numeric", "On Demand"],
        required=False
    )

    if price_col is None:
        df["price_usd_per_hour"] = pd.NA
    else:
        df["price_usd_per_hour"] = df[price_col].apply(parse_price)

    return df


def standardize_gpu_count(df):
    df = df.copy()

    gpu_col = find_column(df, ["GPUs", "GPU", "Gpu"], required=False)

    if gpu_col is None:
        df["gpu_count"] = pd.NA
    else:
        df["gpu_count"] = df[gpu_col].apply(parse_numeric)

    return df


def standardize_vcpu_count(df):
    df = df.copy()

    vcpu_col = find_column(df, ["vCPUs", "vCPU", "VCPUs", "VCpu"], required=False)

    if vcpu_col is None:
        df["vcpu_count"] = pd.NA
    else:
        df["vcpu_count"] = df[vcpu_col].apply(parse_numeric)

    return df


def standardize_gpu_model(df):
    df = df.copy()

    gpu_model_col = find_column(df, ["GPU model", "GPU Model", "Gpu model"])
    df["gpu_model_clean"] = df[gpu_model_col].apply(normalize_gpu_model)

    return df


def map_gpu_tdp(df):
    df = df.copy()
    df["gpu_tdp_w"] = df["gpu_model_clean"].map(GPU_TDP_W)
    return df


# ============================================================
# Compute step
# ============================================================
def compute_power_and_value(df):
    df = df.copy()

    df["gpu_power_w"] = df["gpu_count"] * df["gpu_tdp_w"] * GPU_UTILIZATION

    df["cpu_tdp_w_per_vcpu"] = CPU_TDP_W_PER_VCPU
    df["cpu_power_w"] = df["vcpu_count"] * df["cpu_tdp_w_per_vcpu"] * CPU_UTILIZATION

    df["total_power_w"] = df["gpu_power_w"].fillna(0) + df["cpu_power_w"].fillna(0)
    df["total_power_mw"] = df["total_power_w"] / 1_000_000.0

    df["value_usd_per_mwh"] = pd.NA

    valid_mask = (
        df["price_usd_per_hour"].notna() &
        df["total_power_mw"].notna() &
        (df["total_power_mw"] > 0)
    )

    df.loc[valid_mask, "value_usd_per_mwh"] = (
        df.loc[valid_mask, "price_usd_per_hour"] /
        df.loc[valid_mask, "total_power_mw"]
    )

    return df


# ============================================================
# Review table
# ============================================================
def build_gpu_review(df):
    models = get_unique_gpu_models(df)

    rows = []
    for model in models:
        rows.append({
            "GPU model clean": model,
            "gpu_tdp_w": GPU_TDP_W.get(model, pd.NA),
            "status": "resolved" if model in GPU_TDP_W else "unmapped"
        })

    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================
def main():
    ec2_df = load_ec2_file()

    ec2_df = standardize_price(ec2_df)
    ec2_df = standardize_gpu_count(ec2_df)
    ec2_df = standardize_vcpu_count(ec2_df)
    ec2_df = standardize_gpu_model(ec2_df)
    ec2_df = map_gpu_tdp(ec2_df)
    ec2_df = compute_power_and_value(ec2_df)

    gpu_review = build_gpu_review(ec2_df)

    ec2_df.to_csv(OUTPUT_FILE, index=False)
    gpu_review.to_csv(GPU_REVIEW_FILE, index=False)

    print("Saved:")
    print(" - {}".format(OUTPUT_FILE))
    print(" - {}".format(GPU_REVIEW_FILE))

    print("\nUnique GPU models:")
    for model in get_unique_gpu_models(ec2_df):
        print(" - {}".format(model))

    unresolved = ec2_df[ec2_df["gpu_tdp_w"].isna()]
    print("\nRows with unresolved GPU TDP: {}".format(len(unresolved)))

    missing_price = ec2_df[ec2_df["price_usd_per_hour"].isna()]
    print("Rows with missing price: {}".format(len(missing_price)))

    preview_cols = [
        c for c in [
            "Instance type",
            "GPU model",
            "gpu_model_clean",
            "gpu_count",
            "vcpu_count",
            "price_usd_per_hour",
            "gpu_tdp_w",
            "gpu_power_w",
            "cpu_power_w",
            "total_power_mw",
            "value_usd_per_mwh",
        ] if c in ec2_df.columns
    ]

    print("\nPreview:")
    print(ec2_df[preview_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()