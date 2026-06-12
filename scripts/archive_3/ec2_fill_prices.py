from pathlib import Path
import re
import time
import pandas as pd
import requests


BASE_DIR = Path(
    r"C:\Users\FarhadBillimoria\OneDrive - Aurora Energy Research\Documents\Personal\Endog_demand_AI\pyai"
)

FILTERED_DIR = BASE_DIR / "data" / "processed" / "instances"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "instances"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_ec2_file():
    return pd.read_csv(FILTERED_DIR / "aws_gpu_filtered.csv")


def is_empty(val) -> bool:
    if pd.isna(val):
        return True
    return str(val).strip() == ""


def get_instance_name_column(df: pd.DataFrame) -> str:
    candidates = [
        "Instance type",
        "Instance Type",
        "API Name",
        "Name",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError("Could not find EC2 instance name column.")


def scrape_vantage_price(instance_type, session):
    url = f"https://instances.vantage.sh/aws/ec2/{instance_type}"

    response = session.get(url, timeout=20)
    if response.status_code != 200:
        return None

    text = response.text

    match = re.search(
        r"starting at \$([0-9]+(?:\.[0-9]+)?) per hour",
        text,
        flags=re.IGNORECASE
    )
    if match:
        return float(match.group(1))

    return None


def parse_price(val):
    """
    Convert mixed price formats to numeric float.

    Examples:
    - '12.345 hourly' -> 12.345
    - '$12.345 hourly' -> 12.345
    - '12.345' -> 12.345
    - 12.345 -> 12.345
    - NaN / '' -> pd.NA
    """
    if pd.isna(val):
        return pd.NA

    s = str(val).strip()
    if s == "":
        return pd.NA

    s = s.lower().replace("hourly", "").replace("$", "").strip()

    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", s)
    if match:
        return float(match.group(1))

    return pd.NA


def main():
    aws_df = load_ec2_file()

    instance_col = get_instance_name_column(aws_df)

    if "On Demand (Vantage)" not in aws_df.columns:
        aws_df["On Demand (Vantage)"] = pd.NA

    missing_mask = aws_df["On Demand"].apply(is_empty)

    print(f"Rows with missing On Demand price: {missing_mask.sum()}")

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36"
        }
    )

    for idx in aws_df[missing_mask].index:
        instance_type = str(aws_df.at[idx, instance_col]).strip()
        if not instance_type:
            continue

        price = scrape_vantage_price(instance_type, session)
        aws_df.at[idx, "On Demand (Vantage)"] = price

        print(f"{instance_type}: {price}")
        time.sleep(1.0)

    # Standardize original On Demand to numeric
    aws_df["On Demand Numeric"] = aws_df["On Demand"].apply(parse_price)

    # Ensure scraped values are numeric too
    aws_df["On Demand (Vantage)"] = pd.to_numeric(
        aws_df["On Demand (Vantage)"],
        errors="coerce"
    )

    # Fill missing numeric values using Vantage
    aws_df["On Demand Filled"] = aws_df["On Demand Numeric"].fillna(
        aws_df["On Demand (Vantage)"]
    )

    output_file = OUTPUT_DIR / "aws_gpu_filtered_with_vantage_prices.csv"
    aws_df.to_csv(output_file, index=False)

    print(f"\nSaved: {output_file}")

    print("\nSample of cleaned price columns:")
    cols = [c for c in ["On Demand", "On Demand Numeric", "On Demand (Vantage)", "On Demand Filled"] if c in aws_df.columns]
    print(aws_df[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()