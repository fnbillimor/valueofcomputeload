from pathlib import Path
import re
import time

import pandas as pd
import requests

from ec2_power_common import load_csv, output_path



def is_empty(val) -> bool:
    if pd.isna(val):
        return True
    return str(val).strip() == ""



def get_instance_name_column(df: pd.DataFrame) -> str:
    for col in ["Instance type", "Instance Type", "API Name", "Name"]:
        if col in df.columns:
            return col
    raise ValueError("Could not find EC2 instance name column.")



def parse_price(val):
    if pd.isna(val):
        return pd.NA
    s = str(val).strip().lower().replace("hourly", "").replace("$", "").strip()
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", s)
    return float(m.group(1)) if m else pd.NA



def scrape_vantage_price(instance_type, session):
    url = f"https://instances.vantage.sh/aws/ec2/{instance_type}"
    response = session.get(url, timeout=20)
    if response.status_code != 200:
        return None
    match = re.search(r"starting at \$([0-9]+(?:\.[0-9]+)?) per hour", response.text, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None



def main():
    aws_input = output_path("aws_gpu_filtered.csv")
    aws_df = load_csv(aws_input)
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

    aws_df["On Demand Numeric"] = aws_df["On Demand"].apply(parse_price)
    aws_df["On Demand (Vantage)"] = pd.to_numeric(aws_df["On Demand (Vantage)"], errors="coerce")
    aws_df["On Demand Filled"] = aws_df["On Demand Numeric"].fillna(aws_df["On Demand (Vantage)"])

    output_file = output_path("aws_gpu_filtered_with_vantage_prices.csv")
    aws_df.to_csv(output_file, index=False)

    print(f"Saved → {output_file}")


if __name__ == "__main__":
    main()
