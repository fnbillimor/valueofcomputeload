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


def prompt_manual_price(instance_type: str, price_label: str):
    while True:
        try:
            raw_value = input(
                f"{instance_type}: could not pull {price_label}. "
                "Enter $/hour manually, or press Enter to leave blank: "
            )
        except EOFError:
            print(f"{instance_type}: no interactive input available; leaving {price_label} blank")
            return pd.NA

        raw_value = raw_value.strip()
        if raw_value == "":
            return pd.NA

        parsed = parse_price(raw_value)
        if not pd.isna(parsed):
            return parsed

        print("Could not parse that value. Example accepted inputs: 70.35, $70.35, 70.35 hourly")


def first_float_match(text: str, patterns):
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    return None


def scrape_vantage_prices(instance_type, session):
    """
    Scrape Vantage page for:
      - On Demand
      - Spot  -> mapped to Linux Spot Average cost
      - 1-Year Reserved -> mapped to Linux Reserved cost

    Uses the original 'starting at $... per hour' logic first for On Demand,
    because that seems to work better.
    """
    url = f"https://instances.vantage.sh/aws/ec2/{instance_type}"
    response = session.get(url, timeout=20)

    if response.status_code != 200:
        return None

    text = response.text

    # Best-performing original method for On Demand
    on_demand = first_float_match(
        text,
        [
            r"starting at \$([0-9]+(?:\.[0-9]+)?) per hour",
            r"\$([0-9]+(?:\.[0-9]+)?)\s*per hour",
            r"\$([0-9]+(?:\.[0-9]+)?)\s*On Demand",
            r"On Demand\s*\$([0-9]+(?:\.[0-9]+)?)",
        ],
    )

    # Spot on Vantage page maps to Linux Spot Average cost in your output
    spot = first_float_match(
        text,
        [
            r"\$([0-9]+(?:\.[0-9]+)?)\s*Spot",
            r"Spot\s*\$([0-9]+(?:\.[0-9]+)?)",
            r'"Spot"[^$]{0,200}\$([0-9]+(?:\.[0-9]+)?)',
            r"\$([0-9]+(?:\.[0-9]+)?)[^A-Za-z0-9]{0,50}Spot",
        ],
    )

    reserved_1yr = first_float_match(
        text,
        [
            r"\$([0-9]+(?:\.[0-9]+)?)\s*1-Year Reserved",
            r"1-Year Reserved\s*\$([0-9]+(?:\.[0-9]+)?)",
            r'"1-Year Reserved"[^$]{0,200}\$([0-9]+(?:\.[0-9]+)?)',
            r"\$([0-9]+(?:\.[0-9]+)?)[^A-Za-z0-9]{0,50}1-Year Reserved",
        ],
    )

    return {
        "on_demand": on_demand,
        "spot": spot,
        "reserved_1yr": reserved_1yr,
    }


def ensure_column(df: pd.DataFrame, col: str):
    if col not in df.columns:
        df[col] = pd.NA


def main():
    aws_input = output_path("aws_gpu_filtered.csv")
    aws_df = load_csv(aws_input)
    instance_col = get_instance_name_column(aws_df)

    ensure_column(aws_df, "On Demand (Vantage)")
    ensure_column(aws_df, "Linux Spot Average cost (Vantage)")
    ensure_column(aws_df, "Linux Reserved cost (Vantage)")

    on_demand_missing = (
        aws_df["On Demand"].apply(is_empty)
        if "On Demand" in aws_df.columns
        else pd.Series(True, index=aws_df.index)
    )

    spot_missing = (
        aws_df["Linux Spot Average cost"].apply(is_empty)
        if "Linux Spot Average cost" in aws_df.columns
        else pd.Series(True, index=aws_df.index)
    )

    reserved_missing = (
        aws_df["Linux Reserved cost"].apply(is_empty)
        if "Linux Reserved cost" in aws_df.columns
        else pd.Series(True, index=aws_df.index)
    )

    scrape_mask = on_demand_missing | spot_missing | reserved_missing
    print(f"Rows needing Vantage scrape: {int(scrape_mask.sum())}")

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/146.0.0.0 Safari/537.36"
            )
        }
    )

    for idx in aws_df[scrape_mask].index:
        instance_type = str(aws_df.at[idx, instance_col]).strip()
        if not instance_type:
            continue

        prices = scrape_vantage_prices(instance_type, session)
        if prices is None:
            print(f"{instance_type}: page not found / request failed")
            if bool(spot_missing.loc[idx]):
                aws_df.at[idx, "Linux Spot Average cost (Vantage)"] = prompt_manual_price(
                    instance_type,
                    "Linux Spot Average cost",
                )
            time.sleep(1.0)
            continue

        if bool(spot_missing.loc[idx]) and prices["spot"] is None:
            prices["spot"] = prompt_manual_price(
                instance_type,
                "Linux Spot Average cost",
            )

        # Only overwrite Vantage columns with scraped values
        aws_df.at[idx, "On Demand (Vantage)"] = prices["on_demand"]
        aws_df.at[idx, "Linux Spot Average cost (Vantage)"] = prices["spot"]
        aws_df.at[idx, "Linux Reserved cost (Vantage)"] = prices["reserved_1yr"]

        print(
            f"{instance_type}: "
            f"on_demand={prices['on_demand']}, "
            f"spot={prices['spot']}, "
            f"reserved_1yr={prices['reserved_1yr']}"
        )

        time.sleep(1.0)

    # Existing source columns -> numeric
    if "On Demand" in aws_df.columns:
        aws_df["On Demand Numeric"] = aws_df["On Demand"].apply(parse_price)
    else:
        aws_df["On Demand Numeric"] = pd.NA

    if "Linux Spot Average cost" in aws_df.columns:
        aws_df["Linux Spot Average Numeric"] = aws_df["Linux Spot Average cost"].apply(parse_price)
    else:
        aws_df["Linux Spot Average Numeric"] = pd.NA

    if "Linux Reserved cost" in aws_df.columns:
        aws_df["Linux Reserved Numeric"] = aws_df["Linux Reserved cost"].apply(parse_price)
    else:
        aws_df["Linux Reserved Numeric"] = pd.NA

    # Scraped Vantage columns -> numeric
    aws_df["On Demand (Vantage)"] = pd.to_numeric(aws_df["On Demand (Vantage)"], errors="coerce")
    aws_df["Linux Spot Average cost (Vantage)"] = pd.to_numeric(
        aws_df["Linux Spot Average cost (Vantage)"], errors="coerce"
    )
    aws_df["Linux Reserved cost (Vantage)"] = pd.to_numeric(
        aws_df["Linux Reserved cost (Vantage)"], errors="coerce"
    )

    # Filled columns
    aws_df["On Demand Filled"] = aws_df["On Demand Numeric"].fillna(aws_df["On Demand (Vantage)"])
    aws_df["Linux Spot Average cost Filled"] = aws_df["Linux Spot Average Numeric"].fillna(
        aws_df["Linux Spot Average cost (Vantage)"]
    )
    aws_df["Linux Reserved cost Filled"] = aws_df["Linux Reserved Numeric"].fillna(
        aws_df["Linux Reserved cost (Vantage)"]
    )

    output_file = output_path("aws_gpu_filtered_with_vantage_prices.csv")
    aws_df.to_csv(output_file, index=False)

    print(f"Saved -> {output_file}")


if __name__ == "__main__":
    main()
