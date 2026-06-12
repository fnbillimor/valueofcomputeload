import pandas as pd
from ec2_power_common import output_path, find_column

# ============================================================
# File names
# ============================================================
POWER_FILE = output_path("aws_gpu_cpu_power_and_value.csv")
SOURCE_FILE = output_path("aws_gpu_filtered_with_vantage_prices.csv")
OUTPUT_FILE = output_path("aws_gpu_cpu_power_and_value_wide.csv")


def main():
    power_df = pd.read_csv(POWER_FILE)
    source_df = pd.read_csv(SOURCE_FILE)

    join_col_power = find_column(
        power_df,
        ["API Name", "Instance type", "Instance Type", "instance_type", "Name"],
        required=True,
    )

    join_col_source = find_column(
        source_df,
        ["API Name", "Instance type", "Instance Type", "instance_type", "Name"],
        required=True,
    )

    power_df = power_df.rename(columns={join_col_power: "instance_type"})
    source_df = source_df.rename(columns={join_col_source: "instance_type"})

    existing_cols = set(power_df.columns)
    additional_cols = [
        col for col in source_df.columns
        if col not in existing_cols
    ]

    cols_to_merge = ["instance_type"] + additional_cols
    source_subset = source_df[cols_to_merge]

    merged = power_df.merge(
        source_subset,
        on="instance_type",
        how="left",
        validate="many_to_one",
    )

    merged.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved → {OUTPUT_FILE}")
    print(f"Rows (power file): {len(power_df)}")
    print(f"Rows (merged): {len(merged)}")
    print(f"Columns added: {len(additional_cols)}")

    print("\nPreview:")
    print(merged.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
