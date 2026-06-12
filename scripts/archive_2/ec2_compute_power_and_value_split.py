import pandas as pd
from ec2_power_common import (
    load_csv,
    output_path,
    find_column,
    parse_numeric,
    parse_price,
    normalize_gpu_model,
    normalize_cpu_model,
    GPU_TDP_W,
    cpu_lookup_record,
)



def main():
    df = load_csv()

    gpu_model_col = find_column(
        df,
        ["GPU model", "GPU Model", "Gpu model"],
        required=False,
    )
    cpu_model_col = find_column(
        df,
        ["Physical Processor", "CPU model", "CPU Model", "Processor", "CPU"],
        required=False,
    )
    gpu_count_col = find_column(
        df,
        ["GPUs", "GPU", "Gpu"],
        required=False,
    )
    vcpu_col = find_column(
        df,
        ["vCPUs", "vCPU", "VCPUs", "VCpu"],
        required=False,
    )
    price_col = find_column(
        df,
        ["On Demand Filled", "price_usd_per_hour", "On Demand Numeric", "On Demand"],
        required=False,
    )
    instance_type_col = find_column(
        df,
        ["API Name", "Instance type", "Instance Type", "instance_type", "Name"],
        required=False,
    )

    df["gpu_model_clean"] = (
        df[gpu_model_col].apply(normalize_gpu_model) if gpu_model_col is not None else pd.NA
    )
    df["cpu_model_clean"] = (
        df[cpu_model_col].apply(normalize_cpu_model) if cpu_model_col is not None else pd.NA
    )

    df["gpu_count"] = (
        df[gpu_count_col].apply(parse_numeric) if gpu_count_col is not None else pd.NA
    )
    df["vcpu_count"] = (
        df[vcpu_col].apply(parse_numeric) if vcpu_col is not None else pd.NA
    )
    df["price_usd_per_hour"] = (
        df[price_col].apply(parse_price) if price_col is not None else pd.NA
    )

    df["gpu_count"] = pd.to_numeric(df["gpu_count"], errors="coerce")
    df["vcpu_count"] = pd.to_numeric(df["vcpu_count"], errors="coerce")
    df["price_usd_per_hour"] = pd.to_numeric(df["price_usd_per_hour"], errors="coerce")

    df["gpu_tdp_w_per_gpu"] = df["gpu_model_clean"].map(GPU_TDP_W)
    #df["gpu_power_w_per_gpu_at_util"] = df["gpu_tdp_w_per_gpu"] * GPU_UTILIZATION
    #df["total_gpu_power_w"] = df["gpu_count"] * df["gpu_power_w_per_gpu_at_util"]
    df["gpu_power_w_per_instance"] = df["gpu_count"] * df["gpu_tdp_w_per_gpu"]
    df["cpu_power_w_per_instance"] = df["vcpu_count"] * df["cpu_tdp_w_per_vcpu"]

    cpu_lookup_df = df["cpu_model_clean"].apply(cpu_lookup_record).apply(pd.Series)
    df = pd.concat([df, cpu_lookup_df], axis=1)

    df["cpu_tdp_w_package"] = pd.to_numeric(df["cpu_tdp_w_package"], errors="coerce")
    df["threads_per_package"] = pd.to_numeric(df["threads_per_package"], errors="coerce")
    df["cpu_tdp_w_per_vcpu"] = pd.to_numeric(df["cpu_tdp_w_per_vcpu"], errors="coerce")
    df["gpu_tdp_w_per_gpu"] = pd.to_numeric(df["gpu_tdp_w_per_gpu"], errors="coerce")
    df["gpu_power_w_per_instance"] = pd.to_numeric(df["gpu_power_w_per_instance"], errors="coerce")
    df["cpu_power_w_per_instance"] = pd.to_numeric(df["cpu_power_w_per_instance"], errors="coerce")
    

    GPU_UTILIZATION_IDLE = 0.90
    CPU_UTILIZATION_IDLE = 0.50

    GPU_UTILIZATION_LO = 0.90
    CPU_UTILIZATION_LO = 0.50

    GPU_UTILIZATION_MID = 0.90
    CPU_UTILIZATION_MID = 0.50

    GPU_UTILIZATION_HI = 0.90
    CPU_UTILIZATION_HI = 0.50

    GPU_UTILIZATION_CUSTOM = 0.90
    CPU_UTILIZATION_CUSTOM = 0.50

    #df["total_gpu_power_w"] = pd.to_numeric(df["total_gpu_power_w"], errors="coerce")
    #df["cpu_power_w_per_vcpu_at_util"] = df["cpu_tdp_w_per_vcpu"] * CPU_UTILIZATION
    #df["total_cpu_power_w"] = df["vcpu_count"] * df["cpu_power_w_per_vcpu_at_util"]


    #df["total_cpu_power_w"] = pd.to_numeric(df["total_cpu_power_w"], errors="coerce")


    df["total_power_w"] = df["total_gpu_power_w"].fillna(0) + df["total_cpu_power_w"].fillna(0)
    df["total_power_mw"] = df["total_power_w"] / 1_000_000.0

    df["voll_usd_per_mwh"] = pd.NA
    valid_mask = (
        df["price_usd_per_hour"].notna()
        & df["total_power_mw"].notna()
        & (df["total_power_mw"] > 0)
    )
    df.loc[valid_mask, "voll_usd_per_mwh"] = (
        df.loc[valid_mask, "price_usd_per_hour"] / df.loc[valid_mask, "total_power_mw"]
    )

    df["gpu_mapping_status"] = df["gpu_tdp_w_per_gpu"].apply(
        lambda x: "resolved" if pd.notna(x) else ("missing" if pd.isna(x) else "review_required")
    )

    output_cols_preferred = [
        instance_type_col,
        gpu_model_col,
        cpu_model_col,
        gpu_count_col,
        vcpu_col,
        price_col,
        "gpu_model_clean",
        "cpu_model_clean",
        "gpu_count",
        "vcpu_count",
        "price_usd_per_hour",
        "gpu_tdp_w_per_gpu",
        "gpu_power_w_per_gpu_at_util",
        "total_gpu_power_w",
        "cpu_tdp_w_package",
        "threads_per_package",
        "cpu_tdp_w_per_vcpu",
        "cpu_power_w_per_vcpu_at_util",
        "total_cpu_power_w",
        "total_power_w",
        "total_power_mw",
        "voll_usd_per_mwh",
        "gpu_mapping_status",
        "cpu_mapping_status",
        "cpu_source_type",
        "cpu_source_name",
        "cpu_source_note",
    ]
    output_cols = [c for c in output_cols_preferred if c is not None and c in df.columns]

    output_file = output_path("aws_gpu_cpu_power_and_value.csv")
    df[output_cols].to_csv(output_file, index=False)

    unresolved_mask = (
        (df["gpu_model_clean"].notna() & df["gpu_tdp_w_per_gpu"].isna())
        | (df["cpu_model_clean"].notna() & (df["cpu_mapping_status"] != "resolved"))
    )
    review_cols_preferred = [
        instance_type_col,
        gpu_model_col,
        cpu_model_col,
        gpu_count_col,
        vcpu_col,
        price_col,
        "gpu_model_clean",
        "cpu_model_clean",
        "gpu_tdp_w_per_gpu",
        "cpu_tdp_w_package",
        "threads_per_package",
        "cpu_tdp_w_per_vcpu",
        "gpu_mapping_status",
        "cpu_mapping_status",
        "cpu_source_type",
        "cpu_source_name",
        "cpu_source_note",
    ]
    review_cols = [c for c in review_cols_preferred if c is not None and c in df.columns]

    review_file = output_path("ec2_unresolved_power_mappings_review.csv")
    df.loc[unresolved_mask, review_cols].to_csv(review_file, index=False)

    print(f"Saved → {output_file}")
    print(f"Saved → {review_file}")

    print("\nSummary:")
    print(f"Rows: {len(df)}")
    print(f"Rows with VOLL computed: {df['voll_usd_per_mwh'].notna().sum()}")
    print(f"Rows with unresolved GPU mapping: {(df['gpu_model_clean'].notna() & df['gpu_tdp_w_per_gpu'].isna()).sum()}")
    print(f"Rows with unresolved CPU mapping: {(df['cpu_model_clean'].notna() & (df['cpu_mapping_status'] != 'resolved')).sum()}")

    preview_cols = [
        c for c in [
            instance_type_col,
            "gpu_model_clean",
            "cpu_model_clean",
            "gpu_count",
            "vcpu_count",
            "price_usd_per_hour",
            "total_gpu_power_w",
            "total_cpu_power_w",
            "total_power_mw",
            "voll_usd_per_mwh",
            "cpu_source_type",
        ] if c is not None and c in df.columns
    ]
    print("\nPreview:")
    print(df[preview_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
