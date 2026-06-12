import pandas as pd


def _first_present_numeric(df: pd.DataFrame, candidates, default: float = 0.0) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype="float64")


from ec2_power_common import (
    OTHER_COMPONENTS_FACTOR,
    SCENARIO_CONFIG,
    GPU_TDP_W,
    RAM_ACTIVE_W_PER_GB,
    RAM_IDLE_W_PER_GB,
    cpu_lookup_record,
    find_column,
    load_csv,
    normalize_cpu_model,
    normalize_gpu_model,
    output_path,
    parse_instance_storage,
    parse_memory_gib,
    parse_numeric,
    parse_price,
    scenario_ram_w,
    scenario_ssd_w,
    select_ssd_proxy,
    PUE_FACTOR,
)


def main():
    df = load_csv()

    gpu_model_col = find_column(df, ["GPU model", "GPU Model", "Gpu model"], required=False)
    cpu_model_col = find_column(df, ["Physical Processor", "CPU model", "CPU Model", "Processor", "CPU"], required=False)
    gpu_count_col = find_column(df, ["GPUs", "GPU", "Gpu"], required=False)
    vcpu_col = find_column(df, ["vCPUs", "vCPU", "VCPUs", "VCpu"], required=False)
    memory_col = find_column(df, ["Instance Memory", "Memory", "RAM", "Instance memory"], required=False)
    storage_col = find_column(df, ["Instance Storage", "Storage", "Local SSD"], required=False)

    price_col = find_column(
        df,
        ["On Demand Filled", "price_usd_per_hour", "On Demand Numeric", "On Demand"],
        required=False,
    )
    reserved_price_col = find_column(
        df,
        ["Linux Reserved cost Filled", "Linux Reserved cost", "Linux Reserved Numeric"],
        required=False,
    )
    spot_price_col = find_column(
        df,
        ["Linux Spot Average cost Filled", "Linux Spot Average cost", "Linux Spot Average Numeric"],
        required=False,
    )

    instance_type_col = find_column(
        df,
        ["API Name", "Instance type", "Instance Type", "instance_type", "Name"],
        required=True,
    )

    df["gpu_model_clean"] = df[gpu_model_col].apply(normalize_gpu_model) if gpu_model_col else pd.NA
    df["cpu_model_clean"] = df[cpu_model_col].apply(normalize_cpu_model) if cpu_model_col else pd.NA
    df["gpu_count"] = df[gpu_count_col].apply(parse_numeric) if gpu_count_col else pd.NA
    df["vcpu_count"] = df[vcpu_col].apply(parse_numeric) if vcpu_col else pd.NA
    df["memory_gib"] = df[memory_col].apply(parse_memory_gib) if memory_col else pd.NA

    df["price_usd_per_hour"] = df[price_col].apply(parse_price) if price_col else pd.NA
    df["reserved_price_usd_per_hour"] = df[reserved_price_col].apply(parse_price) if reserved_price_col else pd.NA
    df["spot_price_usd_per_hour"] = df[spot_price_col].apply(parse_price) if spot_price_col else pd.NA

    df["gpu_count"] = pd.to_numeric(df["gpu_count"], errors="coerce")
    df["vcpu_count"] = pd.to_numeric(df["vcpu_count"], errors="coerce")
    df["memory_gib"] = pd.to_numeric(df["memory_gib"], errors="coerce")
    df["price_usd_per_hour"] = pd.to_numeric(df["price_usd_per_hour"], errors="coerce")
    df["reserved_price_usd_per_hour"] = pd.to_numeric(df["reserved_price_usd_per_hour"], errors="coerce")
    df["spot_price_usd_per_hour"] = pd.to_numeric(df["spot_price_usd_per_hour"], errors="coerce")

    cpu_lookup_df = df["cpu_model_clean"].apply(cpu_lookup_record).apply(pd.Series)
    df = pd.concat([df, cpu_lookup_df], axis=1)

    df["gpu_tdp_w_per_gpu"] = df["gpu_model_clean"].map(GPU_TDP_W)
    df["gpu_tdp_w_per_gpu"] = pd.to_numeric(df["gpu_tdp_w_per_gpu"], errors="coerce")
    df["cpu_tdp_w_per_vcpu"] = pd.to_numeric(df["cpu_tdp_w_per_vcpu"], errors="coerce")

    storage_details = (
        df[storage_col].apply(parse_instance_storage).apply(pd.Series)
        if storage_col
        else pd.DataFrame(index=df.index)
    )
    df = pd.concat([df, storage_details], axis=1)

    ssd_proxy_df = df.apply(
        lambda row: select_ssd_proxy(
            storage_total_gb=float(row.get("storage_total_gb", 0.0) or 0.0),
            drive_count_explicit=row.get("storage_drive_count_explicit", pd.NA),
            is_local_ssd=bool(row.get("storage_is_local_ssd", False)),
        ),
        axis=1,
    ).apply(pd.Series)
    df = pd.concat([df, ssd_proxy_df], axis=1)

    df["storage_total_tb"] = pd.to_numeric(df["storage_total_gb"], errors="coerce").fillna(0) / 1000.0

    # SSD proxy catalog is now interpreted primarily as watts per TB.
    # Backward-compatible fallback order:
    #   1) idle_w_per_tb / active_w_per_tb
    #   2) idle_w_tb / active_w_tb
    #   3) idle_w / active_w   (treated as per-TB values)
    df["ssd_idle_w_per_tb"] = _first_present_numeric(
        df,
        ["idle_w_per_tb", "idle_w_tb", "idle_w"],
        default=0.0,
    )
    df["ssd_active_w_per_tb"] = _first_present_numeric(
        df,
        ["active_w_per_tb", "active_w_tb", "active_w"],
        default=0.0,
    )

    df["ssd_idle_w_total"] = df["storage_total_tb"] * df["ssd_idle_w_per_tb"]
    df["ssd_active_w_total"] = df["storage_total_tb"] * df["ssd_active_w_per_tb"]
    df["ram_idle_w_total"] = df["memory_gib"].fillna(0) * RAM_IDLE_W_PER_GB
    df["ram_active_w_total"] = df["memory_gib"].fillna(0) * RAM_ACTIVE_W_PER_GB

    df["gpu_mapping_status"] = df["gpu_tdp_w_per_gpu"].apply(
        lambda x: "resolved" if pd.notna(x) else "missing"
    )

    scenario_frames = []
    for case_name, cfg in SCENARIO_CONFIG.items():
        cpu_util = float(cfg["cpu_utilization"])
        gpu_util = float(cfg["gpu_utilization"])
        ssd_state = str(cfg["ssd_state"]).lower()
        ram_state = str(cfg["ram_state"]).lower()

        out = df.copy()
        out["scenario"] = case_name
        out["cpu_utilization"] = cpu_util
        out["gpu_utilization"] = gpu_util
        out["ssd_state"] = ssd_state
        out["ram_state"] = ram_state

        out["total_cpu_power_w"] = (
            out["vcpu_count"].fillna(0)
            * out["cpu_tdp_w_per_vcpu"].fillna(0)
            * (cpu_util / PUE_FACTOR)
        )
        out["total_gpu_power_w"] = (
            out["gpu_count"].fillna(0)
            * out["gpu_tdp_w_per_gpu"].fillna(0)
            * (gpu_util / PUE_FACTOR)
        )
        out["total_ram_power_w"] = out["memory_gib"].fillna(0).apply(
            lambda x: scenario_ram_w(x, ram_state)
        )
        out["total_ssd_power_w"] = out.apply(
            lambda r: scenario_ssd_w(r["ssd_idle_w_total"], r["ssd_active_w_total"], ssd_state),
            axis=1,
        )
        out["other_components_power_w"] = OTHER_COMPONENTS_FACTOR * (
            out["total_gpu_power_w"]
            + out["total_cpu_power_w"]
            + out["total_ram_power_w"]
            + out["total_ssd_power_w"]
        )
        out["total_power_w"] = (
            out["total_cpu_power_w"]
            + out["total_gpu_power_w"]
            + out["total_ram_power_w"]
            + out["total_ssd_power_w"]
            + out["other_components_power_w"]
        )
        out["total_power_mw"] = out["total_power_w"] / 1_000_000.0

        out["voll_usd_per_mwh"] = pd.NA
        out["voll_reserved_usd_per_mwh"] = pd.NA
        out["voll_spot_usd_per_mwh"] = pd.NA

        valid = (
            out["price_usd_per_hour"].notna()
            & out["total_power_mw"].notna()
            & (out["total_power_mw"] > 0)
        )
        out.loc[valid, "voll_usd_per_mwh"] = (
            out.loc[valid, "price_usd_per_hour"] / out.loc[valid, "total_power_mw"]
        )

        valid_reserved = (
            out["reserved_price_usd_per_hour"].notna()
            & out["total_power_mw"].notna()
            & (out["total_power_mw"] > 0)
        )
        out.loc[valid_reserved, "voll_reserved_usd_per_mwh"] = (
            out.loc[valid_reserved, "reserved_price_usd_per_hour"] / out.loc[valid_reserved, "total_power_mw"]
        )

        valid_spot = (
            out["spot_price_usd_per_hour"].notna()
            & out["total_power_mw"].notna()
            & (out["total_power_mw"] > 0)
        )
        out.loc[valid_spot, "voll_spot_usd_per_mwh"] = (
            out.loc[valid_spot, "spot_price_usd_per_hour"] / out.loc[valid_spot, "total_power_mw"]
        )

        scenario_frames.append(out)

    long_df = pd.concat(scenario_frames, ignore_index=True)

    long_file = output_path("aws_gpu_cpu_ram_ssd_power_and_value_long.csv")
    long_df.to_csv(long_file, index=False)

    summary_cols = [
        instance_type_col,
        "scenario",
        "price_usd_per_hour",
        "reserved_price_usd_per_hour",
        "spot_price_usd_per_hour",
        "cpu_utilization",
        "gpu_utilization",
        "total_cpu_power_w",
        "total_gpu_power_w",
        "total_ram_power_w",
        "total_ssd_power_w",
        "other_components_power_w",
        "total_power_w",
        "total_power_mw",
        "voll_usd_per_mwh",
        "voll_reserved_usd_per_mwh",
        "voll_spot_usd_per_mwh",
        "proxy_model",
        "estimated_drive_count",
        "estimated_per_drive_gb",
        "cpu_mapping_status",
        "gpu_mapping_status",
    ]

    summary_cols = [c for c in summary_cols if c in long_df.columns]
    wide_df = long_df[[instance_type_col] + [c for c in summary_cols if c != instance_type_col]].copy()
    wide_df = wide_df.pivot(index=instance_type_col, columns="scenario")
    wide_df.columns = [f"{metric}_{scenario}" for metric, scenario in wide_df.columns]
    wide_df = wide_df.reset_index()

    base_cols = [
        instance_type_col,
        gpu_model_col,
        cpu_model_col,
        gpu_count_col,
        vcpu_col,
        memory_col,
        storage_col,
        price_col,
        reserved_price_col,
        spot_price_col,
        "gpu_model_clean",
        "cpu_model_clean",
        "gpu_count",
        "vcpu_count",
        "memory_gib",
        "price_usd_per_hour",
        "reserved_price_usd_per_hour",
        "spot_price_usd_per_hour",
        "gpu_tdp_w_per_gpu",
        "cpu_tdp_w_package",
        "threads_per_package",
        "cpu_tdp_w_per_vcpu",
        "storage_is_local_ssd",
        "storage_total_gb",
        "storage_total_tb",
        "storage_drive_count_explicit",
        "proxy_model",
        "proxy_capacity_gb",
        "estimated_drive_count",
        "estimated_per_drive_gb",
        "idle_w_per_tb",
        "active_w_per_tb",
        "ssd_idle_w_per_tb",
        "ssd_active_w_per_tb",
        "idle_w",
        "active_w",
        "ssd_idle_w_total",
        "ssd_active_w_total",
        "ram_idle_w_total",
        "ram_active_w_total",
        "cpu_mapping_status",
        "gpu_mapping_status",
        "cpu_source_type",
        "cpu_source_name",
        "cpu_source_note",
        "source_type",
        "source_note",
    ]
    base_cols = [c for c in base_cols if c is not None and c in df.columns]
    wide_base = df[base_cols].drop_duplicates(subset=[instance_type_col])
    wide_out = wide_base.merge(wide_df, on=instance_type_col, how="left")

    wide_file = output_path("aws_gpu_cpu_ram_ssd_power_and_value_wide.csv")
    wide_out.to_csv(wide_file, index=False)

    review_mask = (
        (df["gpu_model_clean"].notna() & df["gpu_tdp_w_per_gpu"].isna())
        | (df["cpu_model_clean"].notna() & (df["cpu_mapping_status"] != "resolved"))
    )
    review_file = output_path("ec2_unresolved_power_mappings_review.csv")
    df.loc[review_mask, base_cols].to_csv(review_file, index=False)

    print(f"Saved → {long_file}")
    print(f"Saved → {wide_file}")
    print(f"Saved → {review_file}")
    print(f"Rows: {len(df)}")
    print(f"Scenario rows: {len(long_df)}")
    print(f"Rows with On-Demand VOLL computed: {long_df['voll_usd_per_mwh'].notna().sum()}")
    print(f"Rows with Reserved VOLL computed: {long_df['voll_reserved_usd_per_mwh'].notna().sum()}")
    print(f"Rows with Spot VOLL computed: {long_df['voll_spot_usd_per_mwh'].notna().sum()}")


if __name__ == "__main__":
    main()