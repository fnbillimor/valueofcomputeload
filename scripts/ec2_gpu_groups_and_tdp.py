import pandas as pd

from ec2_power_common import load_csv, output_path, find_column, normalize_gpu_model, GPU_TDP_W, GPU_SOURCE_TYPE



def main():
    df = load_csv()
    gpu_col = find_column(df, ["GPU model", "GPU Model", "Gpu model"], required=False)
    if gpu_col is None:
        raise ValueError("No GPU model column found.")

    out = (
        df[gpu_col]
        .apply(normalize_gpu_model)
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .to_frame(name="gpu_model_clean")
        .reset_index(drop=True)
    )
    out["gpu_tdp_w_per_gpu"] = out["gpu_model_clean"].map(GPU_TDP_W)
    out["gpu_source_type"] = out["gpu_model_clean"].map(GPU_SOURCE_TYPE)
    out["status"] = out["gpu_tdp_w_per_gpu"].apply(lambda x: "resolved" if pd.notna(x) else "review_required")

    output_file = output_path("ec2_gpu_groups_tdp.csv")
    out.to_csv(output_file, index=False)
    print(f"Saved → {output_file}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
