import pandas as pd

from ec2_power_common import load_csv, output_path, find_column, normalize_cpu_model, cpu_lookup_record



def main():
    df = load_csv()
    cpu_col = find_column(df, ["Physical Processor", "CPU model", "CPU Model", "Processor", "CPU"], required=False)
    if cpu_col is None:
        raise ValueError("No CPU column found.")

    out = (
        df[cpu_col]
        .apply(normalize_cpu_model)
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .to_frame(name="cpu_model_clean")
        .reset_index(drop=True)
    )
    lookup_df = out["cpu_model_clean"].apply(cpu_lookup_record).apply(pd.Series)
    out = pd.concat([out, lookup_df], axis=1)

    output_file = output_path("ec2_cpu_groups_tdp.csv")
    out.to_csv(output_file, index=False)
    print(f"Saved → {output_file}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
