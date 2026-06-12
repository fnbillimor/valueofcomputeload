from pathlib import Path
import re
import pandas as pd
import numpy as np


# -----------------------------
# PATHS
# -----------------------------
INPUT_DIR = Path(
    r"C:\Users\FarhadBillimoria\OneDrive - Aurora Energy Research\Documents\Personal\Endog_demand_AI\pyai\data\original\mlperf_training"
)

CPU_TDP_FILE = Path(
    r"C:\Users\FarhadBillimoria\OneDrive - Aurora Energy Research\Documents\Personal\Endog_demand_AI\pyai\data\original\mlperf_forms\CPU_TDP_MLPERF.csv"
)

GPU_TDP_FILE = Path(
    r"C:\Users\FarhadBillimoria\OneDrive - Aurora Energy Research\Documents\Personal\Endog_demand_AI\pyai\data\original\mlperf_forms\GPU_TDP_MLPERF.csv"
)

OUTPUT_FILE = Path(
    r"C:\Users\FarhadBillimoria\OneDrive - Aurora Energy Research\Documents\Personal\Endog_demand_AI\pyai\data\processed\mlperf\mlperf_training_all.csv"
)


# -----------------------------
# HELPERS
# -----------------------------
def read_mlperf_csv(path):
    return pd.read_csv(path, encoding="utf-16", sep="\t")


def clean_num(x):
    if pd.isna(x):
        return np.nan
    return pd.to_numeric(str(x).replace(",", "").strip(), errors="coerce")


def extract_gpu_name(text):
    if pd.isna(text):
        return None

    s = str(text)

    for gpu in [
        "H200", "H100", "A100",
        "B200", "B300",
        "L40S", "L4",
        "A10G", "T4G", "T4", "V100"
    ]:
        if re.search(rf"\b{gpu}\b", s, flags=re.IGNORECASE):
            return f"NVIDIA {gpu}"

    return None


def extract_gpu_form_factor(text):
    if pd.isna(text):
        return None

    s = str(text)

    if re.search(r"\bPCIe\b|\bPCIE\b", s, flags=re.IGNORECASE):
        return "PCIe"

    if re.search(r"\bSXM", s, flags=re.IGNORECASE):
        return "SXM"

    return None


def extract_gpu_memory(text):
    if pd.isna(text):
        return None

    m = re.search(r"(\d+)\s*GB", str(text), flags=re.IGNORECASE)
    return float(m.group(1)) if m else None


# -----------------------------
# LOAD TDP LOOKUPS
# -----------------------------
cpu_tdp = pd.read_csv(CPU_TDP_FILE)
gpu_tdp = pd.read_csv(GPU_TDP_FILE)

cpu_tdp["CPU"] = cpu_tdp["CPU"].astype(str).str.strip()
cpu_tdp["TDP"] = pd.to_numeric(cpu_tdp["TDP"], errors="coerce")

gpu_tdp["GPU"] = gpu_tdp["GPU"].astype(str).str.strip()
gpu_tdp["Form Factor"] = gpu_tdp["Form Factor"].astype(str).str.strip()
gpu_tdp["GPU Memory"] = pd.to_numeric(gpu_tdp["GPU Memory"], errors="coerce")
gpu_tdp["TDP"] = pd.to_numeric(gpu_tdp["TDP"], errors="coerce")


# -----------------------------
# PARSE ONE FILE
# -----------------------------
def parse_training_file(input_file: Path) -> pd.DataFrame:
    df = read_mlperf_csv(input_file)

    static_headers = list(df.iloc[2, 0:17])
    data = df.iloc[3:].reset_index(drop=True)

    records = []

    for start in range(0, len(data), 6):
        block = data.iloc[start:start + 6]

        if len(block) < 6:
            print(f"  Skipping incomplete block at row {start}")
            continue

        base = block.iloc[0]

        static = {}
        for i, name in enumerate(static_headers):
            static[name] = base.iloc[i]

        for col in range(18, df.shape[1], 2):
            benchmark = df.iloc[1, col]

            kj_col = col
            latency_col = col + 1

            if latency_col >= df.shape[1]:
                continue

            kj = clean_num(block.iloc[1, kj_col])
            nodes = clean_num(block.iloc[2, kj_col])
            latency_minutes = clean_num(block.iloc[4, latency_col])

            if pd.isna(kj) and pd.isna(latency_minutes):
                continue

            watt_hours = kj / 3.6
            latency_hours = latency_minutes / 60
            watts = watt_hours / latency_hours
            kw = watts / 1000

            row = dict(static)

            row["Source File"] = input_file.name
            row["Benchmark Model"] = benchmark
            row["kJ"] = kj
            row["Latency_minutes"] = latency_minutes
            row["Number Of Nodes"] = nodes
            row["Watt_hours"] = watt_hours
            row["Latency_hours"] = latency_hours
            row["Watts"] = watts
            row["kW"] = kw

            records.append(row)

    return pd.DataFrame(records)


# -----------------------------
# ADD TDP + LOAD FACTOR
# -----------------------------
def add_tdp_columns(out: pd.DataFrame) -> pd.DataFrame:
    out = out.copy()

    out["CPU"] = out["Host Processor Model Name"].astype(str).str.strip()

    out["GPU"] = out["Accelerator Model Name"].apply(extract_gpu_name)
    out["GPU Form Factor"] = out["Accelerator Model Name"].apply(extract_gpu_form_factor)
    out["GPU Memory GB"] = out["Accelerator Model Name"].apply(extract_gpu_memory)

    # CPU exact match
    out = out.merge(
        cpu_tdp.rename(columns={"TDP": "CPU_TDP_W"}),
        on="CPU",
        how="left"
    )

    # GPU matching
    variant_gpus = {"NVIDIA A100", "NVIDIA H100", "NVIDIA H200"}

    out_variant = out[out["GPU"].isin(variant_gpus)].copy()
    out_nonvariant = out[~out["GPU"].isin(variant_gpus)].copy()

    gpu_variant = gpu_tdp[gpu_tdp["GPU"].isin(variant_gpus)].copy()
    gpu_nonvariant = gpu_tdp[~gpu_tdp["GPU"].isin(variant_gpus)].copy()

    # Strict match for A100/H100/H200
    out_variant = out_variant.merge(
        gpu_variant.rename(
            columns={
                "Form Factor": "GPU Form Factor",
                "GPU Memory": "GPU Memory GB",
                "TDP": "GPU_TDP_W",
            }
        )[["GPU", "GPU Form Factor", "GPU Memory GB", "GPU_TDP_W"]],
        on=["GPU", "GPU Form Factor", "GPU Memory GB"],
        how="left"
    )

    # Simple GPU-name match for others
    gpu_nonvariant_simple = (
        gpu_nonvariant[["GPU", "TDP"]]
        .drop_duplicates("GPU")
        .rename(columns={"TDP": "GPU_TDP_W"})
    )

    out_nonvariant = out_nonvariant.merge(
        gpu_nonvariant_simple,
        on="GPU",
        how="left"
    )

    out = pd.concat([out_variant, out_nonvariant], ignore_index=True)

    # Totals
    out["Host Processors Per Node"] = pd.to_numeric(
        out["Host Processors Per Node"], errors="coerce"
    )

    out["Total Accelerators"] = pd.to_numeric(
        out["Total Accelerators"], errors="coerce"
    )

    out["Number Of Nodes"] = pd.to_numeric(
        out["Number Of Nodes"], errors="coerce"
    )

    out["Total CPUs"] = out["Host Processors Per Node"] * out["Number Of Nodes"]
    out["Total GPUs"] = out["Total Accelerators"]

    out["Total CPU TDP"] = out["CPU_TDP_W"] * out["Total CPUs"]
    out["Total GPU TDP"] = out["GPU_TDP_W"] * out["Total GPUs"]
    out["Total TDP"] = out["Total CPU TDP"] + out["Total GPU TDP"]

    out["Watts / CPU+GPU TDP"] = out["Watts"] / out["Total TDP"]

    out["TDP ratio flag"] = np.select(
        [
            out["Watts / CPU+GPU TDP"] < 1.5,
            out["Watts / CPU+GPU TDP"] < 2.0,
        ],
        [
            "ok",
            "check",
        ],
        default="high"
    )

    return out


# -----------------------------
# MAIN
# -----------------------------
def main():
    files = sorted(INPUT_DIR.glob("Table - Training*.csv"))

    if not files:
        raise FileNotFoundError(f"No training files found in {INPUT_DIR}")

    all_outputs = []
    summary = []

    for f in files:
        print(f"\nProcessing {f.name}")

        try:
            parsed = parse_training_file(f)

            if parsed.empty:
                print("  No rows parsed")
                summary.append({
                    "file": f.name,
                    "status": "empty",
                    "rows": 0,
                    "error": "",
                })
                continue

            parsed = add_tdp_columns(parsed)

            print(f"  Rows parsed: {len(parsed)}")
            print(f"  Missing CPU TDP: {parsed['CPU_TDP_W'].isna().sum()}")
            print(f"  Missing GPU TDP: {parsed['GPU_TDP_W'].isna().sum()}")
            print(f"  High TDP flags: {(parsed['TDP ratio flag'] == 'high').sum()}")

            all_outputs.append(parsed)

            summary.append({
                "file": f.name,
                "status": "ok",
                "rows": len(parsed),
                "error": "",
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            summary.append({
                "file": f.name,
                "status": "error",
                "rows": 0,
                "error": str(e),
            })

    if not all_outputs:
        raise RuntimeError("No files successfully parsed.")

    final = pd.concat(all_outputs, ignore_index=True, sort=False)

    sort_cols = [
        c for c in [
            "Source File",
            "Public ID",
            "Organization",
            "System Name (Click + for details)",
            "Benchmark",
        ]
        if c in final.columns
    ]

    final = final.sort_values(sort_cols).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(OUTPUT_FILE, index=False)

    summary_df = pd.DataFrame(summary)

    print("\n=== FILE SUMMARY ===")
    print(summary_df.to_string(index=False))

    print("\n=== FINAL OUTPUT ===")
    print(f"Rows: {len(final):,}")
    print(f"Columns: {len(final.columns):,}")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\n=== FINAL TDP CHECK ===")
    print("Missing CPU TDP:", int(final["CPU_TDP_W"].isna().sum()))
    print("Missing GPU TDP:", int(final["GPU_TDP_W"].isna().sum()))
    print("High TDP flags:", int((final["TDP ratio flag"] == "high").sum()))

    sample_cols = [
        "Source File",
        "Public ID",
        "Organization",
        "System Name (Click + for details)",
        "Benchmark",
        "kJ",
        "Latency_minutes",
        "Watts",
        "kW",
        "CPU",
        "GPU",
        "GPU Form Factor",
        "GPU Memory GB",
        "Number Of Nodes",
        "Total CPUs",
        "Total GPUs",
        "CPU_TDP_W",
        "GPU_TDP_W",
        "Total TDP",
        "Watts / CPU+GPU TDP",
        "TDP ratio flag",
    ]

    sample_cols = [c for c in sample_cols if c in final.columns]

    print("\n=== SAMPLE ===")
    print(final[sample_cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()