from pathlib import Path
import re
import pandas as pd


INPUT_DIR = Path(
    r"C:\Users\FarhadBillimoria\OneDrive - Aurora Energy Research\Documents\Personal\Endog_demand_AI\pyai\data\original\mlperf"
)

OUTPUT_FILE = Path(
    r"C:\Users\FarhadBillimoria\OneDrive - Aurora Energy Research\Documents\Personal\Endog_demand_AI\pyai\data\processed\mlperf\mlperf_inference_stitched_long.csv"
)


def read_mlperf_csv(path: Path) -> pd.DataFrame:
    attempts = [
        {"encoding": "utf-16", "sep": "\t"},
        {"encoding": "utf-8-sig"},
        {"encoding": "latin1"},
    ]

    last_err = None
    for kwargs in attempts:
        try:
            return pd.read_csv(path, **kwargs)
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Could not read file: {path}\nLast error: {last_err}")


def clean_header_name(name) -> str:
    if pd.isna(name):
        return ""
    s = str(name).strip()
    if s.startswith("Unnamed:"):
        return ""
    return s


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    raw_cols = list(df.columns)
    visible_header_row = df.iloc[3].tolist()

    new_cols = []
    seen = {}

    for i, col in enumerate(raw_cols):
        base = clean_header_name(col)

        if not base and i < len(visible_header_row):
            alt = visible_header_row[i]
            if pd.notna(alt):
                base = str(alt).strip()

        if not base:
            base = f"col_{i+1}"

        count = seen.get(base, 0)
        final_name = base if count == 0 else f"{base}.{count}"
        seen[base] = count + 1
        new_cols.append(final_name)

    out = df.copy()
    out.columns = new_cols
    return out


def extract_gpu_name(accelerator_text):
    if pd.isna(accelerator_text):
        return pd.NA

    s = str(accelerator_text).strip()
    if not s:
        return pd.NA

    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"\s+", " ", s).strip()

    m_tesla = re.search(r"\b(Tesla)\s+([A-Za-z0-9]+)", s, flags=re.IGNORECASE)
    if m_tesla:
        return f"Tesla {m_tesla.group(2).upper()}"

    m_vendor = re.search(r"\b(NVIDIA|AMD|Intel|Habana)\b[\s\-]*([A-Za-z0-9]+)", s, flags=re.IGNORECASE)
    if m_vendor:
        vendor = m_vendor.group(1)
        token = m_vendor.group(2).upper()
        vendor_norm = "NVIDIA" if vendor.lower() == "nvidia" else vendor.title()
        return f"{vendor_norm} {token}"

    parts = s.split()
    if len(parts) >= 2:
        return f"{parts[0]} {parts[1]}"

    return s


def extract_gpu_form_factor(accelerator_text):
    if pd.isna(accelerator_text):
        return pd.NA

    s = str(accelerator_text)
    m = re.search(r"\b(PCIE|PCIe|SXM|SXM2|SXM3|SXM4|SXM5)\b", s, flags=re.IGNORECASE)
    if not m:
        return pd.NA

    ff = m.group(1)
    if ff.upper() == "PCIE":
        return "PCIe"
    return ff.upper()


def extract_gpu_memory(accelerator_text):
    if pd.isna(accelerator_text):
        return pd.NA

    s = str(accelerator_text)
    m = re.search(r"(\d+)\s*GB\b", s, flags=re.IGNORECASE)
    if not m:
        return pd.NA

    return f"{m.group(1)}GB"


def first_nonblank_across_benchmark_cols(row, benchmark_cols):
    for col in benchmark_cols:
        value = row[col]
        if pd.notna(value) and str(value).strip() != "":
            return value
    return pd.NA

def transform_mlperf_file(path: Path) -> pd.DataFrame:
    df = read_mlperf_csv(path)
    df = normalize_columns(df)

    if df.shape[0] < 5:
        raise ValueError(f"File too short to parse expected MLPerf structure: {path.name}")

    header_rows = df.iloc[0:4].copy()
    data = df.iloc[4:].reset_index(drop=True)

    if df.shape[1] < 17:
        raise ValueError(f"File has too few columns to contain benchmark section: {path.name}")

    benchmark_cols = list(df.columns[16:])

    preferred_static_cols = [
        "Public ID",
        "Organization",
        "Availability",
        "System Name (click + for details)",
        "Processor (click + for details)",
        "Processors per node",
        "Accelerator (click + for details)",
        "Accelerator Model Name",
        "Total Accelerators",
        "Division/Power",
    ]

    records = []

    for start in range(0, len(data), 4):
        block = data.iloc[start:start + 4]

        if len(block) < 4:
            continue

        row_nodes = block.iloc[0]
        row_accel = block.iloc[1]
        row_results = block.iloc[2]

        num_nodes = first_nonblank_across_benchmark_cols(row_nodes, benchmark_cols)
        accelerators_per_node = first_nonblank_across_benchmark_cols(row_accel, benchmark_cols)

        accelerator_text = row_nodes.get("Accelerator Model Name", pd.NA)
        if pd.isna(accelerator_text) or str(accelerator_text).strip() == "":
            accelerator_text = row_nodes.get("Accelerator (click + for details)", pd.NA)

        for col in benchmark_cols:
            value = row_results[col]

            if pd.isna(value) or str(value).strip() == "":
                continue

            benchmark = header_rows.iloc[0][col]
            benchmark_model = header_rows.iloc[1][col]
            scenario = header_rows.iloc[2][col]
            metric = header_rows.iloc[3][col]

            if pd.isna(metric) or str(metric).strip() == "":
                continue

            record = {}

            for c in preferred_static_cols:
                record[c] = row_nodes[c] if c in row_nodes.index else pd.NA

            record["Source File"] = path.name
            record["# of Nodes"] = num_nodes
            record["Accelerators per node"] = accelerators_per_node
            record["GPU"] = extract_gpu_name(accelerator_text)
            record["GPU Form Factor"] = extract_gpu_form_factor(accelerator_text)
            record["GPU Memory"] = extract_gpu_memory(accelerator_text)

            record["Benchmark"] = benchmark
            record["Benchmark Model"] = benchmark_model
            record["Scenario"] = scenario
            record["Metric"] = metric
            record["Value"] = value

            records.append(record)

    out = pd.DataFrame(records)

    if out.empty:
        return out

    out = out.rename(
        columns={
            "System Name (click + for details)": "System Name",
            "Processor (click + for details)": "CPU",
            "Accelerator (click + for details)": "GPU Raw",
        }
    )

    if "Value" in out.columns:
        out["Value"] = (
            out["Value"]
            .astype(str)
            .str.replace(",", "", regex=False)
        )
        out["Value"] = pd.to_numeric(out["Value"], errors="coerce")

    if "Processors per node" in out.columns:
        out["Processors per node"] = pd.to_numeric(out["Processors per node"], errors="coerce")

    if "# of Nodes" in out.columns:
        out["# of Nodes"] = pd.to_numeric(out["# of Nodes"], errors="coerce")

    if "Accelerators per node" in out.columns:
        out["Accelerators per node"] = pd.to_numeric(out["Accelerators per node"], errors="coerce")

    if "Total Accelerators" in out.columns:
        out["Total Accelerators"] = pd.to_numeric(out["Total Accelerators"], errors="coerce")

    out["Total Processors"] = out["Processors per node"] * out["# of Nodes"]

    desired_order = [
        "Source File",
        "Public ID",
        "Organization",
        "Availability",
        "System Name",
        "CPU",
        "Processors per node",
        "Total Processors",
        "GPU",
        "GPU Raw",
        "GPU Form Factor",
        "GPU Memory",
        "Accelerator Model Name",
        "Total Accelerators",
        "Division/Power",
        "# of Nodes",
        "Accelerators per node",
        "Benchmark",
        "Benchmark Model",
        "Scenario",
        "Metric",
        "Value",
    ]

    existing_order = [c for c in desired_order if c in out.columns]
    remaining = [c for c in out.columns if c not in existing_order]
    out = out[existing_order + remaining]

    return out


def get_input_files(input_dir: Path):
    files = sorted(input_dir.glob("Table - Inference (*.csv)"))
    if not files:
        files = sorted(input_dir.glob("Table - Inference*.csv"))
    return files


def stitch_all_files(input_dir: Path) -> pd.DataFrame:
    files = get_input_files(input_dir)

    if not files:
        raise FileNotFoundError(f"No matching files found in: {input_dir}")

    frames = []
    summary_rows = []

    for path in files:
        print(f"Reading {path.name}")
        try:
            transformed = transform_mlperf_file(path)
            print(f"  -> rows: {len(transformed):,}")
            frames.append(transformed)
            summary_rows.append(
                {
                    "file": path.name,
                    "status": "ok",
                    "rows": len(transformed),
                    "error": "",
                }
            )
        except Exception as e:
            print(f"  -> ERROR: {e}")
            summary_rows.append(
                {
                    "file": path.name,
                    "status": "error",
                    "rows": 0,
                    "error": str(e),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    print("\n=== File summary ===")
    print(summary_df.to_string(index=False))

    good_frames = [f for f in frames if not f.empty]
    if not good_frames:
        raise RuntimeError("No files were successfully transformed.")

    stitched = pd.concat(good_frames, ignore_index=True, sort=False)
    return stitched


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    stitched = stitch_all_files(INPUT_DIR)

    stitched.to_csv(OUTPUT_FILE, index=False)

    print("\n=== Final stitched output ===")
    print(stitched.head(20).to_string(index=False))
    print(f"\nRows: {len(stitched):,}")
    print(f"Columns: {len(stitched.columns):,}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()


from pathlib import Path
import pandas as pd
import numpy as np


# -----------------------------
# PATHS
# -----------------------------
MLPERF_FILE = Path(
    r"C:\Users\FarhadBillimoria\OneDrive - Aurora Energy Research\Documents\Personal\Endog_demand_AI\pyai\data\processed\mlperf\mlperf_inference_stitched_long.csv"
)

CPU_TDP_FILE = Path(
    r"C:\Users\FarhadBillimoria\OneDrive - Aurora Energy Research\Documents\Personal\Endog_demand_AI\pyai\data\original\mlperf_forms\CPU_TDP_MLPERF.csv"
)

GPU_TDP_FILE = Path(
    r"C:\Users\FarhadBillimoria\OneDrive - Aurora Energy Research\Documents\Personal\Endog_demand_AI\pyai\data\original\mlperf_forms\GPU_TDP_MLPERF.csv"
)

OUTPUT_FILE = Path(
    r"C:\Users\FarhadBillimoria\OneDrive - Aurora Energy Research\Documents\Personal\Endog_demand_AI\pyai\data\processed\mlperf\mlperf_with_tdp.csv"
)


# -----------------------------
# HELPERS
# -----------------------------
VARIANT_GPUS = {"NVIDIA A100", "NVIDIA H100", "NVIDIA H200"}


def clean_text_series(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def clean_form_factor(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    s = s.replace({"PCIE": "PCIe", "pcie": "PCIe", "SXM2": "SXM", "SXM3": "SXM", "SXM4": "SXM", "SXM5": "SXM"})
    return s


def clean_gpu_memory_to_numeric(series: pd.Series) -> pd.Series:
    """
    Convert memory fields like '40GB' or 40.0 into numeric GB values.
    """
    s = series.astype("string").str.strip()
    s = s.str.replace("GB", "", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce")


# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(MLPERF_FILE)
cpu_tdp = pd.read_csv(CPU_TDP_FILE)
gpu_tdp = pd.read_csv(GPU_TDP_FILE)


# -----------------------------
# CLEAN JOIN KEYS
# -----------------------------
df["CPU"] = clean_text_series(df["CPU"])
df["GPU"] = clean_text_series(df["GPU"])
df["GPU Form Factor"] = clean_form_factor(df["GPU Form Factor"])
df["GPU Memory GB"] = clean_gpu_memory_to_numeric(df["GPU Memory"])

cpu_tdp["CPU"] = clean_text_series(cpu_tdp["CPU"])
cpu_tdp["TDP"] = pd.to_numeric(cpu_tdp["TDP"], errors="coerce")

gpu_tdp["GPU"] = clean_text_series(gpu_tdp["GPU"])
gpu_tdp["Form Factor"] = clean_form_factor(gpu_tdp["Form Factor"])
gpu_tdp["GPU Memory GB"] = clean_gpu_memory_to_numeric(gpu_tdp["GPU Memory"])
gpu_tdp["TDP"] = pd.to_numeric(gpu_tdp["TDP"], errors="coerce")


# -----------------------------
# MERGE CPU TDP
# -----------------------------
df = df.merge(
    cpu_tdp.rename(columns={"TDP": "CPU_TDP_W"}),
    on="CPU",
    how="left"
)


# -----------------------------
# MERGE GPU TDP
# -----------------------------
# Split the MLPerf data into:
# 1. GPUs that need variant matching
# 2. GPUs that can match on GPU only

df_variant = df[df["GPU"].isin(VARIANT_GPUS)].copy()
df_nonvariant = df[~df["GPU"].isin(VARIANT_GPUS)].copy()

# Variant GPU lookup: only rows for A100/H100/H200
gpu_tdp_variant = gpu_tdp[gpu_tdp["GPU"].isin(VARIANT_GPUS)].copy()
gpu_tdp_nonvariant = gpu_tdp[~gpu_tdp["GPU"].isin(VARIANT_GPUS)].copy()

# For A100/H100/H200, merge on GPU + Form Factor + Memory
df_variant = df_variant.merge(
    gpu_tdp_variant.rename(
        columns={
            "Form Factor": "GPU Form Factor",
            "TDP": "GPU_TDP_W",
        }
    )[["GPU", "GPU Form Factor", "GPU Memory GB", "GPU_TDP_W"]],
    on=["GPU", "GPU Form Factor", "GPU Memory GB"],
    how="left"
)

# For all other GPUs, merge on GPU only
# If the lookup contains duplicate rows for non-variant GPUs, keep first
gpu_tdp_nonvariant_simple = (
    gpu_tdp_nonvariant[["GPU", "TDP"]]
    .drop_duplicates(subset=["GPU"])
    .rename(columns={"TDP": "GPU_TDP_W"})
)

df_nonvariant = df_nonvariant.merge(
    gpu_tdp_nonvariant_simple,
    on="GPU",
    how="left"
)

# Stitch back together
df = pd.concat([df_variant, df_nonvariant], ignore_index=True, sort=False)


# -----------------------------
# CLEAN NUMERIC FIELDS
# -----------------------------

# Keep only power observations
df = df[df["Metric"] == "System Watts"].copy()

df["Processors per node"] = pd.to_numeric(df["Processors per node"], errors="coerce")
df["# of Nodes"] = pd.to_numeric(df["# of Nodes"], errors="coerce")
df["Accelerators per node"] = pd.to_numeric(df["Accelerators per node"], errors="coerce")
df["Total Accelerators"] = pd.to_numeric(df["Total Accelerators"], errors="coerce")
df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

df["CPU_TDP_W"] = pd.to_numeric(df["CPU_TDP_W"], errors="coerce")
df["GPU_TDP_W"] = pd.to_numeric(df["GPU_TDP_W"], errors="coerce")


# -----------------------------
# DERIVE COUNTS
# -----------------------------
df["Total CPUs"] = df["Processors per node"] * df["# of Nodes"]
df["Total GPUs"] = df["Accelerators per node"] * df["# of Nodes"]


# -----------------------------
# TDP CALCULATION
# -----------------------------
df["Total CPU TDP"] = df["CPU_TDP_W"] * df["Total CPUs"]
df["Total GPU TDP"] = df["GPU_TDP_W"] * df["Total GPUs"]
df["Total TDP"] = df["Total CPU TDP"] + df["Total GPU TDP"]


# -----------------------------
# POWER AS PROPORTION OF CPU+GPU TDP
# -----------------------------
df["Watts / CPU+GPU TDP"] = df["Value"] / df["Total TDP"]

# Only meaningful for System Watts rows
df.loc[df["Metric"] != "System Watts", "Watts / CPU+GPU TDP"] = pd.NA


# -----------------------------
# OPTIONAL: SORT FOR READABILITY
# -----------------------------
sort_cols = [c for c in ["Source File", "Public ID", "Benchmark", "Benchmark Model", "Scenario", "Metric"] if c in df.columns]
if sort_cols:
    df = df.sort_values(sort_cols).reset_index(drop=True)


# -----------------------------
# SAVE
# -----------------------------
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved: {OUTPUT_FILE}")


# -----------------------------
# QUICK VALIDATION
# -----------------------------
print("\n=== Sample rows ===")
sample_cols = [
    c for c in [
        "GPU",
        "GPU Form Factor",
        "GPU Memory",
        "GPU Memory GB",
        "CPU",
        "Total GPUs",
        "Total CPUs",
        "GPU_TDP_W",
        "CPU_TDP_W",
        "Total GPU TDP",
        "Total CPU TDP",
        "Total TDP",
        "Metric",
        "Value",
        "Watts / CPU+GPU TDP",
    ]
    if c in df.columns
]
print(df[sample_cols].head(15).to_string(index=False))

print("\n=== Missing lookup check ===")
print("Missing CPU TDP rows:", int(df["CPU_TDP_W"].isna().sum()))
print("Missing GPU TDP rows:", int(df["GPU_TDP_W"].isna().sum()))

print("\n=== Variant GPU rows with missing TDP ===")
variant_missing = df[df["GPU"].isin(VARIANT_GPUS) & df["GPU_TDP_W"].isna()]
if len(variant_missing) == 0:
    print("None")
else:
    cols = [c for c in ["GPU", "GPU Form Factor", "GPU Memory", "GPU Memory GB", "Source File", "Public ID"] if c in variant_missing.columns]
    print(variant_missing[cols].drop_duplicates().to_string(index=False))