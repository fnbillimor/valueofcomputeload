from pathlib import Path
import pandas as pd
import re

# ============================================================
# Fixed project paths
# ============================================================
PROJECT_ROOT = Path(
    r"C:\Users\FarhadBillimoria\OneDrive - Aurora Energy Research\Documents\Personal\Endog_demand_AI\pyai"
)

INSTANCES_DIR = PROJECT_ROOT / "data" / "processed" / "instances"
INSTANCES_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = INSTANCES_DIR / "aws_gpu_filtered_with_vantage_prices.csv"


# ============================================================
# IO helpers
# ============================================================
def load_csv(path=None):
    path = Path(path) if path is not None else INPUT_FILE

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    return pd.read_csv(path)


def output_path(filename: str) -> Path:
    return INSTANCES_DIR / filename


# ============================================================
# Generic helpers
# ============================================================
def find_column(df, candidates, required=True):
    for col in candidates:
        if col in df.columns:
            return col

    if required:
        raise ValueError(f"Could not find any of these columns: {candidates}")

    return None


def parse_numeric(val):
    if pd.isna(val):
        return pd.NA

    s = str(val).strip()
    if s == "":
        return pd.NA

    s = s.replace(",", "")
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", s)

    if match:
        return float(match.group(1))

    return pd.NA


def parse_price(val):
    if pd.isna(val):
        return pd.NA

    s = str(val).strip()
    if s == "":
        return pd.NA

    s = s.replace(",", "")
    s = s.replace("$", "")
    s = s.lower().replace("hourly", "").strip()

    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", s)

    if match:
        return float(match.group(1))

    return pd.NA


# ============================================================
# GPU normalization and lookup
# ============================================================
GPU_NORMALIZATION_MAP = {
    "NVIDIA A100": "A100",
    "NVIDIA A10G": "A10G",
    "NVIDIA B200": "B200",
    "NVIDIA B300": "B300",
    "NVIDIA H100": "H100",
    "NVIDIA H200": "H200",
    "NVIDIA L4": "L4",
    "NVIDIA L40S": "L40S",
    "NVIDIA T4 Tensor Core": "T4",
    "NVIDIA T4G Tensor Core": "T4G",
    "NVIDIA Tesla K80": "K80",
    "NVIDIA Tesla M60": "M60",
    "NVIDIA Tesla V100": "V100",
    "NVIDIA GRID K520": "K520",
    "NVIDIA GB202": "GB202",
    "AMD Radeon Pro V520": "V520",
    "Qualcomm AI 100 Accelerators": "QA100",
    "AWS Trainium2": "AWST2",
    "AWS Inferentia": "Inf1",
    "AWS Inferentia2": "Inf2",    
    
}

GPU_TDP_W = {
    "T4": 70,
    "T4G": 70,
    "L4": 72,
    "L40S": 350,
    "A10G": 300,
    "K80": 300,
    "M60": 300,
    "V100": 300,
    "A100": 400,
    "H100": 700,
    "H200": 700,
    "B200": 1000,
    "B300": 1100,
    # Best-effort additions for extra rows in the dataset
    "K520": 225,
    "GB202": 600,
    "V520": 225,
    "QA100": 75,
    "AWST2": 500,
    "Inf1": 75,
    "Inf2": 100,
    
    
    
}

GPU_SOURCE_TYPE = {
    "T4": "official_datasheet",
    "T4G": "official_datasheet",
    "L4": "official_datasheet",
    "L40S": "official_datasheet",
    "A10G": "official_vendor_spec",
    "K80": "official_datasheet",
    "M60": "official_datasheet",
    "V100": "official_vendor_spec",
    "A100": "official_vendor_spec",
    "H100": "official_vendor_spec",
    "H200": "official_vendor_spec",
    "B200": "official_vendor_spec",
    "B300": "official_vendor_spec",
    "K520": "common_reference",
    "GB202": "common_reference",
    "V520": "common_reference",
    "QA100": "common_reference",
    "AWST2": "AWS public statement inference",
    "Inf1": "Public third-party",
    "Inf2": "Public third-party",
}


def normalize_gpu_model(val):
    if pd.isna(val):
        return pd.NA

    s = str(val).strip()
    if s == "":
        return pd.NA

    if s in GPU_NORMALIZATION_MAP:
        return GPU_NORMALIZATION_MAP[s]

    s = s.replace("NVIDIA", "").replace("Tesla", "").strip()
    s = s.replace("Tensor Core", "").strip()

    return s if s != "" else pd.NA


# ============================================================
# CPU normalization and lookup
# Relaxed sourcing: allow official specs, public cloud docs,
# and common benchmark/reference sites when official TDP is not
# available for AWS-custom or generic family labels.
# ============================================================
def normalize_cpu_model(val):
    if pd.isna(val):
        return pd.NA

    s = str(val).strip()
    if s == "":
        return pd.NA

    s = re.sub(r"\s+", " ", s)
    s = s.replace("  ", " ").strip()

    # Keep canonical labels stable
    if "NVIDIA Grace" in s:
        return "NVIDIA Grace"
    if "AWS Graviton2" in s:
        return "AWS Graviton2 Processor"
    if "AMD EPYC 7R13" in s:
        return "AMD EPYC 7R13 Processor"
    if "AMD EPYC 7R32" in s:
        return "AMD EPYC 7R32"
    if "Intel Xeon Platinum 8275CL" in s:
        return "Intel Xeon Platinum 8275CL (Cascade Lake)"
    if "Intel Xeon Platinum 8275L" in s:
        return "Intel Xeon Platinum 8275L"
    if "Intel Xeon Platinum 8259" in s:
        return "Intel Xeon Platinum 8259 (Cascade Lake)"
    if "Intel Xeon Platinum 8175" in s:
        return "Intel Xeon Platinum 8175 (Skylake)"
    if "Intel Xeon E5-2686 v4" in s:
        return "Intel Xeon E5-2686 v4 (Broadwell)"
    if "Intel Xeon E5-2670" in s:
        return "Intel Xeon E5-2670 (Sandy Bridge)"
    if "Intel Xeon Scalable (Icelake)" in s:
        return "Intel Xeon Scalable (Icelake)"
    if "Intel Xeon Scalable (Sapphire Rapids)" in s:
        return "Intel Xeon Scalable (Sapphire Rapids)"
    if "Intel Xeon Scalable (Emerald Rapids)" in s:
        return "Intel Xeon Scalable (Emerald Rapids)"
    if "Intel Xeon Family" in s:
        return "Intel Xeon Family"

    return s


CPU_LOOKUP = {
    # model: package_tdp_w, threads_per_package, source_type, source_name, source_note
    "NVIDIA Grace": {
        "cpu_tdp_w_package": 500.0,
        "threads_per_package": 72.0,
        "source_type": "official_vendor_spec",
        "source_name": "NVIDIA Grace CPU Superchip product architecture/specs",
        "source_note": "Direct vendor value.",
    },
    "AWS Graviton2 Processor": {
        "cpu_tdp_w_package": 110.0,
        "threads_per_package": 64.0,
        "source_type": "secondary_reference",
        "source_name": "AWS Graviton2 public specs plus benchmark study",
        "source_note": "Best-effort package TDP proxy for Graviton2; no single public AWS TDP figure.",
    },
    "AMD EPYC 7R13 Processor": {
        "cpu_tdp_w_package": 280.0,
        "threads_per_package": 96.0,
        "source_type": "benchmark_proxy",
        "source_name": "Closest EPYC Milan custom-SKU benchmark/reference listings",
        "source_note": "AWS custom Milan part; package TDP inferred from common benchmark/reference sources.",
    },
    "AMD EPYC 7R32": {
        "cpu_tdp_w_package": 280.0,
        "threads_per_package": 96.0,
        "source_type": "benchmark_proxy",
        "source_name": "Closest EPYC Rome custom-SKU benchmark/reference listings",
        "source_note": "AWS custom Rome part; package TDP inferred from common benchmark/reference sources.",
    },
    "Intel Xeon Platinum 8259 (Cascade Lake)": {
        "cpu_tdp_w_package": 210.0,
        "threads_per_package": 48.0,
        "source_type": "benchmark_proxy",
        "source_name": "Closest Cascade Lake benchmark/reference listings",
        "source_note": "Custom AWS-adjacent SKU; inferred from common reference sites.",
    },
    "Intel Xeon Platinum 8275CL (Cascade Lake)": {
        "cpu_tdp_w_package": 210.0,
        "threads_per_package": 48.0,
        "source_type": "benchmark_proxy",
        "source_name": "Closest Cascade Lake benchmark/reference listings",
        "source_note": "Custom AWS SKU; inferred from common reference sites.",
    },
    "Intel Xeon Platinum 8275L": {
        "cpu_tdp_w_package": 165.0,
        "threads_per_package": 48.0,
        "source_type": "benchmark_proxy",
        "source_name": "Closest Cascade Lake benchmark/reference listings",
        "source_note": "No clean public Intel datasheet for exact AWS context; proxy used.",
    },
    "Intel Xeon Platinum 8175 (Skylake)": {
        "cpu_tdp_w_package": 240.0,
        "threads_per_package": 48.0,
        "source_type": "benchmark_proxy",
        "source_name": "Closest Skylake Xeon benchmark/reference listings",
        "source_note": "Proxy package TDP from common reference sources of 8175M.",
    },
    "Intel Xeon E5-2686 v4 (Broadwell)": {
        "cpu_tdp_w_package": 145.0,
        "threads_per_package": 36.0,
        "source_type": "benchmark_proxy",
        "source_name": "Closest Broadwell Xeon benchmark/reference listings",
        "source_note": "AWS-associated custom part; Direct vendor value.",
    },
    "Intel Xeon E5-2670 (Sandy Bridge)": {
        "cpu_tdp_w_package": 115.0,
        "threads_per_package": 16.0,
        "source_type": "benchmark_proxy",
        "source_name": "Common Sandy Bridge Xeon benchmark/reference listings",
        "source_note": "Direct vendor value.",
    },
    "Intel Xeon Scalable (Icelake)": {
        "cpu_tdp_w_package": 270.0,
        "threads_per_package": 64.0,
        "source_type": "family_proxy",
        "source_name": "Representative Ice Lake Xeon benchmark/reference listings",
        "source_note": "Generic family label in source data; representative family proxy used.",
    },
    "Intel Xeon Scalable (Sapphire Rapids)": {
        "cpu_tdp_w_package": 205.0,
        "threads_per_package": 64.0,
        "source_type": "family_proxy",
        "source_name": "Representative Sapphire Rapids Xeon benchmark/reference listings",
        "source_note": "Generic family label in source data; representative family proxy used.",
    },
    "Intel Xeon Scalable (Emerald Rapids)": {
        "cpu_tdp_w_package": 350.0,
        "threads_per_package": 96.0,
        "source_type": "family_proxy",
        "source_name": "Representative Emerald Rapids Xeon benchmark/reference listings",
        "source_note": "Generic family label in source data; representative family proxy used.",
    },
    "Intel Xeon Family": {
        "cpu_tdp_w_package": 250.0,
        "threads_per_package": 32.0,
        "source_type": "family_proxy",
        "source_name": "Generic Xeon family proxy from common reference sites",
        "source_note": "Very generic label; conservative proxy used.",
    },
}


def cpu_lookup_record(cpu_model):
    if pd.isna(cpu_model):
        return {
            "cpu_tdp_w_package": pd.NA,
            "threads_per_package": pd.NA,
            "cpu_tdp_w_per_vcpu": pd.NA,
            "cpu_source_type": pd.NA,
            "cpu_source_name": pd.NA,
            "cpu_source_note": pd.NA,
            "cpu_mapping_status": "missing",
        }

    rec = CPU_LOOKUP.get(str(cpu_model))
    if rec is None:
        return {
            "cpu_tdp_w_package": pd.NA,
            "threads_per_package": pd.NA,
            "cpu_tdp_w_per_vcpu": pd.NA,
            "cpu_source_type": pd.NA,
            "cpu_source_name": pd.NA,
            "cpu_source_note": pd.NA,
            "cpu_mapping_status": "review_required",
        }

    tdp = rec["cpu_tdp_w_package"]
    threads = rec["threads_per_package"]
    per_vcpu = pd.NA if not threads else tdp / threads

    return {
        "cpu_tdp_w_package": tdp,
        "threads_per_package": threads,
        "cpu_tdp_w_per_vcpu": per_vcpu,
        "cpu_source_type": rec["source_type"],
        "cpu_source_name": rec["source_name"],
        "cpu_source_note": rec["source_note"],
        "cpu_mapping_status": "resolved" if pd.notna(tdp) else "review_required",
    }
