from __future__ import annotations

from pathlib import Path
import math
import os
import re
from typing import Any, Dict, Optional

import pandas as pd


# ============================================================
# Project paths
# ============================================================
# Override with POWER_PROJECT_ROOT if needed.
_DEFAULT_PROJECT_ROOT = Path(
    r"C:\Users\FarhadBillimoria\OneDrive - Aurora Energy Research\Documents\Personal\Endog_demand_AI\pyai"
)
PROJECT_ROOT = Path(os.environ.get("POWER_PROJECT_ROOT", _DEFAULT_PROJECT_ROOT))

INSTANCES_DIR = PROJECT_ROOT / "data" / "processed" / "instances"
ORIGINAL_INSTANCES_DIR = PROJECT_ROOT / "data" / "original" / "instances"
INSTANCES_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = INSTANCES_DIR / "aws_gpu_filtered_with_vantage_prices.csv"
FALLBACK_INPUT_FILE = INSTANCES_DIR / "aws_gpu_filtered.csv"


# ============================================================
# Scenario configuration
# Edit these values directly to change scenario utilizations.
# ============================================================
SCENARIO_CONFIG: Dict[str, Dict[str, Any]] = {
    "trng_p10": {
        "cpu_utilization": 0.9,
        "gpu_utilization": 0.9,
        "ssd_state": "idle",
        "ram_state": "active",  # requested default
    },
    "trng_median": {
        "cpu_utilization": 1.1,
        "gpu_utilization": 1.1,
        "ssd_state": "idle",
        "ram_state": "active",  # requested default
    },
    "trng_p90": {
        "cpu_utilization": 1.2,
        "gpu_utilization": 1.2,
        "ssd_state": "idle",
        "ram_state": "active",  # requested default
    },
    "inf_p10": {
        "cpu_utilization": 0.70,
        "gpu_utilization": 0.70,
        "ssd_state": "active",
        "ram_state": "active",
    },
    "inf_median": {
        "cpu_utilization": 0.90,
        "gpu_utilization": 0.90,
        "ssd_state": "active",
        "ram_state": "active",
    },
    "inf_p90": {
        "cpu_utilization": 1.10,
        "gpu_utilization": 1.10,
        "ssd_state": "active",
        "ram_state": "active",
    },
    "lo": {
        "cpu_utilization": 0.70,
        "gpu_utilization": 0.70,
        "ssd_state": "active",
        "ram_state": "active",
    },
    "mid": {
        "cpu_utilization": 0.90,
        "gpu_utilization": 0.90,
        "ssd_state": "active",
        "ram_state": "active",
    },
    "hi": {
        "cpu_utilization": 1.20,
        "gpu_utilization": 1.20,
        "ssd_state": "active",
        "ram_state": "active",
    },
    "custom": {
        "cpu_utilization": 1.20,
        "gpu_utilization": 1.20,
        "ssd_state": "active",
        "ram_state": "active",
    },
}

RAM_IDLE_W_PER_GB = 0.0 #0.19
RAM_ACTIVE_W_PER_GB = 0.0 #0.54
OTHER_COMPONENTS_FACTOR = 0.0 #0.20

PUE_FACTOR = 1.15 

# ============================================================
# IO helpers
# ============================================================
def load_csv(path: Optional[str | Path] = None) -> pd.DataFrame:
    path = Path(path) if path is not None else INPUT_FILE
    if not path.exists() and path == INPUT_FILE and FALLBACK_INPUT_FILE.exists():
        path = FALLBACK_INPUT_FILE
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path)


def output_path(filename: str) -> Path:
    return INSTANCES_DIR / filename


# ============================================================
# Generic helpers
# ============================================================
def find_column(df: pd.DataFrame, candidates, required: bool = True) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise ValueError(f"Could not find any of these columns: {candidates}")
    return None


_NUMERIC_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)")


def parse_numeric(val) -> Any:
    if pd.isna(val):
        return pd.NA
    s = str(val).strip()
    if s == "":
        return pd.NA
    s = s.replace(",", "")
    m = _NUMERIC_RE.search(s)
    if m:
        return float(m.group(1))
    return pd.NA



def parse_price(val) -> Any:
    if pd.isna(val):
        return pd.NA
    s = str(val).strip()
    if s == "":
        return pd.NA
    s = s.replace(",", "").replace("$", "")
    s = s.lower().replace("hourly", "").strip()
    m = _NUMERIC_RE.search(s)
    if m:
        return float(m.group(1))
    return pd.NA



def safe_float(val, default: float = 0.0) -> float:
    try:
        if pd.isna(val):
            return default
        return float(val)
    except Exception:
        return default


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
    "A10G": 150,
    "K80": 300,
    "M60": 300,
    "V100": 300,
    "A100": 400,
    "H100": 700,
    "H200": 700,
    "B200": 1000,
    "B300": 1100,
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
    "K520": "official_vendor_spec",
    "GB202": "official_vendor_spec",
    "V520": "official_vendor_spec",
    "QA100": "official_vendor_spec",
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
    s = s.replace("NVIDIA", "").replace("Tesla", "").replace("Tensor Core", "").strip()
    return s if s else pd.NA


# ============================================================
# CPU normalization and lookup
# ============================================================
def normalize_cpu_model(val):
    if pd.isna(val):
        return pd.NA
    s = re.sub(r"\s+", " ", str(val).strip())
    if s == "":
        return pd.NA
    keep = [
        "NVIDIA Grace",
        "AWS Graviton2 Processor",
        "AMD EPYC 7R13 Processor",
        "AMD EPYC 7R32",
        "Intel Xeon Platinum 8275CL (Cascade Lake)",
        "Intel Xeon Platinum 8275L",
        "Intel Xeon Platinum 8259 (Cascade Lake)",
        "Intel Xeon Platinum 8175 (Skylake)",
        "Intel Xeon E5-2686 v4 (Broadwell)",
        "Intel Xeon E5-2670 (Sandy Bridge)",
        "Intel Xeon Scalable (Icelake)",
        "Intel Xeon Scalable (Sapphire Rapids)",
        "Intel Xeon Scalable (Emerald Rapids)",
        "Intel Xeon Family",
    ]
    for label in keep:
        if label in s:
            return label
    if "NVIDIA Grace" in s:
        return "NVIDIA Grace"
    return s


CPU_LOOKUP = {
    "NVIDIA Grace": {
        "cpu_tdp_w_package": 500.0,
        "threads_per_package": 144.0,
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
        "source_name": "Closest EPYC Milan custom-SKU benchmark/reference listings, serverorbit",
        "source_note": "AWS custom Milan part; package TDP inferred from common benchmark/reference sources.",
    },
    "AMD EPYC 7R32": {
        "cpu_tdp_w_package": 280.0,
        "threads_per_package": 96.0,
        "source_type": "benchmark_proxy",
        "source_name": "Closest EPYC Rome custom-SKU benchmark/reference listings, phoronix",
        "source_note": "AWS custom Rome part; package TDP inferred from common benchmark/reference sources.",
    },
    "Intel Xeon Platinum 8259 (Cascade Lake)": {
        "cpu_tdp_w_package": 210.0,
        "threads_per_package": 48.0,
        "source_type": "benchmark_proxy",
        "source_name": "Closest Cascade Lake benchmark/reference listings, CPU World",
        "source_note": "Custom AWS-adjacent SKU; inferred from common reference sites.",
    },
    "Intel Xeon Platinum 8275CL (Cascade Lake)": {
        "cpu_tdp_w_package": 240.0,
        "threads_per_package": 48.0,
        "source_type": "benchmark_proxy",
        "source_name": "Closest Cascade Lake benchmark/reference listings, CPU World",
        "source_note": "Custom AWS SKU; inferred from common reference sites.",
    },
    "Intel Xeon Platinum 8275L": {
        "cpu_tdp_w_package": 165.0,
        "threads_per_package": 48.0,
        "source_type": "benchmark_proxy",
        "source_name": "Closest Cascade Lake benchmark/reference listings",
        "source_note": "No clean public Intel datasheet for exact AWS context; Intel for 8276L proxy used.",
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
        "source_name": "Broadwell Xeon benchmark reference listings",
        "source_note": "AWS-associated custom part; direct vendor-like proxy value, CPU World.",
    },
    "Intel Xeon E5-2670 (Sandy Bridge)": {
        "cpu_tdp_w_package": 115.0,
        "threads_per_package": 16.0,
        "source_type": "vendor spec",
        "source_name": "Common Sandy Bridge Xeon benchmark/reference listings",
        "source_note": "Direct vendor-like proxy value.",
    },
    "Intel Xeon Scalable (Icelake)": {
        "cpu_tdp_w_package": 270.0,
        "threads_per_package": 64.0,
        "source_type": "family_proxy",
        "source_name": "Representative Ice Lake Xeon benchmark/reference listings",
        "source_note": "Generic family label in source data; representative family proxy used. Xeon Platinum 8368 3.4GHz",
    },
    "Intel Xeon Scalable (Sapphire Rapids)": {
        "cpu_tdp_w_package": 205.0,
        "threads_per_package": 64.0,
        "source_type": "family_proxy",
        "source_name": "Representative Sapphire Rapids Xeon benchmark/reference listings",
        "source_note": "Generic family label in source data; representative family proxy used. 6438N/M/Y+ proxy used",
    },
    "Intel Xeon Scalable (Emerald Rapids)": {
        "cpu_tdp_w_package": 350.0,
        "threads_per_package": 96.0,
        "source_type": "family_proxy",
        "source_name": "Representative Emerald Rapids Xeon benchmark/reference listings",
        "source_note": "Generic family label in source data; representative family proxy used. 8568Y+ proxy used",
    },
    "Intel Xeon Family": {
        "cpu_tdp_w_package": 210.0,
        "threads_per_package": 48.0,
        "source_type": "family_proxy",
        "source_name": "Generic Xeon family proxy from common reference sites",
        "source_note": "Very generic label; conservative proxy used.",
    },
}


def cpu_lookup_record(cpu_model) -> Dict[str, Any]:
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
    tdp = float(rec["cpu_tdp_w_package"])
    threads = float(rec["threads_per_package"])
    per_vcpu = tdp / threads if threads else pd.NA
    return {
        "cpu_tdp_w_package": tdp,
        "threads_per_package": threads,
        "cpu_tdp_w_per_vcpu": per_vcpu,
        "cpu_source_type": rec["source_type"],
        "cpu_source_name": rec["source_name"],
        "cpu_source_note": rec["source_note"],
        "cpu_mapping_status": "resolved",
    }


# ============================================================
# RAM helpers
# ============================================================
def parse_memory_gib(val) -> Any:
    if pd.isna(val):
        return pd.NA
    s = str(val).strip()
    if s == "":
        return pd.NA
    s = s.replace(",", "")
    m = _NUMERIC_RE.search(s)
    if not m:
        return pd.NA
    return float(m.group(1))


# ============================================================
# SSD helpers
# These are enterprise NVMe proxy drives chosen to represent the
# instance-store / temp-local SSD class when exact host SSD model
# is not public in the provider CSVs.
# ============================================================
SSD_PROXY_CATALOG = {
    "no_local_ssd": {
        "proxy_model": "No host-local SSD attributed",
        "proxy_capacity_gb": 0.0,
        "idle_w": 0.0,
        "active_w": 0.0,
        "idle_w_per_tb": 0.0,
        "active_w_per_tb": 0.0,
        "source_type": "not_applicable",
        "source_note": "EBS-only or no local SSD in instance dataset.",
    },
    "small": {
        "proxy_model": "Benchmark for small",
        "proxy_capacity_gb": 1.0,
        "idle_w": 0.0,
        "active_w": 0.0,#5.0,
        "idle_w_per_tb": 0.0,
        "active_w_per_tb": 0.0,#5.0,
        "source_type": "proxy_reference",
        "source_note": "Proxy SSD class with watts expressed on a per-TB basis.",
    },
    "medium": {
        "proxy_model": "Benchmark for medium",
        "proxy_capacity_gb": 2.5,
        "idle_w": 0.0,
        "active_w": 0.0,#2.6,
        "idle_w_per_tb": 0.0,
        "active_w_per_tb": 0.0, #2.6,
        "source_type": "proxy_reference",
        "source_note": "Proxy SSD class with watts expressed on a per-TB basis.",
    },
    "large": {
        "proxy_model": "Benchmark for large",
        "proxy_capacity_gb": 5.0,
        "idle_w": 0.0,
        "active_w": 0.0, #1.5,
        "idle_w_per_tb": 0.0,
        "active_w_per_tb": 0.0, #1.5,
        "source_type": "proxy_reference",
        "source_note": "Proxy SSD class with watts expressed on a per-TB basis.",
    },
    "xlarge": {
        "proxy_model": "Benchmark for xlarge",
        "proxy_capacity_gb": 10.0,
        "idle_w": 0.0,
        "active_w": 0.0, #0.9,
        "idle_w_per_tb": 0.0,
        "active_w_per_tb": 0.0, #0.9,
        "source_type": "proxy_reference",
        "source_note": "Proxy SSD class with watts expressed on a per-TB basis.",
    },
}


def _extract_drive_count(storage_text: str) -> Optional[int]:
    patterns = [
        r"\((\d+)\s*[×x]\s*[0-9,.]+\s*GB",
        r"(\d+)\s*[×x]\s*[0-9,.]+\s*GB",
    ]
    for pat in patterns:
        m = re.search(pat, storage_text)
        if m:
            return int(m.group(1))
    return None



def _extract_total_storage_gb(storage_text: str) -> Optional[float]:
    if not storage_text:
        return None
    s = storage_text.replace(",", "")
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*GB", s)
    if not m:
        return None
    return float(m.group(1))



def parse_instance_storage(storage_val) -> Dict[str, Any]:
    if pd.isna(storage_val):
        return {
            "storage_is_local_ssd": False,
            "storage_total_gb": 0.0,
            "storage_drive_count_explicit": pd.NA,
            "storage_text_clean": pd.NA,
        }

    s = str(storage_val).strip()
    if s == "":
        return {
            "storage_is_local_ssd": False,
            "storage_total_gb": 0.0,
            "storage_drive_count_explicit": pd.NA,
            "storage_text_clean": pd.NA,
        }

    s_lower = s.lower()
    is_local = ("nvme ssd" in s_lower) and ("ebs only" not in s_lower)
    total_gb = _extract_total_storage_gb(s) or 0.0
    drive_count = _extract_drive_count(s)

    return {
        "storage_is_local_ssd": is_local,
        "storage_total_gb": float(total_gb),
        "storage_drive_count_explicit": drive_count if drive_count is not None else pd.NA,
        "storage_text_clean": s,
    }



def select_ssd_proxy(storage_total_gb: float, drive_count_explicit: Any, is_local_ssd: bool) -> Dict[str, Any]:
    if not is_local_ssd or storage_total_gb <= 0:
        rec = SSD_PROXY_CATALOG["no_local_ssd"].copy()
        rec.update({"estimated_drive_count": 0, "estimated_per_drive_gb": 0.0})
        return rec

    if pd.notna(drive_count_explicit) and safe_float(drive_count_explicit, 0) > 0:
        drive_count = int(safe_float(drive_count_explicit))
    else:
        drive_count = None

    if drive_count:
        per_drive_gb = storage_total_gb / drive_count
    else:
        per_drive_gb = storage_total_gb

    threshold_keys = ["small", "medium", "large", "xlarge"]
    threshold_records = sorted(
        (
            (float(SSD_PROXY_CATALOG[key]["proxy_capacity_gb"]) * 1000.0, key)
            for key in threshold_keys
        ),
        key=lambda x: x[0],
    )

    selected_key = threshold_records[-1][1]
    for threshold_gb, key in threshold_records:
        if per_drive_gb <= threshold_gb:
            selected_key = key
            break

    proxy = SSD_PROXY_CATALOG[selected_key].copy()

    if not drive_count:
        proxy_capacity_gb = float(proxy["proxy_capacity_gb"]) * 1000.0
        if proxy_capacity_gb > 0:
            drive_count = max(1, int(math.ceil(storage_total_gb / proxy_capacity_gb)))
        else:
            drive_count = 0
        per_drive_gb = storage_total_gb / drive_count if drive_count else 0.0

    proxy.update(
        {
            "estimated_drive_count": drive_count,
            "estimated_per_drive_gb": float(per_drive_gb),
        }
    )
    return proxy


# ============================================================
# Scenario calculations
# ============================================================
def scenario_ram_w(memory_gib: float, ram_state: str) -> float:
    rate = RAM_ACTIVE_W_PER_GB if str(ram_state).lower() == "active" else RAM_IDLE_W_PER_GB
    return safe_float(memory_gib) * rate



def scenario_ssd_w(idle_w_total: float, active_w_total: float, ssd_state: str) -> float:
    return active_w_total if str(ssd_state).lower() == "active" else idle_w_total
