import requests
import pandas as pd

BASE_URL = "https://api.boavizta.org/v1"

def get_all_instances(provider="aws"):
    url = f"{BASE_URL}/cloud/instance/all_instances"
    params = {"provider": provider}

    response = requests.get(url, params=params)
    response.raise_for_status()

    return response.json()

# Fetch AWS instances
instances = get_all_instances("aws")

# Convert to DataFrame
df = pd.DataFrame(instances)

print(df.head())
print(f"\nTotal instances: {len(df)}")

# Save to CSV
df.to_csv("boavizta_all_instances_aws.csv", index=False)

import requests
import pandas as pd
import os

BASE_URL = "https://api.boavizta.org/v1"

OUTPUT_DIR = r"C:\Users\FarhadBillimoria\OneDrive - Aurora Energy Research\Documents\Personal\Endog_demand_AI\pyai\data\original\boavizta"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "boavizta_all_instances.csv")

def get_all_instances(provider: str):
    url = f"{BASE_URL}/cloud/instance/all_instances"
    params = {"provider": provider}

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()

# Ensure directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

providers = ["aws", "gcp", "azure"]
all_rows = []

for provider in providers:
    try:
        data = get_all_instances(provider)

        for inst in data:
            all_rows.append({
                "provider": provider,
                "instance_type": inst
            })

        print(f"{provider}: {len(data)} instances")

    except Exception as e:
        print(f"Error with {provider}: {e}")

# Create dataframe
df = pd.DataFrame(all_rows)

print(df.head())
print(f"\nTotal instances: {len(df)}")

# Save to your directory
df.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved to: {OUTPUT_FILE}")

import os
import json
import time
import requests
import pandas as pd
from typing import Any, Dict, List, Optional

BASE_URL = "https://api.boavizta.org/v1"
PROVIDER = "aws"

OUTPUT_DIR = r"C:\Users\FarhadBillimoria\OneDrive - Aurora Energy Research\Documents\Personal\Endog_demand_AI\pyai\data\original\boavizta"
RAW_JSON_PATH = os.path.join(OUTPUT_DIR, "boavizta_component_power_raw_aws.json")
FLAT_CSV_PATH = os.path.join(OUTPUT_DIR, "boavizta_component_power_flat_aws.csv")
ERROR_CSV_PATH = os.path.join(OUTPUT_DIR, "boavizta_component_power_errors_aws.csv")

REQUEST_TIMEOUT = 120
SLEEP_SECONDS = 0.15


def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def safe_get(dct: Any, path: List[str], default=None):
    cur = dct
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def get_all_instances(session: requests.Session, provider: str = "aws") -> List[str]:
    """
    Returns a list of AWS instance type strings from Boavizta, e.g.:
    ["a1.medium", "c5.large", ...]
    """
    url = f"{BASE_URL}/cloud/instance/all_instances"
    resp = session.get(
        url,
        params={"provider": provider},
        timeout=REQUEST_TIMEOUT,
        headers={"accept": "application/json"},
    )
    resp.raise_for_status()

    data = resp.json()
    if not isinstance(data, list):
        raise ValueError(f"Unexpected response type from all_instances: {type(data)}")

    # Expect list of strings
    cleaned = []
    for item in data:
        if isinstance(item, str):
            cleaned.append(item)
        else:
            cleaned.append(str(item))

    return cleaned


def get_instance_details(
    session: requests.Session,
    instance_type: str,
    provider: str = "aws",
) -> Dict[str, Any]:
    """
    Query detailed Boavizta cloud instance data with verbose output.
    Tries both `provider` and `cloud_provider` to be robust.
    """
    url = f"{BASE_URL}/cloud/instance"

    param_candidates = [
        {
            "provider": provider,
            "instance_type": instance_type,
            "verbose": "true",
            "duration": 8760,
            "criteria": ["gwp", "adp"],
        },
        {
            "cloud_provider": provider,
            "instance_type": instance_type,
            "verbose": "true",
            "duration": 8760,
            "criteria": ["gwp", "adp"],
        },
    ]

    last_exc: Optional[Exception] = None

    for params in param_candidates:
        try:
            resp = session.get(
                url,
                params=params,
                timeout=REQUEST_TIMEOUT,
                headers={"accept": "application/json"},
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_exc = exc

    raise RuntimeError(f"Failed to fetch details for {instance_type}: {last_exc}")


def find_component_keys(verbose: Dict[str, Any], prefix: str) -> List[str]:
    """
    Find keys like CPU-1, CPU-2, RAM-1, GPU-1, etc.
    """
    target = prefix.upper() + "-"
    return [
        key
        for key in verbose.keys()
        if isinstance(key, str) and key.upper().startswith(target)
    ]


def aggregate_component(verbose: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """
    Aggregate component data across blocks like CPU-1, CPU-2, etc.
    Keeps:
      - metadata from the first block
      - sums of avg_power and impacts across all blocks
      - workload curve from the first block, if present
    """
    keys = find_component_keys(verbose, prefix)
    prefix_l = prefix.lower()

    out: Dict[str, Any] = {
        f"{prefix_l}_component_count": len(keys),
        f"{prefix_l}_component_keys": "|".join(keys) if keys else None,
    }

    if not keys:
        return out

    first = verbose[keys[0]]

    representative_fields = [
        "name",
        "manufacturer",
        "family",
        "model_range",
        "tdp",
        "units",
        "capacity",
        "memory_capacity",
        "vram_capacity",
        "core_units",
        "threads",
        "die_size",
        "avg_power",
        "time_workload",
        "use_time_ratio",
        "hours_life_time",
    ]

    for field in representative_fields:
        value = safe_get(first, [field, "value"])
        unit = safe_get(first, [field, "unit"])
        status = safe_get(first, [field, "status"])
        source = safe_get(first, [field, "source"])

        if value is not None:
            out[f"{prefix_l}_{field}"] = value
        if unit is not None:
            out[f"{prefix_l}_{field}_unit"] = unit
        if status is not None:
            out[f"{prefix_l}_{field}_status"] = status
        if source is not None:
            out[f"{prefix_l}_{field}_source"] = source

    # Extract workload curve if present
    workloads = safe_get(first, ["workloads", "value"], [])
    if isinstance(workloads, list):
        for item in workloads:
            if isinstance(item, dict):
                load_pct = item.get("load_percentage")
                power_watt = item.get("power_watt")
                if isinstance(load_pct, (int, float)) and isinstance(power_watt, (int, float)):
                    out[f"{prefix_l}_power_watt_at_{int(load_pct)}pct"] = float(power_watt)

    # Sum selected numeric fields across all component blocks
    sum_paths = {
        "avg_power_sum": ["avg_power", "value"],
        "gwp_embedded_sum": ["impacts", "gwp", "embedded", "value"],
        "gwp_use_sum": ["impacts", "gwp", "use", "value"],
        "adp_embedded_sum": ["impacts", "adp", "embedded", "value"],
        "adp_use_sum": ["impacts", "adp", "use", "value"],
    }

    for out_name, path in sum_paths.items():
        total = 0.0
        found_any = False

        for key in keys:
            val = safe_get(verbose[key], path)
            if isinstance(val, (int, float)):
                total += float(val)
                found_any = True

        out[f"{prefix_l}_{out_name}"] = total if found_any else None

    return out


def flatten_instance_response(
    instance_type: str,
    provider: str,
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Flatten a Boavizta instance response into a single row for CSV output.
    """
    verbose = data.get("verbose", {})

    row: Dict[str, Any] = {
        "provider": provider,
        "instance_type": instance_type,
        "instance_units": safe_get(verbose, ["units", "value"]),
        "instance_vcpu": safe_get(verbose, ["vcpu", "value"]),
        "instance_memory_gb": safe_get(verbose, ["memory", "value"]),
        "instance_memory_unit": safe_get(verbose, ["memory", "unit"]),
        "instance_avg_power_w": safe_get(verbose, ["avg_power", "value"]),
        "instance_avg_power_unit": safe_get(verbose, ["avg_power", "unit"]),
        "instance_duration_hours": safe_get(verbose, ["duration", "value"]),
        "instance_other_consumption_ratio": safe_get(verbose, ["other_consumption_ratio", "value"]),
        "instance_usage_location": safe_get(verbose, ["usage_location", "value"]),
        "instance_use_time_ratio": safe_get(verbose, ["use_time_ratio", "value"]),
        "instance_hours_life_time": safe_get(verbose, ["hours_life_time", "value"]),
        "instance_gwp_embedded": safe_get(data, ["impacts", "gwp", "embedded", "value"]),
        "instance_gwp_use": safe_get(data, ["impacts", "gwp", "use", "value"]),
        "instance_adp_embedded": safe_get(data, ["impacts", "adp", "embedded", "value"]),
        "instance_adp_use": safe_get(data, ["impacts", "adp", "use", "value"]),
    }

    row.update(aggregate_component(verbose, "CPU"))
    row.update(aggregate_component(verbose, "RAM"))
    row.update(aggregate_component(verbose, "GPU"))

    cpu_avg = row.get("cpu_avg_power_sum") or 0.0
    ram_avg = row.get("ram_avg_power_sum") or 0.0
    gpu_avg = row.get("gpu_avg_power_sum") or 0.0
    row["component_avg_power_sum_w"] = cpu_avg + ram_avg + gpu_avg

    return row


def main() -> None:
    ensure_output_dir()

    session = requests.Session()

    print("Fetching available AWS instances...")
    instances = get_all_instances(session, PROVIDER)
    print(f"Found {len(instances)} AWS instances")

    raw_results: Dict[str, Any] = {}
    flat_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []

    for i, instance_type in enumerate(instances, start=1):
        try:
            print(f"[{i}/{len(instances)}] {instance_type}")
            data = get_instance_details(session, instance_type, PROVIDER)
            raw_results[instance_type] = data
            flat_rows.append(flatten_instance_response(instance_type, PROVIDER, data))
            time.sleep(SLEEP_SECONDS)
        except Exception as exc:
            errors.append(
                {
                    "provider": PROVIDER,
                    "instance_type": instance_type,
                    "error": str(exc),
                }
            )
            print(f"  ERROR: {exc}")

    with open(RAW_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(raw_results, f, indent=2)

    pd.DataFrame(flat_rows).to_csv(FLAT_CSV_PATH, index=False)
    pd.DataFrame(errors).to_csv(ERROR_CSV_PATH, index=False)

    print("\nDone.")
    print(f"Raw JSON saved to: {RAW_JSON_PATH}")
    print(f"Flat CSV saved to: {FLAT_CSV_PATH}")
    print(f"Errors CSV saved to: {ERROR_CSV_PATH}")
    print(f"Successful instances: {len(flat_rows)}")
    print(f"Errored instances: {len(errors)}")


if __name__ == "__main__":
    main()