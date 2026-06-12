# Reusable graphing and result helpers for EC2 GPU power/VOLL analysis.
# Run from the project root or scripts directory after activating .ai_dem.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import importlib.util
import re

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

try:
    from scipy.stats import gaussian_kde
except ImportError:  # Optional dependency; only needed for KDE plots.
    gaussian_kde = None


# ============================================================
# Paths and configuration
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_INSTANCES_DIR = DATA_DIR / "processed" / "instances"
ORIGINAL_PRICES_DIR = DATA_DIR / "original" / "prices"
DEFAULT_OUTPUT_DIR = PROCESSED_INSTANCES_DIR / "graphs"

DEFAULT_POWER_FILE = (
    PROCESSED_INSTANCES_DIR / "aws_gpu_cpu_ram_ssd_power_and_value_wide_expanded.csv"
)

FALLBACK_SCENARIOS: List[str] = [
    "trng_p10",
    "trng_median",
    "trng_p90",
    "inf_p10",
    "inf_median",
    "inf_p90",
    "lo",
    "mid",
    "hi",
    "custom",
]


def load_scenario_config() -> Dict[str, Dict[str, Any]]:
    """Load scenario definitions from ec2_power_common without requiring package setup."""
    script_dir = globals().get("SCRIPT_DIR", Path.cwd() / "scripts")
    config_path = Path(script_dir) / "ec2_power_common.py"
    if not config_path.exists():
        return {scenario: {} for scenario in FALLBACK_SCENARIOS}

    spec = importlib.util.spec_from_file_location("ec2_power_common", config_path)
    if spec is None or spec.loader is None:
        return {scenario: {} for scenario in FALLBACK_SCENARIOS}

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scenario_config = getattr(module, "SCENARIO_CONFIG", None)
    if not isinstance(scenario_config, dict):
        return {scenario: {} for scenario in FALLBACK_SCENARIOS}
    return dict(scenario_config)


SCENARIO_CONFIG: Dict[str, Dict[str, Any]] = load_scenario_config()
DEFAULT_SCENARIOS: List[str] = list(SCENARIO_CONFIG)

SCENARIO_ALIASES: Dict[str, str] = {
    "trn_p10": "trng_p10",
    "trn_median": "trng_median",
    "trn_p90": "trng_p90",
    "training_p10": "trng_p10",
    "training_median": "trng_median",
    "training_p90": "trng_p90",
    "inference_p10": "inf_p10",
    "inference_median": "inf_median",
    "inference_p90": "inf_p90",
}

VOLL_BASIS_PREFIXES: Dict[str, str] = {
    "ondemand": "voll_usd_per_mwh",
    "on_demand": "voll_usd_per_mwh",
    "on-demand": "voll_usd_per_mwh",
    "demand": "voll_usd_per_mwh",
    "spot": "voll_spot_usd_per_mwh",
    "reserved": "voll_reserved_usd_per_mwh",
    "reserve": "voll_reserved_usd_per_mwh",
    "1yr": "voll_reserved_usd_per_mwh",
    "1-year": "voll_reserved_usd_per_mwh",
    "one_year": "voll_reserved_usd_per_mwh",
}

VOLL_BASIS_LABELS: Dict[str, str] = {
    "voll_usd_per_mwh": "On-demand",
    "voll_spot_usd_per_mwh": "Spot",
    "voll_reserved_usd_per_mwh": "1-year reserved",
}

HOURS_PER_MONTH = 730.0
H_CUMULATIVE = 0.0
U_MTH = 0.1
SLA_CLOUD_PLATFORM = "aws_ec2_instance"
SLA_CLOUD_PLATFORM_LABELS: Dict[str, str] = {
    "aws_ec2_instance": "AWS EC2",
    "gcp": "GCP",
    "azure_virtual_machines": "Azure",
}

SERVICE_CREDIT_BANDS: Dict[str, Tuple[Tuple[float, float], ...]] = {
    "azure_virtual_machines": (
        (0.8, 0.10),
        (7.3, 0.25),
        (36.5, 1.00),
    ),
    "aws_ec2_instance": (
        (3.7, 0.10),
        (7.3, 0.30),
        (36.5, 1.00),
    ),
    "gcp": (
        (0.8, 0.10),
        (7.3, 0.25),
        (36.5, 1.00),
    ),
}

SPOT_INTERRUPT_ORDER: List[str] = ["<5%", "5-10%", "10-15%", "15-20%", ">20%"]

HISTORICAL_ONDEMAND_PRICE_TRAJECTORIES: Dict[Tuple[str, str], Dict[str, Any]] = {
    ("aws", "p4d24xlarge"): {
        "current_price": 21.958,
        "price_points": [
            ("1900-01-01", 32.773),
            ("2025-06-17", 24.153),
            ("2025-07-19", 24.153),
            ("2025-10-21", 21.958),
        ],
    },
}


@dataclass(frozen=True)
class ProviderConfig:
    instance_col_candidates: Sequence[str]
    price_datetime_col: Optional[str] = None
    price_value_col: Optional[str] = None
    region_col: Optional[str] = None
    availability_zone_col: Optional[str] = None


@dataclass(frozen=True)
class PriceFileRecord:
    path: Path
    provider: str
    family: str
    instance_token: str
    normalized_instance: str
    region: str


PROVIDER_CONFIG: Dict[str, ProviderConfig] = {
    "aws": ProviderConfig(
        instance_col_candidates=(
            "instance_type",
            "API Name",
            "Instance type",
            "Instance Type",
            "Name",
        ),
        price_datetime_col="Modified Date Time",
        price_value_col="Instance Price",
        region_col="Region",
        availability_zone_col="Availability Zone",
    ),
    "gcp": ProviderConfig(
        instance_col_candidates=("instance_type", "Machine type", "Machine Type", "Name"),
    ),
    "azure": ProviderConfig(
        instance_col_candidates=("instance_type", "VM Size", "VM size", "Name"),
    ),
}


@dataclass(frozen=True)
class PlotStyle:
    figsize: tuple[float, float] = (12.5, 6.8)
    grid_alpha: float = 0.30
    envelope_alpha: float = 0.22
    range_bar_alpha: float = 0.35
    range_bar_width: float = 0.6
    dpi: int = 300
    reference_lines: Mapping[str, float] = field(
        default_factory=lambda: {
            "PJM Offer Cap": 2000,
            "PJM System Price Cap": 3700,
        }
    )

GPU_VINTAGE_GROUPS = {
    "Legacy / Older Vintage": ["K520", "M60", "V520", "K80"],
    "Transitional / Early AI": ["T4", "V100", "Inf1"],
    "Mainstream Modern AI": [
        "A10G", "A100", "Inf2", "L40S", "H100",
        "T4G", "QA100", "GB202", "L4",
    ],
    "Emerging / Frontier AI": ["H200", "B200", "B300"],
}

GPU_VINTAGE_PALETTE = {
    "Legacy / Older Vintage": "#6f7f89",
    "Transitional / Early AI": "#3fa34d",
    "Mainstream Modern AI": "#ff7f0e",
    "Emerging / Frontier AI": "#7b4ab2",
}

GPU_VINTAGE_SHORT_LABELS = {
    "Legacy / Older Vintage": "LEGACY /\nOLDER VINTAGE",
    "Transitional / Early AI": "TRANSITIONAL /\nEARLY AI",
    "Mainstream Modern AI": "MAINSTREAM\nMODERN AI",
    "Emerging / Frontier AI": "EMERGING /\nFRONTIER AI",
}


def classify_gpu_vintage(gpu: str) -> str:
    gpu = str(gpu).strip()
    for group, members in GPU_VINTAGE_GROUPS.items():
        if gpu in members:
            return group
    return "Other / Unclassified"

# ============================================================
# Loading and validation
# ============================================================
def ensure_output_dir(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def find_first_existing_column(
    df: pd.DataFrame,
    candidates: Iterable[str],
    required: bool = True,
) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise ValueError(f"Could not find any of these columns: {list(candidates)}")
    return None


def load_csv(filepath: Path, label: str = "CSV") -> pd.DataFrame:
    if not filepath.exists():
        raise FileNotFoundError(f"{label} not found: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Loaded {label}: {filepath}")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    return df


def load_power_data(filepath: Path = DEFAULT_POWER_FILE) -> pd.DataFrame:
    return load_csv(filepath, label="power data")


def normalize_instance_name(instance_name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(instance_name).lower())


def provider_from_price_prefix(prefix: str) -> Optional[str]:
    prefix_map = {
        "EC2": "aws",
    }
    return prefix_map.get(prefix.upper())


def parse_price_file_path(filepath: Path) -> Optional[PriceFileRecord]:
    parts = filepath.stem.split("_")
    if len(parts) < 3:
        return None

    provider = provider_from_price_prefix(parts[0])
    if provider is None:
        return None

    region = parts[-1]
    instance_token = "_".join(parts[1:-1])
    if not instance_token or not re.match(r"^[a-z]{2}-[a-z-]+-\d$", region):
        return None

    return PriceFileRecord(
        path=filepath,
        provider=provider,
        family=filepath.parent.name,
        instance_token=instance_token,
        normalized_instance=normalize_instance_name(instance_token),
        region=region,
    )


def discover_price_files(
    prices_dir: Path = ORIGINAL_PRICES_DIR,
    provider: Optional[str] = "aws",
) -> pd.DataFrame:
    records = []
    for filepath in sorted(prices_dir.rglob("*.csv")):
        record = parse_price_file_path(filepath)
        if record is None:
            continue
        if provider is not None and record.provider != provider:
            continue
        records.append(
            {
                "provider": record.provider,
                "family": record.family,
                "instance_token": record.instance_token,
                "normalized_instance": record.normalized_instance,
                "region": record.region,
                "path": record.path,
            }
        )
    return pd.DataFrame(records)


def find_price_files_for_instance(
    instance_name: str,
    region: Optional[str] = None,
    provider: str = "aws",
    prices_dir: Path = ORIGINAL_PRICES_DIR,
) -> pd.DataFrame:
    catalog = discover_price_files(prices_dir=prices_dir, provider=provider)
    if catalog.empty:
        return catalog

    normalized = normalize_instance_name(instance_name)
    matches = catalog.loc[catalog["normalized_instance"] == normalized].copy()
    if region is not None:
        matches = matches.loc[matches["region"] == region].copy()
    return matches.sort_values(["family", "region", "path"]).reset_index(drop=True)


def resolve_price_file(
    instance_name: str,
    region: Optional[str] = None,
    provider: str = "aws",
    prices_dir: Path = ORIGINAL_PRICES_DIR,
    ) -> Path:
    matches = find_price_files_for_instance(
        instance_name=instance_name,
        region=region,
        provider=provider,
        prices_dir=prices_dir,
    )
    if matches.empty:
        catalog = discover_price_files(prices_dir=prices_dir, provider=provider)
        available = []
        if not catalog.empty:
            available = (
                catalog[["instance_token", "region"]]
                .drop_duplicates()
                .head(20)
                .to_dict("records")
            )
        region_msg = f" in region '{region}'" if region else ""
        raise FileNotFoundError(
            f"No price file found for {provider} instance '{instance_name}'{region_msg}. "
            f"Sample available price files: {available}"
        )

    if len(matches) > 1 and region is not None:
        nested_matches = matches.loc[matches["family"] != prices_dir.name].copy()
        if len(nested_matches) == 1:
            return Path(nested_matches.iloc[0]["path"])

    if len(matches) > 1:
        options = matches[["family", "instance_token", "region", "path"]].to_dict("records")
        raise ValueError(
            f"Multiple price files found for '{instance_name}'. "
            f"Pass region=... to select one. Options: {options}"
        )

    return Path(matches.iloc[0]["path"])


def load_price_data(
    filepath: Optional[Path] = None,
    instance_name: Optional[str] = None,
    region: Optional[str] = None,
    provider: str = "aws",
    prices_dir: Path = ORIGINAL_PRICES_DIR,
) -> pd.DataFrame:
    if filepath is None:
        if instance_name is None:
            raise ValueError("Provide either filepath or instance_name to load price data.")
        filepath = resolve_price_file(
            instance_name=instance_name,
            region=region,
            provider=provider,
            prices_dir=prices_dir,
        )
    return load_csv(filepath, label="price data")


def get_provider_config(provider: str) -> ProviderConfig:
    try:
        return PROVIDER_CONFIG[provider]
    except KeyError as exc:
        raise ValueError(f"Unsupported provider: {provider}") from exc


def get_instance_row(
    df_power: pd.DataFrame,
    instance_name: str,
    provider: str = "aws",
    ) -> pd.Series:
    cfg = get_provider_config(provider)
    instance_col = find_first_existing_column(df_power, cfg.instance_col_candidates)

    matches = df_power.loc[
        df_power[instance_col].astype(str).str.strip() == str(instance_name).strip()
    ]
    if matches.empty:
        available = df_power[instance_col].dropna().astype(str).head(20).tolist()
        raise ValueError(
            f"No matching instance '{instance_name}' found. "
            f"Sample available values: {available}"
        )
    if len(matches) > 1:
        print(f"Warning: multiple rows found for {instance_name}. Using the first row only.")
    return matches.iloc[0]


def filter_instances(
    df_power: pd.DataFrame,
    exclude_instances: Optional[Sequence[str]] = None,
    provider: str = "aws",
    ) -> pd.DataFrame:
    if not exclude_instances:
        return df_power.copy()

    cfg = get_provider_config(provider)
    instance_col = find_first_existing_column(df_power, cfg.instance_col_candidates)
    excluded = {str(item).strip() for item in exclude_instances}
    return df_power.loc[
        ~df_power[instance_col].astype(str).str.strip().isin(excluded)
    ].copy()


# ============================================================
# VOLL calculations and summaries
# ============================================================
def ondemand_voll_column(scenario: str) -> str:
    return f"voll_usd_per_mwh_{scenario}"


def spot_voll_column(scenario: str) -> str:
    return f"voll_spot_usd_per_mwh_{scenario}"


def total_power_column(scenario: str) -> str:
    return f"total_power_w_{scenario}"


def normalize_scenario(scenario: str) -> str:
    scenario_key = str(scenario).strip().lower().replace("-", "_")
    return SCENARIO_ALIASES.get(scenario_key, scenario_key)


def resolve_voll_value_prefix(
    voll_basis: str = "ondemand",
    value_prefix: Optional[str] = None,
    ) -> str:
    if value_prefix is not None:
        return value_prefix

    basis_key = str(voll_basis).strip().lower().replace(" ", "_")
    if basis_key in VOLL_BASIS_PREFIXES:
        return VOLL_BASIS_PREFIXES[basis_key]
    valid = sorted(VOLL_BASIS_PREFIXES)
    raise ValueError(f"Unknown VOLL basis '{voll_basis}'. Valid options include: {valid}")


def voll_value_column(
    scenario: str,
    voll_basis: str = "ondemand",
    value_prefix: Optional[str] = None,
    ) -> str:
    scenario = normalize_scenario(scenario)
    prefix = resolve_voll_value_prefix(voll_basis=voll_basis, value_prefix=value_prefix)
    return f"{prefix}_{scenario}"


def sla_adjusted_ondemand_voll_column(scenario: str) -> str:
    scenario = normalize_scenario(scenario)
    return f"voll_ondemand_incl_sla_usd_per_mwh_{scenario}"


def relevant_service_credit_share(
    h_cumulative: float = H_CUMULATIVE,
    cloud_platform: str = SLA_CLOUD_PLATFORM,
    ) -> float:
    bands = SERVICE_CREDIT_BANDS.get(cloud_platform)
    if bands is None:
        valid = sorted(SERVICE_CREDIT_BANDS)
        raise ValueError(f"Unknown SLA cloud platform '{cloud_platform}'. Valid options: {valid}")

    credit_share = 0.0
    for downtime_threshold_hrs, service_credit in sorted(bands):
        if h_cumulative + 1 > downtime_threshold_hrs and h_cumulative + 1 - 1 < downtime_threshold_hrs: #h_cumulative + 1 > downtime_threshold_hrs:
            credit_share = service_credit
    return credit_share


def add_sla_adjusted_ondemand_voll(
    df: pd.DataFrame,
    scenarios: Sequence[str],
    h_cumulative: float = H_CUMULATIVE,
    u_mth: float = U_MTH,
    cloud_platform: str = SLA_CLOUD_PLATFORM,
    hours_per_month: float = HOURS_PER_MONTH,
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
    out = df.copy()
    service_credit_share = relevant_service_credit_share(
        h_cumulative=h_cumulative,
        cloud_platform=cloud_platform,
    )
    adjusted_cols: Dict[str, str] = {}
    # VOLL(incl. SLA) scales the pre-SLA VOLL by monthly hours and the relevant credit share.
    sla_multiplier = service_credit_share * hours_per_month * u_mth

    for scenario in scenarios:
        base_col = voll_value_column(scenario, voll_basis="ondemand")
        if base_col not in out.columns:
            continue
        adjusted_col = sla_adjusted_ondemand_voll_column(scenario)
        out[adjusted_col] = pd.to_numeric(out[base_col], errors="coerce") * sla_multiplier
        adjusted_cols[scenario] = adjusted_col
    return out, adjusted_cols


def voll_basis_panel_label(voll_basis: str) -> str:
    value_prefix = resolve_voll_value_prefix(voll_basis=voll_basis)
    return VOLL_BASIS_LABELS.get(value_prefix, str(voll_basis).title())


def available_scenarios(
    df_power: pd.DataFrame,
    scenarios: Optional[Sequence[str]] = None,
    ) -> List[str]:
    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS
    return [
        normalize_scenario(scenario)
        for scenario in scenarios
        if total_power_column(normalize_scenario(scenario)) in df_power.columns
    ]


def compute_voll_timeseries(
    df_price: pd.DataFrame,
    df_power: pd.DataFrame,
    instance_name: str,
    provider: str = "aws",
    scenarios: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Add scenario VOLL time-series columns to a provider price history."""
    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS
    cfg = get_provider_config(provider)
    if cfg.price_datetime_col is None or cfg.price_value_col is None:
        raise ValueError(f"Provider '{provider}' price-format config is not filled in yet.")
    if cfg.price_datetime_col not in df_price.columns:
        raise ValueError(f"Datetime column '{cfg.price_datetime_col}' not found in price data")
    if cfg.price_value_col not in df_price.columns:
        raise ValueError(f"Price column '{cfg.price_value_col}' not found in price data")

    power_row = get_instance_row(df_power, instance_name=instance_name, provider=provider)
    out = df_price.copy()
    out["timestamp"] = pd.to_datetime(out[cfg.price_datetime_col], errors="coerce")
    out[cfg.price_value_col] = pd.to_numeric(out[cfg.price_value_col], errors="coerce")
    out = out.dropna(subset=["timestamp", cfg.price_value_col]).copy()
    out["date"] = out["timestamp"].dt.floor("D")
    out["instance_name"] = instance_name
    out["provider"] = provider

    created_voll_cols = []
    created_reference_cols = []
    for scenario in scenarios:
        power_col = total_power_column(scenario)
        if power_col not in df_power.columns:
            print(f"Skipping scenario '{scenario}': missing {power_col}")
            continue

        power_w = pd.to_numeric(power_row[power_col], errors="coerce")
        if pd.isna(power_w) or power_w <= 0:
            print(f"Skipping scenario '{scenario}': invalid power value {power_w}")
            continue

        voll_col = f"VOLL_{scenario}"
        out[voll_col] = out[cfg.price_value_col] / (power_w / 1_000_000.0)
        created_voll_cols.append(voll_col)

        ondemand_col = ondemand_voll_column(scenario)
        if ondemand_col in df_power.columns:
            reference_col = f"voll_ondemand_{scenario}"
            out[reference_col] = pd.to_numeric(power_row[ondemand_col], errors="coerce")
            created_reference_cols.append(reference_col)

    print("Created VOLL columns:", created_voll_cols)
    print("Created on-demand reference columns:", created_reference_cols)
    return out


def daily_voll_summary(df_voll: pd.DataFrame, scenario: str) -> pd.DataFrame:
    voll_col = f"VOLL_{scenario}"
    if voll_col not in df_voll.columns:
        raise ValueError(f"Column not found: {voll_col}")
    if "date" not in df_voll.columns:
        raise ValueError("Column 'date' not found in VOLL time series")

    daily = (
        df_voll.groupby("date")[voll_col]
        .agg(voll_mean="mean", voll_min="min", voll_max="max", obs_count="count")
        .reset_index()
        .sort_values("date")
    )
    if daily.empty:
        raise ValueError("No daily data available after aggregation.")
    return daily


def scenario_values(
    df_power: pd.DataFrame,
    scenario: str,
    value_prefix: str = "voll_usd_per_mwh",
) -> pd.Series:
    col = voll_value_column(scenario, value_prefix=value_prefix)
    if col not in df_power.columns:
        raise ValueError(f"Column not found: {col}")
    return pd.to_numeric(df_power[col], errors="coerce")


def sort_by_scenario(
    df_power: pd.DataFrame,
    scenario: str = "mid",
    value_prefix: str = "voll_usd_per_mwh",
    ) -> pd.DataFrame:
    col = voll_value_column(scenario, value_prefix=value_prefix)
    if col not in df_power.columns:
        raise ValueError(f"Column not found: {col}")
    return df_power.sort_values(col).reset_index(drop=True)


def filter_low_mid_voll(
    df_power: pd.DataFrame,
    min_mid_voll: Optional[float] = 500,
    value_prefix: str = "voll_usd_per_mwh",
    ) -> pd.DataFrame:
    if min_mid_voll is None:
        return df_power.copy()

    mid_col = voll_value_column("mid", value_prefix=value_prefix)
    if mid_col not in df_power.columns:
        raise ValueError(f"Column not found: {mid_col}")

    mid_values = pd.to_numeric(df_power[mid_col], errors="coerce")
    filtered = df_power.loc[mid_values >= min_mid_voll].copy()
    removed = len(df_power) - len(filtered)
    if removed:
        print(f"Excluded {removed} rows where {mid_col} < {min_mid_voll}.")
    return filtered


# ============================================================
# Plot helpers
# ============================================================
def save_current_figure(
    fig: plt.Figure,
    filename: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    style: PlotStyle = PlotStyle(),
    ) -> Path:
    output_dir = ensure_output_dir(output_dir)
    output_path = output_dir / filename
    fig.savefig(output_path, dpi=style.dpi, bbox_inches="tight")
    print(f"Saved plot: {output_path}")
    return output_path


def write_csv(df: pd.DataFrame, filename: str, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    output_dir = ensure_output_dir(output_dir)
    output_path = output_dir / filename
    df.to_csv(output_path, index=False)
    print(f"Saved CSV: {output_path}")
    return output_path


def tidy_axis(ax: plt.Axes, style: PlotStyle = PlotStyle()) -> None:
    ax.grid(True, alpha=style.grid_alpha, linewidth=0.8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def add_reference_lines(ax: plt.Axes, style: PlotStyle = PlotStyle()) -> None:
    colors = ["tab:green", "tab:orange", "tab:red", "tab:purple"]
    for idx, (label, value) in enumerate(style.reference_lines.items()):
        ax.axhline(
            y=value,
            linestyle="--",
            linewidth=1.8,
            color=colors[idx % len(colors)],
            label=label,
        )


def historical_ondemand_voll_trajectory(
    df_voll: pd.DataFrame,
    scenario: str,
    instance_name: str,
    provider: str = "aws",
    ) -> Optional[pd.DataFrame]:
    trajectory = HISTORICAL_ONDEMAND_PRICE_TRAJECTORIES.get(
        (provider.lower(), normalize_instance_name(instance_name))
    )
    ondemand_col = f"voll_ondemand_{scenario}"
    if trajectory is None or ondemand_col not in df_voll.columns or "date" not in df_voll.columns:
        return None

    ondemand_value = pd.to_numeric(df_voll[ondemand_col], errors="coerce").dropna()
    if ondemand_value.empty:
        return None

    date_values = pd.to_datetime(df_voll["date"], errors="coerce").dropna()
    if date_values.empty:
        return None

    current_price = float(trajectory["current_price"])
    current_voll = float(ondemand_value.iloc[0])
    price_points = [
        (pd.Timestamp(date), float(price))
        for date, price in trajectory["price_points"]
    ]
    price_points = sorted(price_points, key=lambda item: item[0])

    start_date = date_values.min().floor("D")
    end_date = date_values.max().floor("D")
    points = [(start_date, price_points[0][1])]
    for point_date, price in price_points:
        if start_date <= point_date <= end_date:
            points.append((point_date, price))
    points.append((end_date, points[-1][1]))

    out = pd.DataFrame(points, columns=["date", "ondemand_price_usd_per_hour"])
    out = out.drop_duplicates(subset=["date"], keep="last")
    out["voll_ondemand_historical"] = (
        out["ondemand_price_usd_per_hour"] / current_price * current_voll
    )
    return out


def plot_ondemand_reference(
    ax: plt.Axes,
    df_voll: pd.DataFrame,
    scenario: str,
    instance_name: str,
    provider: str = "aws",
    linewidth: float = 2.0,
    ) -> Optional[pd.DataFrame]:
    trajectory = historical_ondemand_voll_trajectory(
        df_voll=df_voll,
        scenario=scenario,
        instance_name=instance_name,
        provider=provider,
    )
    if trajectory is not None:
        ax.plot(
            trajectory["date"],
            trajectory["voll_ondemand_historical"],
            color="black",
            linewidth=linewidth,
            linestyle="-",
            drawstyle="steps-post",
            label=f"On-Demand Implied VOLL ({scenario})",
        )
        return trajectory

    ondemand_col = f"voll_ondemand_{scenario}"
    if ondemand_col in df_voll.columns:
        ondemand_value = pd.to_numeric(df_voll[ondemand_col], errors="coerce").dropna()
        if not ondemand_value.empty:
            ax.axhline(
                y=ondemand_value.iloc[0],
                color="black",
                linewidth=linewidth,
                linestyle="-",
                label=f"On-Demand Implied VOLL ({scenario})",
            )
    return None


def add_historical_ondemand_to_daily(
    daily: pd.DataFrame,
    trajectory: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
    if trajectory is None:
        return daily

    out = daily.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    trajectory = trajectory.sort_values("date")
    out = pd.merge_asof(
        out.sort_values("date"),
        trajectory.sort_values("date"),
        on="date",
        direction="backward",
    )
    return out


def plot_voll_timeseries(
    df_voll: pd.DataFrame,
    scenario: str,
    instance_name: str,
    provider: str = "aws",
    location: Optional[str] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
    save_png: bool = True,
    save_csv: bool = True,
    show: bool = True,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    style: PlotStyle = PlotStyle(),
) -> pd.DataFrame:
    daily = daily_voll_summary(df_voll, scenario)

    fig, ax = plt.subplots(figsize=style.figsize)
    ax.fill_between(
        daily["date"],
        daily["voll_min"],
        daily["voll_max"],
        alpha=style.envelope_alpha,
        label="Spot VOLL - Min-max envelope",
    )
    ax.plot(
        daily["date"],
        daily["voll_mean"],
        linewidth=2.4,
        label=f"Spot VOLL - Ave ({scenario})",
    )

    ondemand_trajectory = plot_ondemand_reference(
        ax=ax,
        df_voll=df_voll,
        scenario=scenario,
        instance_name=instance_name,
        provider=provider,
        linewidth=2.2,
    )
    daily = add_historical_ondemand_to_daily(daily, ondemand_trajectory)

    #add_reference_lines(ax, style)
    location_text = f" - {location}" if location else ""
    ax.set_title(
        f"Implied VoCL for {instance_name} ({provider.upper()}{location_text})\nScenario: {scenario}",
        fontsize=14,
        pad=14,
    )
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("VoCL ($/MWh)", fontsize=12)
    if ymin is not None or ymax is not None:
        ax.set_ylim(bottom=ymin, top=ymax)

    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    tidy_axis(ax, style)
    ax.legend(frameon=False, fontsize=10, loc="best")
    fig.subplots_adjust(top=0.88, bottom=0.16, right=0.97, left=0.10)

    safe_instance = instance_name.replace(".", "_")
    if save_png:
        save_current_figure(
            fig,
            f"voll_timeseries_{provider}_{safe_instance}_{scenario}.png",
            output_dir=output_dir,
            style=style,
        )
    if save_csv:
        write_csv(
            daily,
            f"voll_daily_{provider}_{safe_instance}_{scenario}.csv",
            output_dir=output_dir,
        )
    if show:
        plt.show()
    else:
        plt.close(fig)
    return daily


def plot_voll_timeseries_locations(
    instance_name: str,
    locations: Sequence[str],
    scenario: str,
    provider: str = "aws",
    df_power: Optional[pd.DataFrame] = None,
    power_file: Path = DEFAULT_POWER_FILE,
    price_files: Optional[Mapping[str, Path]] = None,
    prices_dir: Path = ORIGINAL_PRICES_DIR,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
    save_png: bool = True,
    save_csv: bool = True,
    show: bool = True,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    style: PlotStyle = PlotStyle(),
) -> pd.DataFrame:
    if not locations:
        raise ValueError("Provide at least one location.")

    scenario = normalize_scenario(scenario)
    if df_power is None:
        df_power = load_power_data(power_file)

    cfg = get_provider_config(provider)
    price_value_col = cfg.price_value_col
    if price_value_col is None:
        raise ValueError(f"Provider '{provider}' price-format config is not filled in yet.")

    daily_frames = []
    location_voll_frames = []
    for location in locations:
        price_file = price_files.get(location) if price_files else None
        df_price = load_price_data(
            filepath=price_file,
            instance_name=instance_name,
            region=location,
            provider=provider,
            prices_dir=prices_dir,
        )
        df_voll = compute_voll_timeseries(
            df_price=df_price,
            df_power=df_power,
            instance_name=instance_name,
            provider=provider,
            scenarios=[scenario],
        )
        df_voll["location"] = location
        location_voll_frames.append(df_voll)

        daily = daily_voll_summary(df_voll, scenario)
        daily["location"] = location
        daily_frames.append(daily)

    if not daily_frames:
        raise ValueError("No location data available to plot.")

    combined_daily = pd.concat(daily_frames, ignore_index=True)
    combined_voll = pd.concat(location_voll_frames, ignore_index=True)

    fig, ax = plt.subplots(figsize=style.figsize)
    cmap = plt.colormaps.get_cmap("tab10").resampled(len(locations))

    for idx, location in enumerate(locations):
        daily = combined_daily.loc[combined_daily["location"] == location].copy()
        if daily.empty:
            continue
        color = cmap(idx)
        ax.fill_between(
            daily["date"],
            daily["voll_min"],
            daily["voll_max"],
            alpha=style.envelope_alpha * 0.65,
            color=color,
        )
        ax.plot(
            daily["date"],
            daily["voll_mean"],
            linewidth=2.2,
            color=color,
            label=f"{location} spot VOLL - Ave",
        )

    ondemand_trajectory = plot_ondemand_reference(
        ax=ax,
        df_voll=combined_voll,
        scenario=scenario,
        instance_name=instance_name,
        provider=provider,
        linewidth=2.0,
    )
    combined_daily = add_historical_ondemand_to_daily(combined_daily, ondemand_trajectory)

    ax.set_title(
        f"Implied VoCL for {instance_name} ({provider.upper()}) by location\nScenario: {scenario}",
        fontsize=14,
        pad=14,
    )
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("VoCL ($/MWh)", fontsize=12)
    if ymin is not None or ymax is not None:
        ax.set_ylim(bottom=ymin, top=ymax)

    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    tidy_axis(ax, style)
    ax.legend(frameon=False, fontsize=10, loc="best")
    fig.subplots_adjust(top=0.88, bottom=0.16, right=0.97, left=0.10)

    safe_instance = instance_name.replace(".", "_")
    safe_locations = "_".join(str(location).replace("-", "") for location in locations)
    if save_png:
        save_current_figure(
            fig,
            f"voll_timeseries_locations_{provider}_{safe_instance}_{safe_locations}_{scenario}.png",
            output_dir=output_dir,
            style=style,
        )
    if save_csv:
        write_csv(
            combined_daily,
            f"voll_daily_locations_{provider}_{safe_instance}_{safe_locations}_{scenario}.csv",
            output_dir=output_dir,
        )
    if show:
        plt.show()
    else:
        plt.close(fig)
    return combined_daily


def plot_voll_spot_timeseries_clouds(
    df_power: Optional[pd.DataFrame] = None,
    scenario: str = "trng_median",
    aws_instance_name: str = "p4d.24xlarge",
    aws_region: str = "us-east-1",
    azure_instance_name: str = "Standard_ND96amsr_A100_v4",
    azure_region: str = "eastus",
    azure_ondemand_price: float = 32.77,
    months: Optional[int] = 6,
    start_date: Optional[str] = None,
    power_file: Path = DEFAULT_POWER_FILE,
    aws_price_file: Optional[Path] = None,
    azure_price_file: Path = ORIGINAL_PRICES_DIR / "A100" / "Azure_StandardND96amsr_A100_v4.csv",
    prices_dir: Path = ORIGINAL_PRICES_DIR,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
    save_png: bool = True,
    save_csv: bool = True,
    show: bool = True,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    style: PlotStyle = PlotStyle(),
) -> pd.DataFrame:
    scenario = normalize_scenario(scenario)
    if df_power is None:
        df_power = load_power_data(power_file)

    power_row = get_instance_row(df_power, instance_name=aws_instance_name, provider="aws")
    power_col = total_power_column(scenario)
    if power_col not in df_power.columns:
        raise ValueError(f"Column not found: {power_col}")
    power_w = pd.to_numeric(power_row[power_col], errors="coerce")
    if pd.isna(power_w) or power_w <= 0:
        raise ValueError(f"Invalid power value for {aws_instance_name} {scenario}: {power_w}")
    power_mw = power_w / 1_000_000.0

    aws_price = load_price_data(
        filepath=aws_price_file,
        instance_name=aws_instance_name,
        region=aws_region,
        provider="aws",
        prices_dir=prices_dir,
    )
    aws_voll = compute_voll_timeseries(
        df_price=aws_price,
        df_power=df_power,
        instance_name=aws_instance_name,
        provider="aws",
        scenarios=[scenario],
    )
    aws_daily = daily_voll_summary(aws_voll, scenario)
    aws_daily["series"] = f"EC2 {aws_instance_name} {aws_region}"

    azure_raw = load_csv(azure_price_file, label="Azure price data")
    azure = azure_raw.loc[
        azure_raw["VM Name"].astype(str).str.strip() == azure_instance_name
    ].copy()
    if azure_region:
        azure = azure.loc[azure["Region ID"].astype(str).str.strip() == azure_region].copy()
    if azure.empty:
        raise ValueError(
            f"No Azure records found for {azure_instance_name}"
            f"{f' in {azure_region}' if azure_region else ''}."
        )

    azure["timestamp"] = pd.to_datetime(azure["Modified Date Time"], errors="coerce")
    azure["Instance Price"] = pd.to_numeric(azure["Linux Price"], errors="coerce")
    azure = azure.dropna(subset=["timestamp", "Instance Price"]).copy()
    azure["date"] = azure["timestamp"].dt.floor("D")
    azure[f"VOLL_{scenario}"] = azure["Instance Price"] / power_mw
    azure_daily = daily_voll_summary(azure, scenario)
    azure_daily["series"] = f"Azure {azure_instance_name} {azure_region}"

    combined_daily = pd.concat([aws_daily, azure_daily], ignore_index=True)
    max_date = pd.to_datetime(combined_daily["date"]).max()
    cutoff_date = (
        pd.Timestamp(start_date).floor("D")
        if start_date is not None
        else max_date - pd.DateOffset(months=months)
    )
    combined_daily = combined_daily.loc[combined_daily["date"] >= cutoff_date].copy()
    if combined_daily.empty:
        raise ValueError("No data available in the requested comparison window.")

    fig, ax = plt.subplots(figsize=style.figsize)
    series_colors = {
        f"EC2 {aws_instance_name} {aws_region}": "tab:blue",
        f"Azure {azure_instance_name} {azure_region}": "tab:orange",
    }
    for series_name, series_df in combined_daily.groupby("series", sort=False):
        color = series_colors.get(series_name)
        ax.fill_between(
            series_df["date"],
            series_df["voll_min"],
            series_df["voll_max"],
            alpha=style.envelope_alpha * 0.55,
            color=color,
        )
        ax.plot(
            series_df["date"],
            series_df["voll_mean"],
            linewidth=2.3,
            color=color,
            linestyle="--",
            label=f"{series_name} spot VOLL - Ave",
        )

    aws_ondemand_trajectory = historical_ondemand_voll_trajectory(
        df_voll=aws_voll,
        scenario=scenario,
        instance_name=aws_instance_name,
        provider="aws",
    )
    if aws_ondemand_trajectory is None:
        raise ValueError(f"No EC2 on-demand trajectory available for {aws_instance_name}.")
    aws_ondemand_trajectory = aws_ondemand_trajectory.loc[
        aws_ondemand_trajectory["date"] >= cutoff_date
    ].copy()
    ax.plot(
        aws_ondemand_trajectory["date"],
        aws_ondemand_trajectory["voll_ondemand_historical"],
        color="tab:blue",
        linewidth=2.2,
        linestyle="-",
        drawstyle="steps-post",
        label=f"EC2 on-demand implied VOLL ({scenario})",
    )

    azure_ondemand_voll = azure_ondemand_price / power_mw
    ax.axhline(
        y=azure_ondemand_voll,
        color="tab:orange",
        linewidth=2.2,
        linestyle="-",
        label=f"Azure on-demand implied VOLL ({scenario})",
    )
    combined_daily = pd.merge_asof(
        combined_daily.sort_values("date"),
        aws_ondemand_trajectory.rename(
            columns={
                "ondemand_price_usd_per_hour": "aws_ondemand_price_usd_per_hour",
                "voll_ondemand_historical": "aws_ondemand_voll",
            }
        ).sort_values("date"),
        on="date",
        direction="backward",
    )
    combined_daily["azure_ondemand_price_usd_per_hour"] = azure_ondemand_price
    combined_daily["azure_ondemand_voll"] = azure_ondemand_voll
    combined_daily["comparison_power_mw"] = power_mw

    window_text = (
        f"since {cutoff_date:%b %Y}"
        if start_date is not None
        else f"last {months} months"
    )
    ax.set_title(
        f"Spot and on-demand VoCL comparison, {window_text}\n"
        f"EC2 {aws_instance_name} {aws_region} vs Azure {azure_instance_name} {azure_region}",
        fontsize=14,
        pad=14,
    )
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("VoCL ($/MWh)", fontsize=12)
    if ymin is not None or ymax is not None:
        ax.set_ylim(bottom=ymin, top=ymax)

    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    tidy_axis(ax, style)
    ax.legend(frameon=False, fontsize=10, loc="best")
    fig.subplots_adjust(top=0.86, bottom=0.16, right=0.97, left=0.10)

    safe_aws = aws_instance_name.replace(".", "_")
    safe_azure = azure_instance_name.replace(".", "_")
    if save_png:
        save_current_figure(
            fig,
            f"voll_spot_timeseries_clouds_{safe_aws}_{aws_region}_{safe_azure}_{azure_region}_{scenario}.png",
            output_dir=output_dir,
            style=style,
        )
    if save_csv:
        write_csv(
            combined_daily,
            f"voll_spot_timeseries_clouds_{safe_aws}_{aws_region}_{safe_azure}_{azure_region}_{scenario}.csv",
            output_dir=output_dir,
        )
    if show:
        plt.show()
    else:
        plt.close(fig)
    return combined_daily


def plot_voll_ondemand_regions(
    df_power: Optional[pd.DataFrame] = None,
    scenario: str = "trng_median",
    instance_name: str = "p4d.24xlarge",
    provider: str = "aws",
    base_region: str = "us-east-1",
    regional_price_file: Path = ORIGINAL_PRICES_DIR / "A100" / "EC2_p4d24xlarge_ondemandregional.csv",
    power_file: Path = DEFAULT_POWER_FILE,
    sort_ascending: bool = True,
    save_png: bool = True,
    save_csv: bool = True,
    show: bool = True,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    style: PlotStyle = PlotStyle(),
) -> pd.DataFrame:
    scenario = normalize_scenario(scenario)
    if df_power is None:
        df_power = load_power_data(power_file)

    power_row = get_instance_row(df_power, instance_name=instance_name, provider=provider)
    base_voll_col = ondemand_voll_column(scenario)
    if base_voll_col not in df_power.columns:
        raise ValueError(f"Column not found: {base_voll_col}")

    base_voll = pd.to_numeric(pd.Series([power_row[base_voll_col]]), errors="coerce").iloc[0]
    if pd.isna(base_voll) or base_voll <= 0:
        raise ValueError(f"Invalid base on-demand VOLL for {instance_name} {scenario}: {base_voll}")

    df_prices = load_csv(regional_price_file, label="regional on-demand price data")
    required_cols = ["Geography", "Region Name", "Region", "Instance Price"]
    missing = [col for col in required_cols if col not in df_prices.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df_plot = df_prices[required_cols].dropna(subset=["Region", "Instance Price"]).copy()
    df_plot["Instance Price"] = pd.to_numeric(df_plot["Instance Price"], errors="coerce")
    df_plot = df_plot.dropna(subset=["Instance Price"]).copy()
    df_plot = df_plot.loc[df_plot["Instance Price"] > 0].copy()
    if df_plot.empty:
        raise ValueError("No regional on-demand price data available to plot.")

    base_price_values = df_plot.loc[df_plot["Region"] == base_region, "Instance Price"]
    if base_price_values.empty:
        raise ValueError(f"Base region '{base_region}' not found in {regional_price_file}")
    base_price = float(base_price_values.iloc[0])

    df_plot["base_region"] = base_region
    df_plot["base_price_usd_per_hour"] = base_price
    df_plot["base_voll_usd_per_mwh"] = base_voll
    df_plot["ondemand_voll_usd_per_mwh"] = (
        df_plot["Instance Price"] / base_price * base_voll
    )
    df_plot["price_premium_pct_vs_base"] = (
        df_plot["Instance Price"] / base_price - 1.0
    ) * 100
    df_plot = df_plot.sort_values(
        "ondemand_voll_usd_per_mwh",
        ascending=sort_ascending,
    ).reset_index(drop=True)

    fig_width = max(12.5, 0.62 * len(df_plot))
    fig, ax = plt.subplots(figsize=(fig_width, 6.8))
    x = np.arange(len(df_plot))

    geographies = df_plot["Geography"].fillna("Unknown").astype(str)
    unique_geographies = sorted(geographies.unique())
    cmap = plt.colormaps.get_cmap("tab20").resampled(len(unique_geographies))
    geography_to_color = {
        geography: cmap(idx) for idx, geography in enumerate(unique_geographies)
    }
    colors = geographies.map(geography_to_color)

    ax.bar(
        x,
        df_plot["ondemand_voll_usd_per_mwh"],
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.78,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df_plot["Region"], rotation=45, ha="right")
    ax.set_xlabel("EC2 Region")
    ax.set_ylabel("On-demand VoCL ($/MWh)")
    ax.set_title(
        f"On-demand VoCL by region for {instance_name}\n"
        f"Scenario: {scenario}, scaled from {base_region} on-demand VOLL"
    )
    tidy_axis(ax, style)

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=geography_to_color[geography],
            markeredgecolor="black",
            markersize=9,
            label=geography,
        )
        for geography in unique_geographies
    ]
    ax.legend(
        handles=legend_handles,
        labels=unique_geographies,
        frameon=False,
        fontsize=10,
        loc="best",
    )
    fig.tight_layout()

    safe_instance = instance_name.replace(".", "_")
    if save_png:
        save_current_figure(
            fig,
            f"voll_ondemand_regions_{provider}_{safe_instance}_{scenario}.png",
            output_dir=output_dir,
            style=style,
        )
    if save_csv:
        write_csv(
            df_plot,
            f"voll_ondemand_regions_{provider}_{safe_instance}_{scenario}.csv",
            output_dir=output_dir,
        )
    if show:
        plt.show()
    else:
        plt.close(fig)
    return df_plot


def plot_voll_histogram(
    df_power: pd.DataFrame,
    scenario: str = "mid",
    value_prefix: str = "voll_usd_per_mwh",
    min_mid_voll: Optional[float] = 500,
    bins: int = 10,
    title: Optional[str] = None,
    save_png: bool = False,
    show: bool = True,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    style: PlotStyle = PlotStyle(),
 ) -> pd.Series:
    df_power = filter_low_mid_voll(
        df_power,
        min_mid_voll=min_mid_voll,
        value_prefix=value_prefix,
    )
    data = scenario_values(df_power, scenario, value_prefix).dropna()
    if data.empty:
        raise ValueError(f"No data available for {value_prefix}_{scenario}")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(data, bins=bins, edgecolor="black", alpha=0.75)
    ax.set_xlabel("VoCL ($/MWh)")
    ax.set_ylabel("Count")
    ax.set_title(title or f"Distribution of VoCL ({scenario})")
    tidy_axis(ax, style)
    fig.tight_layout()

    if save_png:
        save_current_figure(fig, f"voll_histogram_{value_prefix}_{scenario}.png", output_dir, style)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return data


def plot_voll_density(
    df_power: pd.DataFrame,
    scenario: str = "mid",
    value_prefix: str = "voll_usd_per_mwh",
    min_mid_voll: Optional[float] = 500,
    log_x_grid: bool = True,
    title: Optional[str] = None,
    save_png: bool = False,
    show: bool = True,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    style: PlotStyle = PlotStyle(),
 ) -> pd.DataFrame:
    if gaussian_kde is None:
        raise ImportError("scipy is required for KDE density plots.")

    df_power = filter_low_mid_voll(
        df_power,
        min_mid_voll=min_mid_voll,
        value_prefix=value_prefix,
    )
    data = scenario_values(df_power, scenario, value_prefix).dropna()
    data = data[data > 0]
    if len(data) < 2:
        raise ValueError("KDE requires at least two positive observations.")

    kde = gaussian_kde(data)
    if log_x_grid:
        x_vals = np.logspace(np.log10(data.min()), np.log10(data.max()), 500)
    else:
        x_vals = np.linspace(data.min(), data.max(), 500)
    y_vals = kde(x_vals)
    density = pd.DataFrame({"voll": x_vals, "density": y_vals})

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(x_vals, y_vals, linewidth=2)
    ax.set_xlabel("VoCL ($/MWh)")
    ax.set_ylabel("Density")
    ax.set_title(title or f"Density of VoCL ({scenario})")
    tidy_axis(ax, style)
    fig.tight_layout()

    if save_png:
        save_current_figure(fig, f"voll_density_{value_prefix}_{scenario}.png", output_dir, style)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return density

def plot_voll_violin_by_gpu(
    df_power: pd.DataFrame,
    scenario: str,
    voll_basis: str = "ondemand",
    value_prefix: Optional[str] = None,
    gpu_col: str = "gpu_model_clean",
    instance_col: str = "instance_type",
    min_group_size: int = 1,
    min_mid_voll: Optional[float] = 500,
    sort_by: str = "vintage",
    log_y: bool = False,
    title: Optional[str] = None,
    save_png: bool = False,
    save_csv: bool = False,
    show: bool = True,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    style: PlotStyle = PlotStyle(),
) -> pd.DataFrame:
    """Plot one VOLL distribution violin per GPU model, color-coded by GPU vintage group."""
    scenario = normalize_scenario(scenario)
    value_prefix = resolve_voll_value_prefix(voll_basis=voll_basis, value_prefix=value_prefix)
    value_col = voll_value_column(scenario, value_prefix=value_prefix)

    df_power = filter_low_mid_voll(
        df_power,
        min_mid_voll=min_mid_voll,
        value_prefix="voll_usd_per_mwh",
    )

    required_cols = [gpu_col, value_col]
    if instance_col in df_power.columns:
        required_cols.append(instance_col)

    missing = [col for col in required_cols if col not in df_power.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    keep_cols = [gpu_col, value_col]
    if instance_col in df_power.columns:
        keep_cols.append(instance_col)

    df_plot = df_power[keep_cols].copy()
    df_plot[gpu_col] = df_plot[gpu_col].fillna("Unknown").astype(str).str.strip()
    df_plot[value_col] = pd.to_numeric(df_plot[value_col], errors="coerce")
    df_plot = df_plot.dropna(subset=[gpu_col, value_col])
    df_plot = df_plot.loc[df_plot[value_col] > 0].copy()

    df_plot["gpu_vintage_group"] = df_plot[gpu_col].map(classify_gpu_vintage)

    group_sizes = df_plot.groupby(gpu_col)[value_col].transform("size")
    df_plot = df_plot.loc[group_sizes >= min_group_size].copy()

    if df_plot.empty:
        raise ValueError(f"No VOLL data available for {value_col} by {gpu_col}.")

    grouped = df_plot.groupby(gpu_col)[value_col]

    if sort_by == "vintage":
        ordered_groups = []
        for vintage_group, gpu_order in GPU_VINTAGE_GROUPS.items():
            available = [gpu for gpu in gpu_order if gpu in set(df_plot[gpu_col])]
            ordered_groups.extend(available)

        remaining = [
            gpu for gpu in grouped.median().sort_values().index.tolist()
            if gpu not in ordered_groups
        ]
        ordered_groups.extend(remaining)

    elif sort_by == "median":
        ordered_groups = grouped.median().sort_values().index.tolist()
    elif sort_by == "mean":
        ordered_groups = grouped.mean().sort_values().index.tolist()
    elif sort_by == "name":
        ordered_groups = sorted(df_plot[gpu_col].unique())
    else:
        raise ValueError("sort_by must be one of: vintage, median, mean, name")

    fig_width = max(12, 0.82 * len(ordered_groups))
    fig, ax = plt.subplots(figsize=(fig_width, 6.8))

    positions = np.arange(1, len(ordered_groups) + 1)

    for idx, gpu in enumerate(ordered_groups):
        values = df_plot.loc[df_plot[gpu_col] == gpu, value_col].to_numpy()
        pos = positions[idx]

        vintage_group = classify_gpu_vintage(gpu)
        color = GPU_VINTAGE_PALETTE.get(vintage_group, "#9e9e9e")

        if len(values) >= 2 and np.nanstd(values) > 0:
            violin = ax.violinplot(
                [values],
                positions=[pos],
                widths=0.72,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for body in violin["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor("black")
                body.set_alpha(0.45)

        jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.array([0])

        ax.scatter(
            np.full(len(values), pos) + jitter,
            values,
            s=18,
            color=color,
            edgecolor="black",
            linewidth=0.35,
            alpha=0.85,
            zorder=3,
        )

        ax.scatter(
            [pos],
            [np.nanmedian(values)],
            marker="D",
            s=38,
            color="black",
            zorder=4,
            label="Median" if idx == 0 else None,
        )

    # Group separators and headings
    ymax = df_plot[value_col].max()
    ymin = df_plot[value_col].min()
    y_label = ymax * 1.04 if not log_y else ymax * 1.15

    group_boundaries = []
    group_centers = {}

    for vintage_group, gpu_order in GPU_VINTAGE_GROUPS.items():
        present = [gpu for gpu in gpu_order if gpu in ordered_groups]
        if not present:
            continue

        idxs = [ordered_groups.index(gpu) + 1 for gpu in present]
        group_centers[vintage_group] = np.mean(idxs)

        if max(idxs) < len(ordered_groups):
            group_boundaries.append(max(idxs) + 0.5)

    for boundary in group_boundaries:
        ax.axvline(
            boundary,
            color="black",
            linestyle="--",
            linewidth=1.0,
            alpha=0.45,
        )

    for vintage_group, center in group_centers.items():
        ax.text(
            center,
            y_label,
            GPU_VINTAGE_SHORT_LABELS.get(vintage_group, vintage_group),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=GPU_VINTAGE_PALETTE.get(vintage_group, "black"),
        )

    basis_label = VOLL_BASIS_LABELS.get(value_prefix, value_prefix)

    ax.set_xticks(positions)
    ax.set_xticklabels(ordered_groups, rotation=45, ha="right")
    ax.set_xlabel("GPU Model")
    ax.set_ylabel("VoCL ($/MWh)")
    ax.set_title(title or f"{basis_label} VoCL by GPU - {scenario}")

    if log_y:
        ax.set_yscale("log")
    else:
        ax.set_ylim(top=ymax * 1.16)

    tidy_axis(ax, style)

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=10,
            label=group,
        )
        for group, color in GPU_VINTAGE_PALETTE.items()
        if group in set(df_plot["gpu_vintage_group"])
    ]

    legend_handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="D",
            color="black",
            linestyle="none",
            markersize=7,
            label="Median",
        )
    )

    ax.legend(
        handles=legend_handles,
        frameon=True,
        loc="lower right",
        ncol=3,
    )

    fig.tight_layout()

    safe_prefix = value_prefix.replace("_usd_per_mwh", "").replace("voll_", "voll")

    if save_png:
        save_current_figure(
            fig,
            f"voll_violin_by_gpu_{safe_prefix}_{scenario}_vintage_groups.png",
            output_dir=output_dir,
            style=style,
        )

    if save_csv:
        write_csv(
            df_plot,
            f"voll_violin_by_gpu_{safe_prefix}_{scenario}_vintage_groups.csv",
            output_dir=output_dir,
        )

    if show:
        plt.show()
    else:
        plt.close(fig)

    return df_plot

def prepare_voll_gpu_panel_data(
    df_power: pd.DataFrame,
    scenarios: Sequence[str] = ("trng_median", "inf_median"),
    voll_bases: Sequence[str] = ("ondemand", "spot"),
    gpu_col: str = "gpu_model_clean",
    instance_col: str = "instance_type",
    min_mid_voll: Optional[float] = 500,
    sort_by: str = "vintage",
) -> Tuple[pd.DataFrame, List[str], List[Tuple[str, str, str]]]:
    scenarios = [normalize_scenario(scenario) for scenario in scenarios]
    raw_value_specs = [
        (basis, scenario, voll_value_column(scenario, voll_basis=basis))
        for basis in voll_bases
        for scenario in scenarios
    ]
    required_cols = [gpu_col] + [value_col for _, _, value_col in raw_value_specs]
    if instance_col in df_power.columns:
        required_cols.append(instance_col)
    missing = [col for col in required_cols if col not in df_power.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df_filtered = filter_low_mid_voll(
        df_power,
        min_mid_voll=min_mid_voll,
        value_prefix="voll_usd_per_mwh",
    )
    df_plot = df_filtered[list(dict.fromkeys(required_cols))].copy()
    df_plot[gpu_col] = df_plot[gpu_col].fillna("Unknown").astype(str).str.strip()
    df_plot = df_plot.loc[df_plot[gpu_col] != ""].copy()
    for _, _, value_col in raw_value_specs:
        df_plot[value_col] = pd.to_numeric(df_plot[value_col], errors="coerce")

    value_specs = raw_value_specs

    available_mask = pd.Series(False, index=df_plot.index)
    for _, _, value_col in value_specs:
        available_mask |= df_plot[value_col] > 0
    df_plot = df_plot.loc[available_mask].copy()
    if df_plot.empty:
        raise ValueError("No VOLL data available for the requested panel.")

    df_plot["gpu_vintage_group"] = df_plot[gpu_col].map(classify_gpu_vintage)
    if sort_by in {"vintage", "median"}:
        median_cols = [value_col for _, _, value_col in value_specs]
        median_by_gpu = (
            df_plot.set_index(gpu_col)[median_cols]
            .median(axis=1, skipna=True)
            .groupby(level=0)
            .median()
            .sort_values()
        )

    if sort_by == "vintage":
        ordered_groups = []
        present_gpus = set(df_plot[gpu_col])
        for _, gpu_order in GPU_VINTAGE_GROUPS.items():
            ordered_groups.extend(gpu for gpu in gpu_order if gpu in present_gpus)
        ordered_groups.extend(gpu for gpu in median_by_gpu.index.tolist() if gpu not in ordered_groups)
    elif sort_by == "median":
        ordered_groups = median_by_gpu.index.tolist()
    elif sort_by == "name":
        ordered_groups = sorted(df_plot[gpu_col].unique())
    else:
        raise ValueError("sort_by must be one of: vintage, median, name")

    return df_plot, ordered_groups, value_specs


def voll_gpu_panel_legend_handles(df_plot: pd.DataFrame) -> List[Any]:
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=9,
            label=group,
        )
        for group, color in GPU_VINTAGE_PALETTE.items()
        if group in set(df_plot["gpu_vintage_group"])
    ]
    legend_handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="D",
            color="black",
            linestyle="none",
            markersize=6,
            label="Median",
        )
    )
    return legend_handles


def add_voll_gpu_vintage_titles(
    ax: plt.Axes,
    ordered_groups: Sequence[str],
    y: float = 0.965,
    fontsize: int = 8,
) -> None:
    group_centers = {}
    for vintage_group, gpu_order in GPU_VINTAGE_GROUPS.items():
        present = [gpu for gpu in gpu_order if gpu in ordered_groups]
        if not present:
            continue
        idxs = [ordered_groups.index(gpu) + 1 for gpu in present]
        group_centers[vintage_group] = np.mean(idxs)

    for vintage_group, center in group_centers.items():
        ax.text(
            center,
            y,
            GPU_VINTAGE_SHORT_LABELS.get(vintage_group, vintage_group),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=fontsize,
            fontweight="bold",
            color=GPU_VINTAGE_PALETTE.get(vintage_group, "black"),
        )


def draw_voll_gpu_panel_axis(
    ax: plt.Axes,
    df_plot: pd.DataFrame,
    ordered_groups: Sequence[str],
    value_col: str,
    gpu_col: str,
    title: str,
    kind: str,
    log_y: bool,
    show_xticklabels: bool,
    style: PlotStyle,
    allow_zero: bool = False,
) -> None:
    positions = np.arange(1, len(ordered_groups) + 1)
    for pos, gpu in zip(positions, ordered_groups):
        min_value = 0 if allow_zero else 0
        comparator = (lambda series: series >= min_value) if allow_zero else (lambda series: series > min_value)
        values = (
            df_plot.loc[df_plot[gpu_col] == gpu, value_col]
            .dropna()
            .loc[comparator]
            .to_numpy()
        )
        if len(values) == 0:
            continue

        vintage_group = classify_gpu_vintage(gpu)
        color = GPU_VINTAGE_PALETTE.get(vintage_group, "#9e9e9e")
        if kind == "violin":
            if len(values) >= 2 and np.nanstd(values) > 0:
                violin = ax.violinplot(
                    [values],
                    positions=[pos],
                    widths=0.72,
                    showmeans=False,
                    showmedians=False,
                    showextrema=False,
                )
                for body in violin["bodies"]:
                    body.set_facecolor(color)
                    body.set_edgecolor("black")
                    body.set_alpha(0.45)

            jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.array([0])
            ax.scatter(
                np.full(len(values), pos) + jitter,
                values,
                s=14,
                color=color,
                edgecolor="black",
                linewidth=0.3,
                alpha=0.8,
                zorder=3,
            )
        elif kind == "box":
            box = ax.boxplot(
                [values],
                positions=[pos],
                widths=0.58,
                patch_artist=True,
                showfliers=True,
                medianprops={"color": "black", "linewidth": 1.5},
                boxprops={"edgecolor": "black", "linewidth": 0.9},
                whiskerprops={"color": "black", "linewidth": 0.9},
                capprops={"color": "black", "linewidth": 0.9},
                flierprops={
                    "marker": "o",
                    "markersize": 3,
                    "markerfacecolor": "white",
                    "markeredgecolor": "black",
                    "alpha": 0.65,
                },
            )
            for patch in box["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.55)
        else:
            raise ValueError("kind must be either 'violin' or 'box'")

        ax.scatter([pos], [np.nanmedian(values)], marker="D", s=28, color="black", zorder=4)

    for _, gpu_order in GPU_VINTAGE_GROUPS.items():
        present = [gpu for gpu in gpu_order if gpu in ordered_groups]
        if not present:
            continue
        boundary = max(ordered_groups.index(gpu) + 1 for gpu in present) + 0.5
        if boundary < len(ordered_groups) + 0.5:
            ax.axvline(
                boundary,
                color="black",
                linestyle="--",
                linewidth=0.8,
                alpha=0.35,
            )

    ax.set_title(title)
    ax.set_ylabel("VoCL ($/MWh)")
    ax.set_xticks(positions)
    if show_xticklabels:
        ax.set_xticklabels(ordered_groups, rotation=45, ha="right", fontsize=9)
        ax.set_xlabel("GPU Model")
    else:
        ax.set_xticklabels([])
    if log_y:
        ax.set_yscale("log")
    tidy_axis(ax, style)


def plot_voll_gpu_basis_scenario_panel(
    df_power: pd.DataFrame,
    kind: str = "violin",
    scenarios: Sequence[str] = ("trng_median", "inf_median"),
    voll_bases: Sequence[str] = ("ondemand", "spot"),
    gpu_col: str = "gpu_model_clean",
    instance_col: str = "instance_type",
    min_mid_voll: Optional[float] = 500,
    sort_by: str = "vintage",
    log_y: bool = False,
    title: Optional[str] = None,
    save_png: bool = False,
    save_csv: bool = False,
    show: bool = True,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    style: PlotStyle = PlotStyle(),
) -> pd.DataFrame:
    df_plot, ordered_groups, value_specs = prepare_voll_gpu_panel_data(
        df_power=df_power,
        scenarios=scenarios,
        voll_bases=voll_bases,
        gpu_col=gpu_col,
        instance_col=instance_col,
        min_mid_voll=min_mid_voll,
        sort_by=sort_by,
    )
    scenarios = [normalize_scenario(scenario) for scenario in scenarios]
    value_lookup = {(basis, scenario): value_col for basis, scenario, value_col in value_specs}

    fig_width = max(15, 0.72 * len(ordered_groups))
    fig, axes = plt.subplots(2, 2, figsize=(fig_width, 10.5), sharex=True, sharey=True)
    for row_idx, basis in enumerate(voll_bases):
        for col_idx, scenario in enumerate(scenarios):
            value_col = value_lookup[(basis, scenario)]
            basis_label = voll_basis_panel_label(basis)
            draw_voll_gpu_panel_axis(
                ax=axes[row_idx, col_idx],
                df_plot=df_plot,
                ordered_groups=ordered_groups,
                value_col=value_col,
                gpu_col=gpu_col,
                title=f"{basis_label} - {scenario}",
                kind=kind,
                log_y=log_y,
                show_xticklabels=row_idx == len(voll_bases) - 1,
                style=style,
            )

    axes[0, 1].legend(
        handles=voll_gpu_panel_legend_handles(df_plot),
        frameon=True,
        loc="lower right",
        ncol=1,
        fontsize=8,
    )
    fig.suptitle(
        title or f"VoCL distributions by GPU, price basis, and scenario ({kind})",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    safe_scenarios = "_".join(scenarios)
    safe_bases = "_".join(voll_bases)
    filename_base = f"voll_{kind}_panel_by_gpu_{safe_bases}_{safe_scenarios}"
    if save_png:
        save_current_figure(fig, f"{filename_base}.png", output_dir, style)
    if save_csv:
        write_csv(df_plot, f"{filename_base}.csv", output_dir=output_dir)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return df_plot


def plot_voll_violin_basis_scenario_panel(
    df_power: pd.DataFrame,
    scenarios: Sequence[str] = ("trng_median", "inf_median"),
    save_png: bool = False,
    save_csv: bool = False,
    show: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    return plot_voll_gpu_basis_scenario_panel(
        df_power=df_power,
        kind="violin",
        scenarios=scenarios,
        save_png=save_png,
        save_csv=save_csv,
        show=show,
        **kwargs,
    )


def plot_voll_boxplot_basis_scenario_panel(
    df_power: pd.DataFrame,
    scenarios: Sequence[str] = ("trng_median", "inf_median"),
    save_png: bool = False,
    save_csv: bool = False,
    show: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    return plot_voll_gpu_basis_scenario_panel(
        df_power=df_power,
        kind="box",
        scenarios=scenarios,
        save_png=save_png,
        save_csv=save_csv,
        show=show,
        **kwargs,
    )


def plot_voll_boxplot_sla_spot_scenario_panel(
    df_power: pd.DataFrame,
    scenarios: Sequence[str] = ("trng_median", "inf_median"),
    h_cumulative_values: Sequence[float] = (0.0, 4.0),
    u_mth: float = U_MTH,
    cloud_platform: str = SLA_CLOUD_PLATFORM,
    hours_per_month: float = HOURS_PER_MONTH,
    gpu_col: str = "gpu_model_clean",
    instance_col: str = "instance_type",
    min_mid_voll: Optional[float] = 500,
    sort_by: str = "vintage",
    log_y: bool = False,
    title: Optional[str] = None,
    save_png: bool = False,
    save_csv: bool = False,
    show: bool = True,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    style: PlotStyle = PlotStyle(),
    include_vintage_titles: bool = False,
    include_sla_rows: bool = True,
    include_ondemand_row: bool = True,
    include_spot_row: bool = True,
    ) -> pd.DataFrame:
    scenarios = [normalize_scenario(scenario) for scenario in scenarios]
    df_plot, ordered_groups, value_specs = prepare_voll_gpu_panel_data(
        df_power=df_power,
        scenarios=scenarios,
        voll_bases=("ondemand", "spot"),
        gpu_col=gpu_col,
        instance_col=instance_col,
        min_mid_voll=min_mid_voll,
        sort_by=sort_by,
    )
    value_lookup = {(basis, scenario): value_col for basis, scenario, value_col in value_specs}

    row_specs = []
    for h_cumulative in h_cumulative_values:
        if not include_ondemand_row and float(h_cumulative) == 0.0:
            continue
        if not include_sla_rows and float(h_cumulative) != 0.0:
            continue
        if float(h_cumulative) == 0.0:
            service_credit_share = 0.0
            scenario_cols = {
                scenario: value_lookup[("ondemand", scenario)]
                for scenario in scenarios
            }
            basis = "ondemand"
            label = "On-demand (no cumulative downtime)"
        else:
            service_credit_share = relevant_service_credit_share(
                h_cumulative=h_cumulative,
                cloud_platform=cloud_platform,
            )
            multiplier = 1 + service_credit_share * hours_per_month * u_mth
            scenario_cols = {}
            for scenario in scenarios:
                base_col = value_lookup[("ondemand", scenario)]
                adjusted_col = (
                    f"voll_ondemand_incl_sla_h{str(h_cumulative).replace('.', 'p')}_"
                    f"usd_per_mwh_{scenario}"
                )
                df_plot[adjusted_col] = pd.to_numeric(df_plot[base_col], errors="coerce") * multiplier
                scenario_cols[scenario] = adjusted_col
            basis = f"ondemand_sla_h{str(h_cumulative).replace('.', 'p')}"
            label = f"On-demand incl. SLA ({h_cumulative:g} hrs cumulative downtime)"
        row_specs.append(
            {
                "basis": basis,
                "label": label,
                "h_cumulative": h_cumulative,
                "service_credit_share": service_credit_share,
                "column_for_scenario": scenario_cols,
            }
        )

    if include_spot_row:
        row_specs.append(
            {
                "basis": "spot",
                "label": "Spot",
                "h_cumulative": np.nan,
                "service_credit_share": np.nan,
                "column_for_scenario": {
                    scenario: value_lookup[("spot", scenario)]
                    for scenario in scenarios
                },
            }
        )
    if not row_specs:
        raise ValueError("No rows selected for the VOLL GPU scenario panel.")

    fig_width = max(15, 0.72 * len(ordered_groups))
    fig_height_by_rows = {1: 5.8, 2: 9.8}
    fig_height = fig_height_by_rows.get(len(row_specs), 14.2)
    fig, axes = plt.subplots(
        len(row_specs),
        len(scenarios),
        figsize=(fig_width, fig_height),
        sharex=True,
        sharey=False,
        squeeze=False,
    )
    for row_idx, row_spec in enumerate(row_specs):
        row_ylim = (0, 140000) if str(row_spec["basis"]).startswith("ondemand_sla") else (0, 20000)
        for col_idx, scenario in enumerate(scenarios):
            draw_voll_gpu_panel_axis(
                ax=axes[row_idx, col_idx],
                df_plot=df_plot,
                ordered_groups=ordered_groups,
                value_col=row_spec["column_for_scenario"][scenario],
                gpu_col=gpu_col,
                title=f"{row_spec['label']} - {scenario}",
                kind="box",
                log_y=log_y,
                show_xticklabels=row_idx == len(row_specs) - 1,
                style=style,
                allow_zero=True,
            )
            if not log_y:
                axes[row_idx, col_idx].set_ylim(row_ylim)
            if include_vintage_titles:
                add_voll_gpu_vintage_titles(
                    axes[row_idx, col_idx],
                    ordered_groups=ordered_groups,
                    y=0.965,
                    fontsize=8,
                )

    axes[0, -1].legend(
        handles=voll_gpu_panel_legend_handles(df_plot),
        frameon=True,
        loc="lower right",
        ncol=1,
        fontsize=8,
    )
    fig.suptitle(
        title or (
            "VoCL distributions by GPU, SLA-adjusted on-demand, spot, and scenario"
            if include_sla_rows
            else "VoCL distributions by GPU, on-demand, spot, and scenario"
        ),
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))

    metadata_cols = [gpu_col, "gpu_vintage_group"]
    if instance_col in df_plot.columns:
        metadata_cols.append(instance_col)
    output_cols = list(dict.fromkeys(metadata_cols))
    for row_spec in row_specs:
        for scenario, value_col in row_spec["column_for_scenario"].items():
            output_col = f"{row_spec['basis']}_{scenario}"
            df_plot[output_col] = df_plot[value_col]
            output_cols.append(output_col)
    df_out = df_plot[output_cols].copy()

    safe_h = "_".join(f"h{str(h).replace('.', 'p')}" for h in h_cumulative_values)
    safe_scenarios = "_".join(scenarios)
    if include_sla_rows and include_ondemand_row and include_spot_row:
        filename_base = f"voll_box_panel_by_gpu_ondemand_sla_{safe_h}_spot_{safe_scenarios}"
    elif include_ondemand_row and include_spot_row:
        filename_base = f"voll_box_panel_by_gpu_ondemand_spot_{safe_scenarios}"
    elif include_sla_rows and not include_ondemand_row and not include_spot_row:
        filename_base = f"voll_box_panel_by_gpu_ondemand_sla_{safe_h}_{safe_scenarios}"
    else:
        filename_base = f"voll_box_panel_by_gpu_custom_{safe_scenarios}"
    if include_vintage_titles:
        filename_base = f"{filename_base}_vintage_titles"
    if save_png:
        save_current_figure(fig, f"{filename_base}.png", output_dir, style)
    if save_csv:
        write_csv(df_out, f"{filename_base}.csv", output_dir=output_dir)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return df_out


def plot_voll_violin_by_gpu_old(
    df_power: pd.DataFrame,
    scenario: str,
    voll_basis: str = "ondemand",
    value_prefix: Optional[str] = None,
    gpu_col: str = "gpu_model_clean",
    instance_col: str = "instance_type",
    min_group_size: int = 1,
    min_mid_voll: Optional[float] = 500,
    sort_by: str = "median",
    log_y: bool = False,
    title: Optional[str] = None,
    save_png: bool = False,
    save_csv: bool = False,
    show: bool = True,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    style: PlotStyle = PlotStyle(),
) -> pd.DataFrame:
    """Plot one VOLL distribution violin per GPU model for a scenario and price basis."""
    scenario = normalize_scenario(scenario)
    value_prefix = resolve_voll_value_prefix(voll_basis=voll_basis, value_prefix=value_prefix)
    value_col = voll_value_column(scenario, value_prefix=value_prefix)
    df_power = filter_low_mid_voll(
        df_power,
        min_mid_voll=min_mid_voll,
        value_prefix="voll_usd_per_mwh",
    )

    required_cols = [gpu_col, value_col]
    if instance_col in df_power.columns:
        required_cols.append(instance_col)
    missing = [col for col in required_cols if col not in df_power.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    keep_cols = [gpu_col, value_col]
    if instance_col in df_power.columns:
        keep_cols.append(instance_col)

    df_plot = df_power[keep_cols].copy()
    df_plot[gpu_col] = df_plot[gpu_col].fillna("Unknown").astype(str)
    df_plot[value_col] = pd.to_numeric(df_plot[value_col], errors="coerce")
    df_plot = df_plot.dropna(subset=[gpu_col, value_col])
    df_plot = df_plot.loc[df_plot[value_col] > 0].copy()

    group_sizes = df_plot.groupby(gpu_col)[value_col].transform("size")
    df_plot = df_plot.loc[group_sizes >= min_group_size].copy()
    if df_plot.empty:
        raise ValueError(f"No VOLL data available for {value_col} by {gpu_col}.")

    grouped = df_plot.groupby(gpu_col)[value_col]
    if sort_by == "median":
        ordered_groups = grouped.median().sort_values().index.tolist()
    elif sort_by == "mean":
        ordered_groups = grouped.mean().sort_values().index.tolist()
    elif sort_by == "name":
        ordered_groups = sorted(df_plot[gpu_col].unique())
    else:
        raise ValueError("sort_by must be one of: median, mean, name")

    fig_width = max(10, 0.8 * len(ordered_groups))
    fig, ax = plt.subplots(figsize=(fig_width, 6.5))

    colors = plt.colormaps.get_cmap("tab20").resampled(len(ordered_groups))
    positions = np.arange(1, len(ordered_groups) + 1)

    for idx, gpu in enumerate(ordered_groups):
        values = df_plot.loc[df_plot[gpu_col] == gpu, value_col].to_numpy()
        pos = positions[idx]

        if len(values) >= 2 and np.nanstd(values) > 0:
            violin = ax.violinplot(
                [values],
                positions=[pos],
                widths=0.72,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for body in violin["bodies"]:
                body.set_facecolor(colors(idx))
                body.set_edgecolor("black")
                body.set_alpha(0.45)

        jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.array([0])
        ax.scatter(
            np.full(len(values), pos) + jitter,
            values,
            s=18,
            color=colors(idx),
            edgecolor="black",
            linewidth=0.35,
            alpha=0.85,
            zorder=3,
        )
        ax.scatter(
            [pos],
            [np.nanmedian(values)],
            marker="D",
            s=34,
            color="black",
            zorder=4,
            label="Median" if idx == 0 else None,
        )

    basis_label = VOLL_BASIS_LABELS.get(value_prefix, value_prefix)
    ax.set_xticks(positions)
    ax.set_xticklabels(ordered_groups, rotation=45, ha="right")
    ax.set_xlabel("GPU Model")
    ax.set_ylabel("VoCL ($/MWh)")
    ax.set_title(title or f"{basis_label} VoCL by GPU - {scenario}")
    if log_y:
        ax.set_yscale("log")
    tidy_axis(ax, style)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, loc="best")
    fig.tight_layout()

    safe_prefix = value_prefix.replace("_usd_per_mwh", "").replace("voll_", "voll")
    if save_png:
        save_current_figure(
            fig,
            f"voll_violin_by_gpu_{safe_prefix}_{scenario}.png",
            output_dir=output_dir,
            style=style,
        )
    if save_csv:
        write_csv(
            df_plot,
            f"voll_violin_by_gpu_{safe_prefix}_{scenario}.csv",
            output_dir=output_dir,
        )
    if show:
        plt.show()
    else:
        plt.close(fig)
    return df_plot


def plot_voll_range_by_instance(
    df_power: pd.DataFrame,
    instance_col: str = "instance_type",
    low_scenario: str = "lo",
    mid_scenario: str = "mid",
    high_scenario: str = "hi",
    value_prefix: str = "voll_usd_per_mwh",
    color_col: Optional[str] = "gpu_model_clean",
    marker_scenarios: Optional[Mapping[str, str]] = None,
    range_label: str = "p10-p90 range",
    min_mid_voll: Optional[float] = 500,
    show_instance_labels: bool = False,
    title: Optional[str] = None,
    save_png: bool = False,
    show: bool = True,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    style: PlotStyle = PlotStyle(),
) -> pd.DataFrame:
    low_scenario = normalize_scenario(low_scenario)
    mid_scenario = normalize_scenario(mid_scenario)
    high_scenario = normalize_scenario(high_scenario)
    if marker_scenarios is None:
        marker_scenarios = {
            "inf_median": "Inference median",
            "trng_median": "Training median",
        }

    df_power = filter_low_mid_voll(
        df_power,
        min_mid_voll=min_mid_voll,
        value_prefix="voll_usd_per_mwh",
    )

    required_cols = [
        instance_col,
        f"{value_prefix}_{low_scenario}",
        f"{value_prefix}_{high_scenario}",
    ]
    required_cols.extend(
        f"{value_prefix}_{normalize_scenario(scenario)}"
        for scenario in marker_scenarios
    )
    missing = [col for col in required_cols if col not in df_power.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df_plot = sort_by_scenario(df_power, scenario=mid_scenario, value_prefix=value_prefix)
    x = np.arange(len(df_plot))
    y_low = pd.to_numeric(df_plot[f"{value_prefix}_{low_scenario}"], errors="coerce")
    y_mid = pd.to_numeric(df_plot[f"{value_prefix}_{mid_scenario}"], errors="coerce")
    y_high = pd.to_numeric(df_plot[f"{value_prefix}_{high_scenario}"], errors="coerce")
    y_range = y_high - y_low

    fig_size = (18, 7) if show_instance_labels else (14, 6)
    fig, ax = plt.subplots(figsize=fig_size)

    if color_col and color_col in df_plot.columns:
        groups = df_plot[color_col].fillna("Unknown").astype(str)
        unique_groups = sorted(groups.unique())
        cmap = plt.colormaps.get_cmap("tab20").resampled(len(unique_groups))
        group_to_color = {group: cmap(i) for i, group in enumerate(unique_groups)}
        colors = groups.map(group_to_color)
    else:
        groups = None
        group_to_color = {}
        colors = "tab:blue"

    ax.bar(
        x,
        y_range,
        bottom=y_low,
        width=style.range_bar_width,
        color=colors,
        edgecolor="black",
        linewidth=1.0,
        alpha=style.range_bar_alpha,
        label=range_label,
    )

    markers = ["D", "X", "P", "^", "v", "s", "*"]
    for idx, (scenario, label) in enumerate(marker_scenarios.items()):
        scenario = normalize_scenario(scenario)
        col = f"{value_prefix}_{scenario}"
        y_marker = pd.to_numeric(df_plot[col], errors="coerce")
        ax.scatter(
            x,
            y_marker,
            s=28,
            marker=markers[idx % len(markers)],
            edgecolor="black",
            linewidth=0.35,
            zorder=5,
            label=label,
        )

    if show_instance_labels:
        ax.set_xticks(x)
        ax.set_xticklabels(df_plot[instance_col], rotation=90, ha="center", fontsize=7)
    else:
        ax.set_xticks([])
    ax.set_xlabel("Instances")

    ax.set_ylabel("VoCL ($/MWh)")
    ax.set_title(title or "On-demand VoCL range with training and inference medians by instance")
    tidy_axis(ax, style)

    handles, labels = ax.get_legend_handles_labels()
    if group_to_color:
        group_handles = [
            plt.Line2D([0], [0], color=group_to_color[group], lw=6, label=group)
            for group in group_to_color
        ]
        handles = handles + group_handles
        labels = labels + list(group_to_color.keys())
    legend_title = "GPU class" if color_col == "gpu_model_clean" else color_col
    ax.legend(
        handles,
        labels,
        title=legend_title if group_to_color else None,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0, 0, 0.84, 1))

    if save_png:
        save_current_figure(fig, f"voll_range_by_instance_{value_prefix}_{mid_scenario}.png", output_dir, style)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return df_plot


def plot_spot_interrupt_vs_voll(
    df_power: pd.DataFrame,
    scenario: str = "trng_median",
    interrupt_col: str = "Linux Spot Interrupt Frequency",
    instance_col: str = "instance_type",
    gpu_col: str = "gpu_model_clean",
    color_col: Optional[str] = "gpu_vintage_group",
    min_mid_voll: Optional[float] = 500,
    title: Optional[str] = None,
    save_png: bool = False,
    save_csv: bool = False,
    show: bool = True,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    style: PlotStyle = PlotStyle(),
) -> pd.DataFrame:
    scenario = normalize_scenario(scenario)
    value_col = voll_value_column(scenario, voll_basis="spot")
    df_power = filter_low_mid_voll(
        df_power,
        min_mid_voll=min_mid_voll,
        value_prefix="voll_usd_per_mwh",
    )

    required_cols = [interrupt_col, value_col]
    if instance_col in df_power.columns:
        required_cols.append(instance_col)
    if color_col == "gpu_vintage_group" and gpu_col in df_power.columns:
        required_cols.append(gpu_col)
    if color_col and color_col in df_power.columns:
        required_cols.append(color_col)
    missing = [col for col in required_cols if col not in df_power.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    keep_cols = list(dict.fromkeys(required_cols))
    df_plot = df_power[keep_cols].copy()
    df_plot[interrupt_col] = df_plot[interrupt_col].astype(str).str.strip()
    df_plot.loc[df_plot[interrupt_col].isin(["", "nan", "None"]), interrupt_col] = pd.NA
    df_plot[value_col] = pd.to_numeric(df_plot[value_col], errors="coerce")
    df_plot = df_plot.dropna(subset=[interrupt_col, value_col]).copy()
    df_plot = df_plot.loc[df_plot[value_col] > 0].copy()
    if color_col == "gpu_vintage_group" and gpu_col in df_plot.columns:
        df_plot[color_col] = df_plot[gpu_col].map(classify_gpu_vintage)
    if df_plot.empty:
        raise ValueError(f"No data available for {interrupt_col} vs {value_col}.")

    category_order = [
        category for category in SPOT_INTERRUPT_ORDER if category in set(df_plot[interrupt_col])
    ]
    category_order.extend(
        sorted(category for category in df_plot[interrupt_col].unique() if category not in category_order)
    )
    category_to_x = {category: idx for idx, category in enumerate(category_order)}
    df_plot["interrupt_category_order"] = df_plot[interrupt_col].map(category_to_x)

    fig, ax = plt.subplots(figsize=(10.5, 6.2))

    if color_col and color_col in df_plot.columns:
        groups = df_plot[color_col].fillna("Unknown").astype(str)
        if color_col == "gpu_vintage_group":
            unique_groups = [
                group for group in GPU_VINTAGE_GROUPS
                if group in set(groups)
            ]
            unique_groups.extend(
                sorted(group for group in groups.unique() if group not in unique_groups)
            )
            group_to_color = {
                group: GPU_VINTAGE_PALETTE.get(group, "#9e9e9e")
                for group in unique_groups
            }
        else:
            unique_groups = sorted(groups.unique())
            cmap = plt.colormaps.get_cmap("tab20").resampled(len(unique_groups))
            group_to_color = {group: cmap(i) for i, group in enumerate(unique_groups)}
        for group in unique_groups:
            group_df = df_plot.loc[groups == group]
            jitter = np.linspace(-0.16, 0.16, len(group_df)) if len(group_df) > 1 else np.array([0])
            ax.scatter(
                group_df["interrupt_category_order"].to_numpy() + jitter,
                group_df[value_col],
                s=34,
                color=group_to_color[group],
                edgecolor="black",
                linewidth=0.35,
                alpha=0.85,
                label=group,
            )
    else:
        jitter = np.linspace(-0.16, 0.16, len(df_plot)) if len(df_plot) > 1 else np.array([0])
        ax.scatter(
            df_plot["interrupt_category_order"].to_numpy() + jitter,
            df_plot[value_col],
            s=34,
            edgecolor="black",
            linewidth=0.35,
            alpha=0.85,
        )

    ax.set_xticks(range(len(category_order)))
    ax.set_xticklabels(category_order)
    ax.set_xlabel("Linux Spot Interrupt Frequency")
    ax.set_ylabel("Spot VoCL ($/MWh)")
    ax.set_title(title or f"Linux spot interrupt frequency vs spot VoCL - {scenario}")
    tidy_axis(ax, style)
    if color_col and color_col in df_plot.columns:
        legend_title = "GPU vintage" if color_col == "gpu_vintage_group" else color_col
        ax.legend(title=legend_title, loc="best")
    fig.tight_layout()

    if save_png:
        save_current_figure(
            fig,
            f"spot_interrupt_vs_voll_spot_{scenario}.png",
            output_dir=output_dir,
            style=style,
        )
    if save_csv:
        write_csv(
            df_plot,
            f"spot_interrupt_vs_voll_spot_{scenario}.csv",
            output_dir=output_dir,
        )
    if show:
        plt.show()
    else:
        plt.close(fig)
    return df_plot


def flexibility_discount_by_gpu(
    df_power: pd.DataFrame,
    scenario: str = "inf_median",
    gpu_col: str = "gpu_model_clean",
    instance_col: str = "instance_type",
    min_mid_voll: Optional[float] = 500,
) -> pd.DataFrame:
    scenario = normalize_scenario(scenario)
    ondemand_col = voll_value_column(scenario, voll_basis="ondemand")
    spot_col = voll_value_column(scenario, voll_basis="spot")

    df_power = filter_low_mid_voll(
        df_power,
        min_mid_voll=min_mid_voll,
        value_prefix="voll_usd_per_mwh",
    )

    required_cols = [gpu_col, ondemand_col, spot_col]
    if instance_col in df_power.columns:
        required_cols.append(instance_col)
    missing = [col for col in required_cols if col not in df_power.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df_power[required_cols].copy()
    df[gpu_col] = df[gpu_col].fillna("Unknown").astype(str)
    df[ondemand_col] = pd.to_numeric(df[ondemand_col], errors="coerce")
    df[spot_col] = pd.to_numeric(df[spot_col], errors="coerce")
    df = df.dropna(subset=[gpu_col, ondemand_col, spot_col]).copy()
    df = df.loc[(df[ondemand_col] > 0) & (df[spot_col] > 0)].copy()
    if df.empty:
        raise ValueError(f"No data available for flexibility discount in scenario {scenario}.")

    df["flex_discount_usd_per_mwh"] = df[ondemand_col] - df[spot_col]
    df["flex_discount_pct_of_ondemand"] = (
        df["flex_discount_usd_per_mwh"] / df[ondemand_col] * 100
    )

    summary = (
        df.groupby(gpu_col)
        .agg(
            instance_count=(instance_col if instance_col in df.columns else ondemand_col, "count"),
            avg_voll_ondemand=(ondemand_col, "mean"),
            avg_voll_spot=(spot_col, "mean"),
            avg_flex_discount_usd_per_mwh=("flex_discount_usd_per_mwh", "mean"),
            avg_flex_discount_pct_of_ondemand=("flex_discount_pct_of_ondemand", "mean"),
        )
        .reset_index()
        .sort_values("avg_flex_discount_usd_per_mwh", ascending=False)
    )
    return summary


def flexibility_discount_instance_data(
    df_power: pd.DataFrame,
    scenario: str = "inf_median",
    gpu_col: str = "gpu_model_clean",
    instance_col: str = "instance_type",
    min_mid_voll: Optional[float] = 500,
) -> pd.DataFrame:
    scenario = normalize_scenario(scenario)
    ondemand_col = voll_value_column(scenario, voll_basis="ondemand")
    spot_col = voll_value_column(scenario, voll_basis="spot")

    df_power = filter_low_mid_voll(
        df_power,
        min_mid_voll=min_mid_voll,
        value_prefix="voll_usd_per_mwh",
    )

    required_cols = [gpu_col, ondemand_col, spot_col]
    if instance_col in df_power.columns:
        required_cols.append(instance_col)
    missing = [col for col in required_cols if col not in df_power.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df_power[required_cols].copy()
    df[gpu_col] = df[gpu_col].fillna("Unknown").astype(str)
    df[ondemand_col] = pd.to_numeric(df[ondemand_col], errors="coerce")
    df[spot_col] = pd.to_numeric(df[spot_col], errors="coerce")
    df = df.dropna(subset=[gpu_col, ondemand_col, spot_col]).copy()
    df = df.loc[(df[ondemand_col] > 0) & (df[spot_col] > 0)].copy()
    if df.empty:
        raise ValueError(f"No data available for flexibility discount in scenario {scenario}.")

    df["flex_discount_usd_per_mwh"] = df[ondemand_col] - df[spot_col]
    df["flex_discount_pct_of_ondemand"] = (
        df["flex_discount_usd_per_mwh"] / df[ondemand_col] * 100
    )
    return df


def plot_flexibility_discount_by_gpu(
    df_power: pd.DataFrame,
    scenario: str = "inf_median",
    gpu_col: str = "gpu_model_clean",
    instance_col: str = "instance_type",
    min_mid_voll: Optional[float] = 500,
    save_png: bool = False,
    save_csv: bool = False,
    show: bool = True,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    style: PlotStyle = PlotStyle(),
) -> pd.DataFrame:
    scenario = normalize_scenario(scenario)
    summary = flexibility_discount_by_gpu(
        df_power=df_power,
        scenario=scenario,
        gpu_col=gpu_col,
        instance_col=instance_col,
        min_mid_voll=min_mid_voll,
    )

    fig, ax_bar = plt.subplots(figsize=(12.5, 6.8))
    ax_pct = ax_bar.twinx()
    x = np.arange(len(summary))
    labels = summary[gpu_col].tolist()

    bars = ax_bar.bar(
        x,
        summary["avg_flex_discount_usd_per_mwh"],
        color="tab:blue",
        edgecolor="black",
        linewidth=0.8,
        alpha=0.75,
        label="Discount ($/MWh)",
    )
    points = ax_pct.scatter(
        x,
        summary["avg_flex_discount_pct_of_ondemand"],
        color="black",
        edgecolor="black",
        s=46,
        marker="D",
        zorder=4,
        label="Discount (% of on-demand VOLL)",
    )

    ax_bar.set_ylabel("Discount ($/MWh)")
    ax_pct.set_ylabel("Discount (% of on-demand VoCL)")
    ax_bar.set_xlabel("GPU Model")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, rotation=45, ha="right")
    ax_bar.set_title(f"Average flexibility discount by GPU - {scenario}")
    tidy_axis(ax_bar, style)
    ax_pct.spines["top"].set_visible(False)

    ax_bar.legend(
        [bars, points],
        ["Discount ($/MWh)", "Discount (% of on-demand VOLL)"],
        frameon=False,
        loc="best",
    )

    fig.tight_layout()

    if save_png:
        save_current_figure(
            fig,
            f"flexibility_discount_by_gpu_{scenario}.png",
            output_dir=output_dir,
            style=style,
        )
    if save_csv:
        write_csv(
            summary,
            f"flexibility_discount_by_gpu_{scenario}.csv",
            output_dir=output_dir,
        )
    if show:
        plt.show()
    else:
        plt.close(fig)
    return summary


def plot_flexibility_discount_boxplot_by_gpu(
    df_power: pd.DataFrame,
    scenario: str = "inf_median",
    gpu_col: str = "gpu_model_clean",
    instance_col: str = "instance_type",
    min_mid_voll: Optional[float] = 500,
    excluded_vintage_groups: Sequence[str] = ("Legacy / Older Vintage",),
    save_png: bool = False,
    save_csv: bool = False,
    show: bool = True,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    style: PlotStyle = PlotStyle(),
) -> pd.DataFrame:
    scenario = normalize_scenario(scenario)
    df_plot = flexibility_discount_instance_data(
        df_power=df_power,
        scenario=scenario,
        gpu_col=gpu_col,
        instance_col=instance_col,
        min_mid_voll=min_mid_voll,
    )

    df_plot["gpu_vintage_group"] = df_plot[gpu_col].map(classify_gpu_vintage)
    if excluded_vintage_groups:
        df_plot = df_plot.loc[
            ~df_plot["gpu_vintage_group"].isin(excluded_vintage_groups)
        ].copy()
    if df_plot.empty:
        raise ValueError("No flexibility discount data available after vintage-group filtering.")

    medians = df_plot.groupby(gpu_col)["flex_discount_usd_per_mwh"].median()

    ordered_groups = []
    for vintage_group, gpu_order in GPU_VINTAGE_GROUPS.items():
        available = [gpu for gpu in gpu_order if gpu in set(df_plot[gpu_col])]
        ordered_groups.extend(available)

    remaining = [
        gpu for gpu in medians.sort_values().index.tolist()
        if gpu not in ordered_groups
    ]
    ordered_groups.extend(remaining)

    data = [
        df_plot.loc[df_plot[gpu_col] == group, "flex_discount_usd_per_mwh"].to_numpy()
        for group in ordered_groups
    ]

    fig_width = max(12.5, 0.82 * len(ordered_groups))
    fig, ax = plt.subplots(figsize=(fig_width, 6.8))
    box = ax.boxplot(
        data,
        labels=ordered_groups,
        patch_artist=True,
        showfliers=True,
        medianprops={"color": "black", "linewidth": 1.8},
        boxprops={"edgecolor": "black", "linewidth": 1.0},
        whiskerprops={"color": "black", "linewidth": 1.0},
        capprops={"color": "black", "linewidth": 1.0},
        flierprops={
            "marker": "o",
            "markersize": 4,
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "alpha": 0.7,
        },
    )
    for group, patch in zip(ordered_groups, box["boxes"]):
        vintage_group = classify_gpu_vintage(group)
        patch.set_facecolor(GPU_VINTAGE_PALETTE.get(vintage_group, "#9e9e9e"))
        patch.set_alpha(0.55)

    ymax = df_plot["flex_discount_usd_per_mwh"].max()
    y_label = ymax * 1.04
    group_boundaries = []
    group_centers = {}

    for vintage_group, gpu_order in GPU_VINTAGE_GROUPS.items():
        present = [gpu for gpu in gpu_order if gpu in ordered_groups]
        if not present:
            continue

        idxs = [ordered_groups.index(gpu) + 1 for gpu in present]
        group_centers[vintage_group] = np.mean(idxs)

        if max(idxs) < len(ordered_groups):
            group_boundaries.append(max(idxs) + 0.5)

    for boundary in group_boundaries:
        ax.axvline(
            boundary,
            color="black",
            linestyle="--",
            linewidth=1.0,
            alpha=0.45,
        )

    for vintage_group, center in group_centers.items():
        ax.text(
            center,
            y_label,
            GPU_VINTAGE_SHORT_LABELS.get(vintage_group, vintage_group),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=GPU_VINTAGE_PALETTE.get(vintage_group, "black"),
        )

    ax.set_xlabel("GPU Model")
    ax.set_ylabel("Flexibility discount ($/MWh)")
    ax.set_title(f"Flexibility discount distribution by GPU - {scenario}")
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(top=ymax * 1.16)
    tidy_axis(ax, style)

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=10,
            label=group,
        )
        for group, color in GPU_VINTAGE_PALETTE.items()
        if group in set(df_plot["gpu_vintage_group"])
    ]
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            frameon=True,
            loc="best",
            ncol=3,
        )

    fig.tight_layout()

    if save_png:
        save_current_figure(
            fig,
            f"flexibility_discount_boxplot_by_gpu_{scenario}.png",
            output_dir=output_dir,
            style=style,
        )
    if save_csv:
        write_csv(
            df_plot,
            f"flexibility_discount_boxplot_by_gpu_{scenario}.csv",
            output_dir=output_dir,
        )
    if show:
        plt.show()
    else:
        plt.close(fig)
    return df_plot


def flexibility_discount_boxplot_panel_data(
    df_power: pd.DataFrame,
    scenario: str,
    gpu_col: str = "gpu_model_clean",
    instance_col: str = "instance_type",
    min_mid_voll: Optional[float] = 500,
    excluded_vintage_groups: Sequence[str] = ("Legacy / Older Vintage",),
) -> Tuple[pd.DataFrame, List[str], List[np.ndarray]]:
    scenario = normalize_scenario(scenario)
    df_plot = flexibility_discount_instance_data(
        df_power=df_power,
        scenario=scenario,
        gpu_col=gpu_col,
        instance_col=instance_col,
        min_mid_voll=min_mid_voll,
    )
    df_plot["gpu_vintage_group"] = df_plot[gpu_col].map(classify_gpu_vintage)
    if excluded_vintage_groups:
        df_plot = df_plot.loc[
            ~df_plot["gpu_vintage_group"].isin(excluded_vintage_groups)
        ].copy()
    if df_plot.empty:
        raise ValueError("No flexibility discount data available after vintage-group filtering.")

    medians = df_plot.groupby(gpu_col)["flex_discount_usd_per_mwh"].median()
    ordered_groups = []
    present_gpus = set(df_plot[gpu_col])
    for _, gpu_order in GPU_VINTAGE_GROUPS.items():
        ordered_groups.extend(gpu for gpu in gpu_order if gpu in present_gpus)
    ordered_groups.extend(
        gpu for gpu in medians.sort_values().index.tolist()
        if gpu not in ordered_groups
    )
    data = [
        df_plot.loc[df_plot[gpu_col] == group, "flex_discount_usd_per_mwh"].to_numpy()
        for group in ordered_groups
    ]
    return df_plot, ordered_groups, data


def draw_flexibility_discount_boxplot_axis(
    ax: plt.Axes,
    df_plot: pd.DataFrame,
    ordered_groups: Sequence[str],
    data: Sequence[np.ndarray],
    title: str,
    show_ylabel: bool = True,
    y_top: Optional[float] = None,
    value_col: str = "flex_discount_usd_per_mwh",
    y_label_text: str = "Flexibility discount ($/MWh)",
    style: PlotStyle = PlotStyle(),
) -> None:
    box = ax.boxplot(
        data,
        labels=ordered_groups,
        patch_artist=True,
        showfliers=True,
        medianprops={"color": "black", "linewidth": 1.8},
        boxprops={"edgecolor": "black", "linewidth": 1.0},
        whiskerprops={"color": "black", "linewidth": 1.0},
        capprops={"color": "black", "linewidth": 1.0},
        flierprops={
            "marker": "o",
            "markersize": 4,
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "alpha": 0.7,
        },
    )
    for group, patch in zip(ordered_groups, box["boxes"]):
        vintage_group = classify_gpu_vintage(group)
        patch.set_facecolor(GPU_VINTAGE_PALETTE.get(vintage_group, "#9e9e9e"))
        patch.set_alpha(0.55)

    ymax = df_plot[value_col].max()
    y_label = (y_top if y_top is not None else ymax * 1.16) * 0.94
    group_boundaries = []
    group_centers = {}
    for vintage_group, gpu_order in GPU_VINTAGE_GROUPS.items():
        present = [gpu for gpu in gpu_order if gpu in ordered_groups]
        if not present:
            continue
        idxs = [ordered_groups.index(gpu) + 1 for gpu in present]
        group_centers[vintage_group] = np.mean(idxs)
        if max(idxs) < len(ordered_groups):
            group_boundaries.append(max(idxs) + 0.5)

    for boundary in group_boundaries:
        ax.axvline(
            boundary,
            color="black",
            linestyle="--",
            linewidth=1.0,
            alpha=0.45,
        )

    for vintage_group, center in group_centers.items():
        ax.text(
            center,
            y_label,
            GPU_VINTAGE_SHORT_LABELS.get(vintage_group, vintage_group),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=GPU_VINTAGE_PALETTE.get(vintage_group, "black"),
        )

    ax.set_xlabel("GPU Model")
    if show_ylabel:
        ax.set_ylabel(y_label_text)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    if y_top is None:
        ax.set_ylim(top=ymax * 1.16)
    else:
        ax.set_ylim(top=y_top)
    tidy_axis(ax, style)


def plot_flexibility_discount_boxplot_by_gpu_panel(
    df_power: pd.DataFrame,
    scenarios: Sequence[str] = ("inf_median", "trng_median"),
    gpu_col: str = "gpu_model_clean",
    instance_col: str = "instance_type",
    min_mid_voll: Optional[float] = 500,
    excluded_vintage_groups: Sequence[str] = ("Legacy / Older Vintage",),
    save_png: bool = False,
    save_csv: bool = False,
    show: bool = True,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    style: PlotStyle = PlotStyle(),
) -> pd.DataFrame:
    scenarios = [normalize_scenario(scenario) for scenario in scenarios]
    if len(scenarios) != 2:
        raise ValueError("Provide exactly two scenarios for the 1x2 flexibility discount panel.")

    panel_data = []
    for scenario in scenarios:
        df_plot, ordered_groups, data = flexibility_discount_boxplot_panel_data(
            df_power=df_power,
            scenario=scenario,
            gpu_col=gpu_col,
            instance_col=instance_col,
            min_mid_voll=min_mid_voll,
            excluded_vintage_groups=excluded_vintage_groups,
        )
        panel_data.append((scenario, df_plot, ordered_groups, data))

    y_top = max(df_plot["flex_discount_usd_per_mwh"].max() for _, df_plot, _, _ in panel_data) * 1.16
    fig_width = max(16, 0.86 * max(len(groups) for _, _, groups, _ in panel_data) * len(panel_data))
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, 6.8), sharey=True, squeeze=False)

    for idx, (scenario, df_plot, ordered_groups, data) in enumerate(panel_data):
        draw_flexibility_discount_boxplot_axis(
            ax=axes[0, idx],
            df_plot=df_plot,
            ordered_groups=ordered_groups,
            data=data,
            title=scenario,
            show_ylabel=idx == 0,
            y_top=y_top,
            style=style,
        )

    legend_groups = set(pd.concat([df_plot for _, df_plot, _, _ in panel_data])["gpu_vintage_group"])
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=10,
            label=group,
        )
        for group, color in GPU_VINTAGE_PALETTE.items()
        if group in legend_groups
    ]
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            frameon=True,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.08),
            ncol=min(3, len(legend_handles)),
            fontsize=9,
        )

    fig.suptitle("Flexibility discount distributions by GPU", y=0.99)
    fig.tight_layout(rect=(0, 0.12, 1, 0.93))

    df_out = pd.concat(
        [df_plot.assign(panel_scenario=scenario) for scenario, df_plot, _, _ in panel_data],
        ignore_index=True,
    )
    filename_base = f"flexibility_discount_boxplot_by_gpu_{'_'.join(scenarios)}_panel"
    if save_png:
        save_current_figure(fig, f"{filename_base}.png", output_dir=output_dir, style=style)
    if save_csv:
        write_csv(df_out, f"{filename_base}.csv", output_dir=output_dir)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return df_out


def forward_curve_discount_instance_data(
    df_power: pd.DataFrame,
    scenario: str = "inf_median",
    gpu_col: str = "gpu_model_clean",
    instance_col: str = "instance_type",
    min_mid_voll: Optional[float] = 500,
) -> pd.DataFrame:
    scenario = normalize_scenario(scenario)
    ondemand_col = voll_value_column(scenario, voll_basis="ondemand")
    reserved_col = voll_value_column(scenario, voll_basis="reserved")

    df_power = filter_low_mid_voll(
        df_power,
        min_mid_voll=min_mid_voll,
        value_prefix="voll_usd_per_mwh",
    )

    required_cols = [gpu_col, ondemand_col, reserved_col]
    if instance_col in df_power.columns:
        required_cols.append(instance_col)
    missing = [col for col in required_cols if col not in df_power.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df_power[required_cols].copy()
    df[gpu_col] = df[gpu_col].fillna("Unknown").astype(str)
    df[ondemand_col] = pd.to_numeric(df[ondemand_col], errors="coerce")
    df[reserved_col] = pd.to_numeric(df[reserved_col], errors="coerce")
    df = df.dropna(subset=[gpu_col, ondemand_col, reserved_col]).copy()
    df = df.loc[(df[ondemand_col] > 0) & (df[reserved_col] > 0)].copy()
    if df.empty:
        raise ValueError(f"No forward curve data available for scenario {scenario}.")

    df["forward_curve_discount_usd_per_mwh"] = df[ondemand_col] - df[reserved_col]
    df["forward_curve_discount_pct_of_ondemand"] = (
        df["forward_curve_discount_usd_per_mwh"] / df[ondemand_col] * 100
    )
    return df


def forward_curve_boxplot_panel_data(
    df_power: pd.DataFrame,
    scenario: str,
    gpu_col: str = "gpu_model_clean",
    instance_col: str = "instance_type",
    min_mid_voll: Optional[float] = 500,
    excluded_vintage_groups: Sequence[str] = ("Legacy / Older Vintage",),
) -> Tuple[pd.DataFrame, List[str], List[np.ndarray]]:
    scenario = normalize_scenario(scenario)
    df_plot = forward_curve_discount_instance_data(
        df_power=df_power,
        scenario=scenario,
        gpu_col=gpu_col,
        instance_col=instance_col,
        min_mid_voll=min_mid_voll,
    )
    df_plot["gpu_vintage_group"] = df_plot[gpu_col].map(classify_gpu_vintage)
    if excluded_vintage_groups:
        df_plot = df_plot.loc[
            ~df_plot["gpu_vintage_group"].isin(excluded_vintage_groups)
        ].copy()
    if df_plot.empty:
        raise ValueError("No forward curve data available after vintage-group filtering.")

    value_col = "forward_curve_discount_usd_per_mwh"
    medians = df_plot.groupby(gpu_col)[value_col].median()
    ordered_groups = []
    present_gpus = set(df_plot[gpu_col])
    for _, gpu_order in GPU_VINTAGE_GROUPS.items():
        ordered_groups.extend(gpu for gpu in gpu_order if gpu in present_gpus)
    ordered_groups.extend(
        gpu for gpu in medians.sort_values().index.tolist()
        if gpu not in ordered_groups
    )
    data = [
        df_plot.loc[df_plot[gpu_col] == group, value_col].to_numpy()
        for group in ordered_groups
    ]
    return df_plot, ordered_groups, data


def plot_forward_curve_boxplot_by_gpu_panel(
    df_power: pd.DataFrame,
    scenarios: Sequence[str] = ("inf_median", "trng_median"),
    gpu_col: str = "gpu_model_clean",
    instance_col: str = "instance_type",
    min_mid_voll: Optional[float] = 500,
    excluded_vintage_groups: Sequence[str] = ("Legacy / Older Vintage",),
    save_png: bool = False,
    save_csv: bool = False,
    show: bool = True,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    style: PlotStyle = PlotStyle(),
) -> pd.DataFrame:
    scenarios = [normalize_scenario(scenario) for scenario in scenarios]
    if not scenarios:
        raise ValueError("Provide at least one scenario for the forward curve panel.")

    panel_data = []
    for scenario in scenarios:
        df_plot, ordered_groups, data = forward_curve_boxplot_panel_data(
            df_power=df_power,
            scenario=scenario,
            gpu_col=gpu_col,
            instance_col=instance_col,
            min_mid_voll=min_mid_voll,
            excluded_vintage_groups=excluded_vintage_groups,
        )
        panel_data.append((scenario, df_plot, ordered_groups, data))

    value_col = "forward_curve_discount_usd_per_mwh"
    y_top = max(df_plot[value_col].max() for _, df_plot, _, _ in panel_data) * 1.16
    panel_count = len(panel_data)
    fig_width = max(10, 0.86 * max(len(groups) for _, _, groups, _ in panel_data) * panel_count)
    fig, axes = plt.subplots(1, panel_count, figsize=(fig_width, 6.8), sharey=True, squeeze=False)

    for idx, (scenario, df_plot, ordered_groups, data) in enumerate(panel_data):
        draw_flexibility_discount_boxplot_axis(
            ax=axes[0, idx],
            df_plot=df_plot,
            ordered_groups=ordered_groups,
            data=data,
            title=scenario,
            show_ylabel=idx == 0,
            y_top=y_top,
            value_col=value_col,
            y_label_text="On-demand minus reserved VoCL ($/MWh)",
            style=style,
        )

    legend_groups = set(pd.concat([df_plot for _, df_plot, _, _ in panel_data])["gpu_vintage_group"])
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=10,
            label=group,
        )
        for group, color in GPU_VINTAGE_PALETTE.items()
        if group in legend_groups
    ]
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            frameon=True,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.08),
            ncol=min(3, len(legend_handles)),
            fontsize=9,
        )

    fig.suptitle("Contracting discount distribution: on-demand minus 1-year reserved VOLL", y=0.99)
    fig.tight_layout(rect=(0, 0.12, 1, 0.93))

    df_out = pd.concat(
        [df_plot.assign(panel_scenario=scenario) for scenario, df_plot, _, _ in panel_data],
        ignore_index=True,
    )
    filename_base = f"forward_curve_boxplot_by_gpu_{'_'.join(scenarios)}_panel"
    if save_png:
        save_current_figure(fig, f"{filename_base}.png", output_dir=output_dir, style=style)
    if save_csv:
        write_csv(df_out, f"{filename_base}.csv", output_dir=output_dir)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return df_out


def canonical_gpu_instances(
    df_power: pd.DataFrame,
    scenario: str = "trng_median",
    gpu_col: str = "gpu_model_clean",
    instance_col: str = "instance_type",
    gpu_vintage_groups: Sequence[str] = (
        "Mainstream Modern AI",
        "Emerging / Frontier AI",
    ),
    min_mid_voll: Optional[float] = 500,
) -> pd.DataFrame:
    scenario = normalize_scenario(scenario)
    voll_col = voll_value_column(scenario, voll_basis="ondemand")
    required_cols = [instance_col, gpu_col, voll_col]
    missing = [col for col in required_cols if col not in df_power.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = filter_low_mid_voll(
        df_power,
        min_mid_voll=min_mid_voll,
        value_prefix="voll_usd_per_mwh",
    )
    df = df[required_cols].copy()
    df[gpu_col] = df[gpu_col].fillna("Unknown").astype(str).str.strip()
    df[voll_col] = pd.to_numeric(df[voll_col], errors="coerce")
    df = df.dropna(subset=[gpu_col, instance_col, voll_col]).copy()
    df = df.loc[(df[gpu_col] != "") & (df[voll_col] > 0)].copy()
    df["gpu_vintage_group"] = df[gpu_col].map(classify_gpu_vintage)
    df = df.loc[df["gpu_vintage_group"].isin(gpu_vintage_groups)].copy()
    if df.empty:
        raise ValueError("No mainstream or emerging GPU instances available for canonical selection.")

    selected_rows = []
    for _, group in df.groupby(gpu_col, sort=False):
        median_voll = group[voll_col].median()
        best_idx = (group[voll_col] - median_voll).abs().sort_values().index[0]
        selected = group.loc[best_idx].copy()
        selected["canonical_basis_scenario"] = scenario
        selected["gpu_class_median_voll_usd_per_mwh"] = median_voll
        selected["distance_to_gpu_class_median"] = abs(selected[voll_col] - median_voll)
        selected_rows.append(selected)

    canonical = pd.DataFrame(selected_rows)
    canonical = canonical.rename(columns={voll_col: "base_ondemand_voll_usd_per_mwh"})
    canonical["gpu_vintage_order"] = canonical["gpu_vintage_group"].map(
        {group: idx for idx, group in enumerate(gpu_vintage_groups)}
    )
    return canonical.sort_values(
        ["gpu_vintage_order", gpu_col, instance_col],
    ).drop(columns=["gpu_vintage_order"]).reset_index(drop=True)


def plot_sla_sensitivity_canonical_gpu_panel(
    df_power: pd.DataFrame,
    scenario: str = "trng_median",
    h_values: Optional[Sequence[float]] = None,
    u_values: Optional[Sequence[float]] = None,
    h_cumulative: float = 4.0,
    u_mth: float = U_MTH,
    cloud_platform: str = SLA_CLOUD_PLATFORM,
    hours_per_month: float = HOURS_PER_MONTH,
    gpu_col: str = "gpu_model_clean",
    instance_col: str = "instance_type",
    min_mid_voll: Optional[float] = 500,
    title: Optional[str] = None,
    save_png: bool = False,
    save_csv: bool = False,
    show: bool = True,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    style: PlotStyle = PlotStyle(),
) -> pd.DataFrame:
    scenario = normalize_scenario(scenario)
    if h_values is None:
        h_values = np.linspace(0, 20, 21)
    if u_values is None:
        u_values = np.linspace(0, 0.3, 31)

    h_values_array = np.asarray(h_values, dtype=float)
    u_values_array = np.asarray(u_values, dtype=float)
    canonical = canonical_gpu_instances(
        df_power=df_power,
        scenario=scenario,
        gpu_col=gpu_col,
        instance_col=instance_col,
        min_mid_voll=min_mid_voll,
    )

    h_credit_shares = np.array(
        [
            relevant_service_credit_share(
                h_cumulative=float(h_value),
                cloud_platform=cloud_platform,
            )
            for h_value in h_values_array
        ]
    )
    fixed_h_credit_share = relevant_service_credit_share(
        h_cumulative=h_cumulative-1,
        cloud_platform=cloud_platform,
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4), sharey=True)
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(canonical), 1)))
    records = []

    for color, (_, row) in zip(colors, canonical.iterrows()):
        base_voll = float(row["base_ondemand_voll_usd_per_mwh"])
        label = f"{row[gpu_col]} ({row[instance_col]})"
        line_style = "-" if row["gpu_vintage_group"] == "Mainstream Modern AI" else "--"
 
        h_panel_voll = base_voll * (1 + h_credit_shares * hours_per_month * u_mth )
        axes[0].plot(
            h_values_array,
            h_panel_voll,
            color=color,
            linestyle=line_style,
            linewidth=1.8,
            drawstyle="steps-post",
            label=label,
        )

        u_panel_voll = base_voll * (1+ fixed_h_credit_share * hours_per_month * u_values_array)
        axes[1].plot(
            u_values_array,
            u_panel_voll,
            color=color,
            linestyle=line_style,
            linewidth=1.8,
            label=label,
        )

        for h_value, voll_value, credit_share in zip(h_values_array, h_panel_voll, h_credit_shares):
            records.append(
                {
                    "panel": "H_CUMULATIVE",
                    instance_col: row[instance_col],
                    gpu_col: row[gpu_col],
                    "gpu_vintage_group": row["gpu_vintage_group"],
                    "canonical_basis_scenario": scenario,
                    "base_ondemand_voll_usd_per_mwh": base_voll,
                    "gpu_class_median_voll_usd_per_mwh": row["gpu_class_median_voll_usd_per_mwh"],
                    "distance_to_gpu_class_median": row["distance_to_gpu_class_median"],
                    "H_CUMULATIVE": h_value,
                    "U_mth": u_mth,
                    "service_credit_share": credit_share,
                    "voll_ondemand_incl_sla_usd_per_mwh": voll_value,
                }
            )
        for u_value, voll_value in zip(u_values_array, u_panel_voll):
            records.append(
                {
                    "panel": "U_mth",
                    instance_col: row[instance_col],
                    gpu_col: row[gpu_col],
                    "gpu_vintage_group": row["gpu_vintage_group"],
                    "canonical_basis_scenario": scenario,
                    "base_ondemand_voll_usd_per_mwh": base_voll,
                    "gpu_class_median_voll_usd_per_mwh": row["gpu_class_median_voll_usd_per_mwh"],
                    "distance_to_gpu_class_median": row["distance_to_gpu_class_median"],
                    "H_CUMULATIVE": h_cumulative,
                    "U_mth": u_value,
                    "service_credit_share": fixed_h_credit_share,
                    "voll_ondemand_incl_sla_usd_per_mwh": voll_value,
                }
            )

    h_axis_xlim = (0, 20)
    axes[0].set_xlim(h_axis_xlim)
    axes[0].xaxis.set_major_locator(mticker.MultipleLocator(2.0))
    for threshold_hrs, service_credit in SERVICE_CREDIT_BANDS[cloud_platform]:
        if threshold_hrs < h_axis_xlim[0] or threshold_hrs > h_axis_xlim[1]:
            continue
        axes[0].axvline(
            threshold_hrs,
            color="black",
            linestyle=":",
            linewidth=0.8,
            alpha=0.35,
        )
        axes[0].text(
            threshold_hrs,
            0.98,
            f"{service_credit:.0%}",
            transform=axes[0].get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
        )

    axes[0].set_title("SLA sensitivity to cumulative downtime")
    axes[0].set_xlabel("Cumulative downtime hours in month")
    axes[0].set_ylabel("On-demand VoCL incl. SLA ($/MWh)")
    axes[1].set_title("SLA sensitivity to monthly utilization")
    axes[1].set_xlabel("Monthly utilization")
    axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{value:.0%}"))
    y_formatter = mticker.FuncFormatter(lambda value, _: f"{value:,.0f}" if value >= 1 else f"{value:g}")
    for ax in axes:
        ax.yaxis.set_major_formatter(y_formatter)
        tidy_axis(ax, style)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        fontsize=8,
        frameon=True,
    )
    fig.suptitle(
        title or f"Canonical mainstream and emerging GPU SLA-inclusive on-demand VoCL ({scenario})",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))

    df_plot = pd.DataFrame(records)
    filename_base = f"voll_sla_sensitivity_canonical_gpu_{scenario}"
    if save_png:
        save_current_figure(fig, f"{filename_base}.png", output_dir, style)
    if save_csv:
        write_csv(df_plot, f"{filename_base}.csv", output_dir=output_dir)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return df_plot


def plot_sla_sensitivity_sample_voll_cloud_panel(
    sample_voll_values: Sequence[float] = (5000, 10000),
    cloud_platforms: Sequence[str] = ("aws_ec2_instance", "gcp", "azure_virtual_machines"),
    scenario_label: str = "trng_median",
    h_values: Optional[Sequence[float]] = None,
    u_mth: float = U_MTH,
    hours_per_month: float = HOURS_PER_MONTH,
    plot_x_jitter_hours: float = 0.04,
    title: Optional[str] = None,
    save_png: bool = False,
    save_csv: bool = False,
    show: bool = True,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    style: PlotStyle = PlotStyle(),
) -> pd.DataFrame:
    if h_values is None:
        h_values = np.linspace(0, 20, 21)

    h_values_array = np.asarray(h_values, dtype=float)
    sample_voll_array = np.asarray(sample_voll_values, dtype=float)
    cloud_platforms = [str(platform) for platform in cloud_platforms]

    missing_platforms = [platform for platform in cloud_platforms if platform not in SERVICE_CREDIT_BANDS]
    if missing_platforms:
        raise ValueError(f"Unknown SLA cloud platform(s): {missing_platforms}")

    fig, ax = plt.subplots(figsize=(10.8, 6.4))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(cloud_platforms), 1)))
    line_styles = ["-", "--", ":", "-."]
    records = []
    jitter_center = (len(cloud_platforms) - 1) / 2

    for platform_idx, (cloud_platform, color) in enumerate(zip(cloud_platforms, colors)):
        platform_label = SLA_CLOUD_PLATFORM_LABELS.get(cloud_platform, cloud_platform)
        h_credit_shares = np.array(
            [
                relevant_service_credit_share(
                    h_cumulative=float(h_value),
                    cloud_platform=cloud_platform,
                )
                for h_value in h_values_array
            ]
        )
        for voll_idx, base_voll in enumerate(sample_voll_array):
            line_style = line_styles[voll_idx % len(line_styles)]
            label = f"{platform_label}, ${base_voll:,.0f}/MWh"

            h_panel_voll = base_voll * (1 + h_credit_shares * hours_per_month * u_mth)
            plot_x_jitter = (platform_idx - jitter_center) * plot_x_jitter_hours
            ax.plot(
                h_values_array + plot_x_jitter,
                h_panel_voll,
                color=color,
                linestyle=line_style,
                linewidth=2.0,
                drawstyle="steps-post",
                label=label,
            )

            for h_value, voll_value, credit_share in zip(h_values_array, h_panel_voll, h_credit_shares):
                records.append(
                    {
                        "panel": "H_CUMULATIVE",
                        "cloud_platform": cloud_platform,
                        "cloud_platform_label": platform_label,
                        "base_voll_usd_per_mwh": base_voll,
                        "scenario_label": scenario_label,
                        "H_CUMULATIVE": h_value,
                        "U_mth": u_mth,
                        "service_credit_share": credit_share,
                        "plot_x_jitter_hours": plot_x_jitter,
                        "voll_incl_sla_usd_per_mwh": voll_value,
                    }
                )

    h_axis_xlim = (0, 20)
    ax.set_xlim(h_axis_xlim)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2.0))
    for cloud_platform, color in zip(cloud_platforms, colors):
        for threshold_hrs, service_credit in SERVICE_CREDIT_BANDS[cloud_platform]:
            if threshold_hrs < h_axis_xlim[0] or threshold_hrs > h_axis_xlim[1]:
                continue
            ax.axvline(
                threshold_hrs,
                color=color,
                linestyle=":",
                linewidth=0.9,
                alpha=0.35,
            )
            ax.text(
                threshold_hrs,
                0.98 - 0.055 * cloud_platforms.index(cloud_platform),
                f"{SLA_CLOUD_PLATFORM_LABELS.get(cloud_platform, cloud_platform)} {service_credit:.0%}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8,
                color=color,
            )

    ax.set_title("SLA sensitivity to cumulative downtine")
    ax.set_xlabel("Cumulative downtime hours in month")
    ax.set_ylabel("VoCL incl. SLA ($/MWh)")
    y_formatter = mticker.FuncFormatter(lambda value, _: f"{value:,.0f}" if value >= 1 else f"{value:g}")
    ax.yaxis.set_major_formatter(y_formatter)
    tidy_axis(ax, style)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        fontsize=8,
        frameon=True,
    )
    fig.suptitle(
        title or "Sample VoCL SLA-inclusive sensitivity by cloud provider",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0.14, 1, 0.94))

    df_plot = pd.DataFrame(records)
    safe_scenario = str(scenario_label).replace(" ", "_")
    filename_base = f"voll_sla_sensitivity_sample_voll_cloud_providers_{safe_scenario}"
    if save_png:
        save_current_figure(fig, f"{filename_base}.png", output_dir, style)
    if save_csv:
        write_csv(df_plot, f"{filename_base}.csv", output_dir=output_dir)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return df_plot


# ============================================================
# Convenience workflow
# ============================================================
def build_voll_timeseries_for_instance(
    instance_name: str,
    provider: str = "aws",
    region: Optional[str] = None,
    power_file: Path = DEFAULT_POWER_FILE,
    price_file: Optional[Path] = None,
    prices_dir: Path = ORIGINAL_PRICES_DIR,
    scenarios: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    df_power = load_power_data(power_file)
    df_price = load_price_data(
        filepath=price_file,
        instance_name=instance_name,
        region=region,
        provider=provider,
        prices_dir=prices_dir,
    )
    return compute_voll_timeseries(
        df_price=df_price,
        df_power=df_power,
        instance_name=instance_name,
        provider=provider,
        scenarios=scenarios,
    )


def run_example() -> None:
    instance_name = "p4d.24xlarge"
    b300_instance_name = "p6-b300.48xlarge"
    b300_region = "us-west-2"
    provider = "aws"
    region = "us-west-2"
    locations = ["us-east-1", "us-east-2", "us-west-2"]
    timeseries_scenario = "trng_median"
    suite_scenario = "inf_median"

    df_power = load_power_data()
    df_price = load_price_data(
        instance_name=instance_name,
        region=region,
        provider=provider,
    )
    df_voll = compute_voll_timeseries(
        df_price=df_price,
        df_power=df_power,
        instance_name=instance_name,
        provider=provider,
    )

    preview_cols = [
        "timestamp",
        "Instance Price",
        f"VOLL_{timeseries_scenario}",
        f"voll_ondemand_{timeseries_scenario}",
    ]
    preview_cols = [col for col in preview_cols if col in df_voll.columns]
    print("\nPreview:")
    print(df_voll[preview_cols].head())

    plot_voll_timeseries(
        df_voll=df_voll,
        scenario=timeseries_scenario,
        instance_name=instance_name,
        provider=provider,
        location=region,
        ymin=0,
        save_png=True,
        save_csv=True,
        show=False,
    )

    b300_price = load_price_data(
        instance_name=b300_instance_name,
        region=b300_region,
        provider=provider,
    )
    b300_voll = compute_voll_timeseries(
        df_price=b300_price,
        df_power=df_power,
        instance_name=b300_instance_name,
        provider=provider,
        scenarios=[timeseries_scenario],
    )
    plot_voll_timeseries(
        df_voll=b300_voll,
        scenario=timeseries_scenario,
        instance_name=b300_instance_name,
        provider=provider,
        location=b300_region,
        ymin=0,
        save_png=True,
        save_csv=True,
        show=False,
    )

    plot_voll_timeseries_locations(
        instance_name=instance_name,
        locations=locations,
        scenario=timeseries_scenario,
        provider=provider,
        df_power=df_power,
        ymin=0,
        save_png=True,
        save_csv=True,
        show=False,
    )

    plot_voll_spot_timeseries_clouds(
        df_power=df_power,
        scenario=timeseries_scenario,
        aws_instance_name="p4d.24xlarge",
        aws_region="us-east-1",
        azure_instance_name="Standard_ND96amsr_A100_v4",
        azure_region="eastus",
        azure_ondemand_price=32.77,
        start_date="2024-01-25",
        save_png=True,
        save_csv=True,
        show=False,
    )

    plot_voll_ondemand_regions(
        df_power=df_power,
        scenario=timeseries_scenario,
        instance_name="p4d.24xlarge",
        provider=provider,
        base_region="us-east-1",
        save_png=True,
        save_csv=True,
        show=False,
    )

    for voll_basis in ["ondemand", "spot", "reserved"]:
        plot_voll_violin_by_gpu(
            df_power=df_power,
            scenario=suite_scenario,
            voll_basis=voll_basis,
            save_png=True,
            save_csv=True,
            show=False,
        )

    plot_voll_violin_basis_scenario_panel(
        df_power=df_power,
        scenarios=("trng_median", "inf_median"),
        save_png=True,
        save_csv=True,
        show=False,
    )

    plot_voll_boxplot_basis_scenario_panel(
        df_power=df_power,
        scenarios=("trng_median", "inf_median"),
        save_png=True,
        save_csv=True,
        show=False,
    )

    plot_voll_boxplot_sla_spot_scenario_panel(
        df_power=df_power,
        scenarios=("trng_median", "inf_median"),
        save_png=True,
        save_csv=True,
        show=False,
    )

    plot_voll_boxplot_sla_spot_scenario_panel(
        df_power=df_power,
        scenarios=("trng_median", "inf_median"),
        save_png=True,
        save_csv=True,
        show=False,
        include_vintage_titles=True,
    )

    plot_voll_boxplot_sla_spot_scenario_panel(
        df_power=df_power,
        scenarios=("trng_median", "inf_median"),
        save_png=True,
        save_csv=True,
        show=False,
        include_vintage_titles=True,
        include_sla_rows=False,
    )

    plot_voll_boxplot_sla_spot_scenario_panel(
        df_power=df_power,
        scenarios=("trng_median", "inf_median"),
        h_cumulative_values=(4.0,),
        save_png=True,
        save_csv=True,
        show=False,
        include_vintage_titles=True,
        include_ondemand_row=False,
        include_spot_row=False,
    )

    plot_sla_sensitivity_canonical_gpu_panel(
        df_power=df_power,
        scenario="trng_median",
        save_png=True,
        save_csv=True,
        show=False,
    )

    plot_sla_sensitivity_sample_voll_cloud_panel(
        scenario_label="trng_median",
        save_png=True,
        save_csv=True,
        show=False,
    )

    plot_voll_histogram(
        df_power=df_power,
        scenario=suite_scenario,
        value_prefix="voll_usd_per_mwh",
        save_png=True,
        show=False,
    )

    try:
        plot_voll_density(
            df_power=df_power,
            scenario=suite_scenario,
            value_prefix="voll_usd_per_mwh",
            save_png=True,
            show=False,
        )
    except ImportError as exc:
        print(f"Skipped density plot: {exc}")

    plot_voll_range_by_instance(
        df_power=df_power,
        low_scenario="lo",
        mid_scenario="mid",
        high_scenario="hi",
        value_prefix="voll_usd_per_mwh",
        marker_scenarios={
            "inf_median": "Inference median",
            "trng_median": "Training median",
        },
        range_label="p10-p90 range",
        show_instance_labels=True,
        save_png=True,
        show=False,
    )

    plot_spot_interrupt_vs_voll(
        df_power=df_power,
        scenario="trng_median",
        save_png=True,
        save_csv=True,
        show=False,
    )

    plot_flexibility_discount_by_gpu(
        df_power=df_power,
        scenario=suite_scenario,
        save_png=True,
        save_csv=True,
        show=False,
    )

    plot_flexibility_discount_boxplot_by_gpu(
        df_power=df_power,
        scenario=suite_scenario,
        save_png=True,
        save_csv=True,
        show=False,
    )

    plot_flexibility_discount_boxplot_by_gpu(
        df_power=df_power,
        scenario=timeseries_scenario,
        save_png=True,
        save_csv=True,
        show=False,
    )

    plot_flexibility_discount_boxplot_by_gpu_panel(
        df_power=df_power,
        scenarios=("inf_median", "trng_median"),
        save_png=True,
        save_csv=True,
        show=False,
    )

    plot_forward_curve_boxplot_by_gpu_panel(
        df_power=df_power,
        scenarios=("inf_median", "trng_median"),
        save_png=True,
        save_csv=True,
        show=False,
    )

    plot_forward_curve_boxplot_by_gpu_panel(
        df_power=df_power,
        scenarios=("trng_median",),
        save_png=True,
        save_csv=True,
        show=False,
    )

    print(f"\nSaved graph suite to: {DEFAULT_OUTPUT_DIR}")


if __name__ == "__main__":
    run_example()
