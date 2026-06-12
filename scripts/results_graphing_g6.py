# Run in terminal
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# .\.ai_dem\Scripts\Activate.ps1

from pathlib import Path
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
#import matplotlib


# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(
    r"C:\Users\FarhadBillimoria\OneDrive - Aurora Energy Research\Documents\Personal\Endog_demand_AI\pyai"
)

POWER_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "instances"
    / "aws_gpu_cpu_ram_ssd_power_and_value_wide_expanded.csv"
)

PRICE_FILE = (
    BASE_DIR
    / "data"
    / "original"
    / "prices"
    / "B300"
    / "EC2_p6b30048xlarge_us-west-2.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed" / "instances" / "graphs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Scenario setup
# ============================================================
SCENARIOS: List[str] = ["idle", "lo", "mid", "hi", "custom"]


# ============================================================
# Provider configuration
# This gives you infrastructure for AWS/GCP/Azure later.
# ============================================================
PROVIDER_CONFIG: Dict[str, Dict[str, object]] = {
    "aws": {
        "instance_col_candidates": ["instance_type", "API Name", "Instance type", "Instance Type", "Name"],
        "price_datetime_col": "Modified Date Time",
        "price_value_col": "Instance Price",
        "region_col": "Region",
        "availability_zone_col": "Availability Zone",
    },
    "gcp": {
        "instance_col_candidates": ["instance_type", "Machine type", "Machine Type", "Name"],
        "price_datetime_col": None,   # fill in later
        "price_value_col": None,      # fill in later
        "region_col": None,
        "availability_zone_col": None,
    },
    "azure": {
        "instance_col_candidates": ["instance_type", "VM Size", "VM size", "Name"],
        "price_datetime_col": None,   # fill in later
        "price_value_col": None,      # fill in later
        "region_col": None,
        "availability_zone_col": None,
    },
}


# ============================================================
# Utilities
# ============================================================
def find_first_existing_column(df, candidates, required=True):
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise ValueError(f"Could not find any of these columns: {candidates}")
    return None


def load_power_data(filepath: Path = POWER_FILE) -> pd.DataFrame:
    if not filepath.exists():
        raise FileNotFoundError(f"Power file not found: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Loaded power data: {filepath}")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    return df


def load_price_data(filepath: Path = PRICE_FILE) -> pd.DataFrame:
    if not filepath.exists():
        raise FileNotFoundError(f"Price file not found: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Loaded price data: {filepath}")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    return df

def compute_voll(
    df_price: pd.DataFrame,
    df_power: pd.DataFrame,
    instance_name: str,
    provider: str = "aws",
) -> pd.DataFrame:
    """
    Step 1:
      Select the row in df_power for the chosen instance_name.

    Step 2:
      Create scenario-specific spot VOLL columns in df_price:
          VOLL_<scenario> = Instance Price / (total_power_w_<scenario> / 1_000_000)

      Also copy the precomputed on-demand implied VOLL values from df_power:
          voll_ondemand_<scenario>

    Returns:
      df_price with timestamp/date fields and VOLL columns added.
    """
    if provider not in PROVIDER_CONFIG:
        raise ValueError(f"Unsupported provider: {provider}")

    cfg = PROVIDER_CONFIG[provider]

    # ----------------------------
    # Identify instance column in power dataframe
    # ----------------------------
    instance_col = find_first_existing_column(
        df_power,
        cfg["instance_col_candidates"],
        required=True,
    )

    # ----------------------------
    # Step 1: select exact instance row
    # ----------------------------
    power_match = df_power.loc[
        df_power[instance_col].astype(str).str.strip() == str(instance_name).strip()
    ].copy()

    if power_match.empty:
        available = df_power[instance_col].dropna().astype(str).head(20).tolist()
        raise ValueError(
            f"No matching instance '{instance_name}' found in df_power. "
            f"Sample available values: {available}"
        )

    if len(power_match) > 1:
        print(f"Warning: multiple rows found for {instance_name}. Using the first row only.")
        power_match = power_match.iloc[[0]]

    power_row = power_match.iloc[0]

    # ----------------------------
    # Parse df_price for current provider
    # ----------------------------
    price_dt_col = cfg["price_datetime_col"]
    price_val_col = cfg["price_value_col"]

    if price_dt_col is None or price_val_col is None:
        raise ValueError(f"Provider '{provider}' price-format config is not filled in yet.")

    if price_dt_col not in df_price.columns:
        raise ValueError(f"Datetime column '{price_dt_col}' not found in df_price")
    if price_val_col not in df_price.columns:
        raise ValueError(f"Price column '{price_val_col}' not found in df_price")

    out = df_price.copy()

    out["timestamp"] = pd.to_datetime(out[price_dt_col], errors="coerce")
    out[price_val_col] = pd.to_numeric(out[price_val_col], errors="coerce")

    out = out.dropna(subset=["timestamp", price_val_col]).copy()
    out["date"] = out["timestamp"].dt.floor("D")
    out["instance_name"] = instance_name
    out["provider"] = provider

    # ----------------------------
    # Map scenarios to precomputed on-demand VOLL columns in df_power
    # ----------------------------
    ondemand_voll_map = {
        "lo": "voll_usd_per_mwh_lo",
        "mid": "voll_usd_per_mwh_mid",
        "hi": "voll_usd_per_mwh_hi",
        "custom": "voll_usd_per_mwh_custom",
    }

    # Optional: if you also have idle in df_power, add it here.
    # Otherwise idle will simply not get an on-demand line.
    if "voll_usd_per_mwh_idle" in df_power.columns:
        ondemand_voll_map["idle"] = "voll_usd_per_mwh_idle"

    # ----------------------------
    # Step 2: create VOLL columns
    # ----------------------------
    for scenario in SCENARIOS:
        power_col = f"total_power_w_{scenario}"

        if power_col not in df_power.columns:
            raise ValueError(f"Expected power column not found in df_power: {power_col}")

        power_w = pd.to_numeric(power_row[power_col], errors="coerce")
        if pd.isna(power_w) or power_w <= 0:
            raise ValueError(
                f"Invalid power value for '{instance_name}' in '{power_col}': {power_w}"
            )

        power_mw = power_w / 1_000_000.0

        # Spot-implied VOLL from time-varying price data
        out[f"VOLL_{scenario}"] = out[price_val_col] / power_mw

        # On-demand implied VOLL from precomputed df_power columns
        ondemand_col = ondemand_voll_map.get(scenario)
        if ondemand_col is not None:
            if ondemand_col not in df_power.columns:
                raise ValueError(
                    f"Expected on-demand VOLL column not found in df_power: {ondemand_col}"
                )
            out[f"voll_ondemand_{scenario}"] = pd.to_numeric(
                power_row[ondemand_col],
                errors="coerce",
            )

    print("\nCreated VOLL columns:")
    print([c for c in out.columns if c.startswith("VOLL_")])

    print("\nCreated on-demand reference columns:")
    print([c for c in out.columns if c.startswith("voll_ondemand_")])

    return out



def plot_voll_bounds(
    df_power,
    ymin=None,
    ymax=None,
    save_png=True,
    save_csv=True,
    ):

    df_power_sorted = df_power.sort_values(by="voll_usd_per_mwh_mid", ascending=True)

    voll_col = f"VOLL_{scenario}"
    if voll_col not in df_voll.columns:
        raise ValueError(f"Column not found: {voll_col}")

    if "date" not in df_voll.columns:
        raise ValueError("Column 'date' not found in df_voll")

    daily = (
        df_voll.groupby("date")[voll_col]
        .agg(voll_mean="mean", voll_min="min", voll_max="max", obs_count="count")
        .reset_index()
        .sort_values("date")
    )

    if daily.empty:
        raise ValueError("No daily data available after aggregation.")

    # Scenario-specific on-demand VOLL reference
    ondemand_col = f"voll_ondemand_{scenario}"
    on_demand_voll_value = None
    if ondemand_col in df_voll.columns:
        tmp = pd.to_numeric(df_voll[ondemand_col], errors="coerce").dropna()
        if not tmp.empty:
            on_demand_voll_value = tmp.iloc[0]

    fig, ax = plt.subplots(figsize=(12.5, 6.8))

    ax.fill_between(
        daily["date"],
        daily["voll_min"],
        daily["voll_max"],
        alpha=0.22,
        label="Spot VOLL - Min-max envelope",
    )

    ax.plot(
        daily["date"],
        daily["voll_mean"],
        linewidth=2.4,
        label=f"Spot VOLL - Ave ({scenario})",
    )

    if on_demand_voll_value is not None:
        ax.axhline(
            y=on_demand_voll_value,
            color="black",
            linewidth=2.2,
            linestyle="-",
            label=f"On-Demand Implied VOLL ({scenario})",
        )

    ax.axhline(
        y=2000,
        linestyle="--",
        linewidth=1.8,
        color="tab:green",
        label="PJM Offer Cap",
    )

    ax.axhline(
        y=3700,
        linestyle="--",
        linewidth=1.8,
        color="tab:orange",
        label="PJM System Price Cap",
    )

    ax.set_title(
        f"Implied VOLL for {instance_name} ({provider.upper()})\nScenario: {scenario}",
        fontsize=14,
        pad=14,
    )
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("VOLL ($/MWh)", fontsize=12)

    ax.grid(True, alpha=0.30, linewidth=0.8)
    ax.legend(frameon=False, fontsize=10, loc="upper right")

    if ymin is not None or ymax is not None:
        ax.set_ylim(bottom=ymin, top=ymax)

    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.subplots_adjust(top=0.88, bottom=0.16, right=0.97, left=0.10)

    if save_png:
        png_file = OUTPUT_DIR / f"voll_timeseries_{provider}_{instance_name.replace('.', '_')}_{scenario}.png"
        plt.savefig(png_file, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {png_file}")

    if save_csv:
        csv_file = OUTPUT_DIR / f"voll_daily_{provider}_{instance_name.replace('.', '_')}_{scenario}.csv"
        daily.to_csv(csv_file, index=False)
        print(f"Saved daily summary: {csv_file}")

    plt.show()
    return daily


def plot_voll_timeseries(
    df_voll,
    scenario,
    instance_name,
    provider="aws",
    location="us-east-1",
    ymin=None,
    ymax=None,
    save_png=True,
    save_csv=True,
):
    voll_col = f"VOLL_{scenario}"
    if voll_col not in df_voll.columns:
        raise ValueError(f"Column not found: {voll_col}")

    if "date" not in df_voll.columns:
        raise ValueError("Column 'date' not found in df_voll")

    daily = (
        df_voll.groupby("date")[voll_col]
        .agg(voll_mean="mean", voll_min="min", voll_max="max", obs_count="count")
        .reset_index()
        .sort_values("date")
    )

    if daily.empty:
        raise ValueError("No daily data available after aggregation.")

    # Scenario-specific on-demand VOLL reference
    ondemand_col = f"voll_ondemand_{scenario}"
    on_demand_voll_value = None
    if ondemand_col in df_voll.columns:
        tmp = pd.to_numeric(df_voll[ondemand_col], errors="coerce").dropna()
        if not tmp.empty:
            on_demand_voll_value = tmp.iloc[0]

    fig, ax = plt.subplots(figsize=(12.5, 6.8))

    ax.fill_between(
        daily["date"],
        daily["voll_min"],
        daily["voll_max"],
        alpha=0.22,
        label="Spot VOLL - Min-max envelope",
    )

    ax.plot(
        daily["date"],
        daily["voll_mean"],
        linewidth=2.4,
        label=f"Spot VOLL - Ave ({scenario})",
    )

    if on_demand_voll_value is not None:
        ax.axhline(
            y=on_demand_voll_value,
            color="black",
            linewidth=2.2,
            linestyle="-",
            label=f"On-Demand Implied VOLL ({scenario})",
        )

    ax.axhline(
        y=2000,
        linestyle="--",
        linewidth=1.8,
        color="tab:green",
        label="PJM Offer Cap",
    )

    ax.axhline(
        y=3700,
        linestyle="--",
        linewidth=1.8,
        color="tab:orange",
        label="PJM System Price Cap",
    )

    ax.set_title(
        f"Implied VOLL for {instance_name} ({provider.upper()})\nScenario: {scenario}",
        fontsize=14,
        pad=14,
    )
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("VOLL ($/MWh)", fontsize=12)

    ax.grid(True, alpha=0.30, linewidth=0.8)
    ax.legend(frameon=False, fontsize=10, loc="upper right")

    if ymin is not None or ymax is not None:
        ax.set_ylim(bottom=ymin, top=ymax)

    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.subplots_adjust(top=0.88, bottom=0.16, right=0.97, left=0.10)

    if save_png:
        png_file = OUTPUT_DIR / f"voll_timeseries_{provider}_{instance_name.replace('.', '_')}_{scenario}.png"
        plt.savefig(png_file, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {png_file}")

    if save_csv:
        csv_file = OUTPUT_DIR / f"voll_daily_{provider}_{instance_name.replace('.', '_')}_{scenario}.csv"
        daily.to_csv(csv_file, index=False)
        print(f"Saved daily summary: {csv_file}")

    plt.show()
    return daily
# ============================================================
# Example main
# ============================================================

df_power_sorted

instance_name = "g5.xlarge"
provider = "aws"
scenario_to_plot = "mid"   # choose from idle, lo, mid, hi, custom

df_power = load_power_data()
df_price = load_price_data()

instance_col = "instance_type"

excl_inst = ["p6e-gb200.36xlarge", "trn2.48xlarge", "p5e.48xlarge"]
df_power_exp6e = df_power.loc[
    ~df_power[instance_col].astype(str).str.strip().isin(excl_inst)
    ]

df_instance = df_power.loc[df_power[instance_col].astype(str).str.strip() == instance_name]
df_voll_ondemand_mid =  df_instance["voll_usd_per_mwh_mid"]
df_voll_ondemand_mid

df_voll = compute_voll(
        df_price=df_price,
        df_power=df_power,
        instance_name=instance_name,
        provider=provider,
    )

print("\nPreview:")
preview_cols = [
        "timestamp",
        "Instance Price",
        "VOLL_idle",
        "VOLL_lo",
        "VOLL_mid",
        "VOLL_hi",
        "VOLL_custom",
    ]
preview_cols = [c for c in preview_cols if c in df_voll.columns]
print(df_voll[preview_cols].head())
plot_voll_timeseries(
    df_voll=df_voll,
    scenario="mid",
    instance_name="g5.xlarge",
    location="us-east-1",
    ymin=0,
    ymax=12000,   # adjust based on your data
)


df_power_sorted = df_power_exp6e.sort_values(by="voll_usd_per_mwh_mid", ascending=True)
voll_mid_data = df_power_sorted["voll_usd_per_mwh_mid"].dropna()
plt.histogram(voll_mid_data)#, bins=30)
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Sort by mid VOLL
df_plot = df_power_exp6e.sort_values("voll_usd_per_mwh_mid").reset_index(drop=True)

x = np.arange(len(df_plot))

y_lo = df_plot["voll_usd_per_mwh_lo"]
y_mid = df_plot["voll_usd_per_mwh_mid"]
y_hi = df_plot["voll_usd_per_mwh_hi"]

y_range = y_hi - y_lo

plt.figure(figsize=(14, 6))

# --- Range bars (light fill, dark edge) ---
plt.bar(
    x,
    y_range,
    bottom=y_lo,
    width=0.6,
    edgecolor="black",   # darker border
    linewidth=1.2,
    alpha=0.25           # lighter fill
)

# --- Smaller markers ---
plt.scatter(x, y_lo, s=10, label="Lo", zorder=3)
plt.scatter(x, y_mid, s=20, marker='D', label="Mid", zorder=4)
plt.scatter(x, y_hi, s=10, label="Hi", zorder=3)

# --- Axes ---
plt.xticks(x, df_plot[instance_col], rotation=90)
plt.ylabel("VOLL ($/MWh)")
plt.xlabel("Instance Type")
plt.title("VOLL Range (Lo–Hi) with Midpoint by Instance")

#plt.yscale("log")

plt.legend()
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# Prepare data
df_power_sorted = df_power_exp6e.sort_values(by="voll_usd_per_mwh_mid")
voll_mid_data = df_power_sorted["voll_usd_per_mwh_mid"].dropna()

# KDE
kde = gaussian_kde(voll_mid_data)

# X grid (use log spacing given heavy tails)
x_vals = np.logspace(
    np.log10(voll_mid_data.min()),
    np.log10(voll_mid_data.max()),
    500
)

y_vals = kde(x_vals)

# Plot
plt.figure()
plt.plot(x_vals, y_vals)

#plt.xscale("log")
plt.xlabel("VOLL ($/MWh)")
plt.ylabel("Density")
plt.title("Density of VOLL (Mid Scenario)")

plt.show()