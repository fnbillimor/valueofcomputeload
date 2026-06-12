from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

INPUT_INF_FILE = Path(
    r"C:\Users\FarhadBillimoria\OneDrive - Aurora Energy Research\Documents\Personal\Endog_demand_AI\pyai\data\processed\mlperf\mlperf_with_tdp.csv"
)

INPUT_TRN_FILE = Path(
    r"C:\Users\FarhadBillimoria\OneDrive - Aurora Energy Research\Documents\Personal\Endog_demand_AI\pyai\data\processed\mlperf\mlperf_training_all.csv"
)

OUTPUT_DIR = Path(
    r"C:\Users\FarhadBillimoria\OneDrive - Aurora Energy Research\Documents\Personal\Endog_demand_AI\pyai\data\processed\mlperf"
)


# ============================================================
# Global figure settings
# ============================================================

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})


# ============================================================
# Helpers
# ============================================================

def classify_benchmark_model(x):
    if pd.isna(x):
        return pd.NA

    s_low = str(x).strip().lower()

    if "llama2_70b_lora" in s_low:
        return "Llama2_70b_lora"
    if "stable diffusion" in s_low:
        return "Stable Diffusion"
    if "gpt3" in s_low:
        return "gpt3"
    if "gptj" in s_low:
        return "gptj"
    if "retinanet" in s_low:
        return "RetinaNet"
    if "3d u-net" in s_low or "3d_unet" in s_low or "3d-unet" in s_low:
        return "3D U-Net"
    if "bert" in s_low:
        return "BERT"
    if "dlrm" in s_low:
        return "DLRM"
    if "resnet" in s_low:
        return "ResNet"
    if "rnn-t" in s_low or "rnnt" in s_low:
        return "RNN-T"
    if "ssd-large" in s_low:
        return "SSD-Large"

    return "Other"


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


def numeric_sort_key(x):
    if pd.isna(x):
        return float("inf")

    s = str(x).strip()

    try:
        return float(s)
    except ValueError:
        nums = re.findall(r"\d+(?:\.\d+)?", s)
        return float(nums[0]) if nums else float("inf")


def get_order(sub, group_col, value_col, order_type):
    if order_type == "median":
        return (
            sub.groupby(group_col)[value_col]
            .median()
            .sort_values()
            .index
            .tolist()
        )

    if order_type == "alphabetical":
        return sorted(
            sub[group_col].dropna().unique(),
            key=lambda x: str(x).lower(),
        )

    if order_type == "numeric":
        return sorted(
            sub[group_col].dropna().unique(),
            key=numeric_sort_key,
        )

    raise ValueError("order_type must be 'median', 'alphabetical', or 'numeric'")


# ============================================================
# Main boxplot function
# ============================================================

def plot_box_by_group_then_dataset(
    df_inf,
    df_trn,
    group_col: str,
    value_col: str,
    label_inf: str = "Inference",
    label_trn: str = "Training",
    ref_line: float = 1.0,
    figsize=(7.0, 3.2),
    showfliers: bool = False,
    output_path=None,
    order_type: str = "median",
    count_offset: float = 0.20,
    dataset_offset: float = 0.28,
    bottom_margin: float = 0.42,
    y_limits=None,
):
    inf = df_inf[[group_col, value_col]].copy()
    trn = df_trn[[group_col, value_col]].copy()

    inf["Dataset"] = label_inf
    trn["Dataset"] = label_trn

    df_plot = pd.concat([inf, trn], ignore_index=True)
    df_plot[value_col] = pd.to_numeric(df_plot[value_col], errors="coerce")
    df_plot = df_plot.dropna(subset=[group_col, value_col, "Dataset"])

    if df_plot.empty:
        raise ValueError(f"No valid data for {group_col} and {value_col}")

    data = []
    positions = []
    tick_labels = []
    block_midpoints = {}

    pos = 1

    for dataset_label in [label_inf, label_trn]:
        sub = df_plot[df_plot["Dataset"] == dataset_label]
        group_order = get_order(sub, group_col, value_col, order_type)

        start_pos = pos

        for group in group_order:
            vals = sub.loc[sub[group_col] == group, value_col]
            data.append(vals)
            positions.append(pos)
            tick_labels.append(group)
            pos += 1

        end_pos = pos - 1

        if end_pos >= start_pos:
            block_midpoints[dataset_label] = (start_pos + end_pos) / 2

        pos += 1

    sample_counts = [len(vals) for vals in data]
    count_labels = [f"n={c}" for c in sample_counts]

    fig, ax = plt.subplots(figsize=figsize)

    box = ax.boxplot(
        data,
        positions=positions,
        patch_artist=True,
        showfliers=showfliers,
    )

    for patch in box["boxes"]:
        patch.set_alpha(0.4)

    ax.set_xticks(positions)
    ax.set_xticklabels(
        tick_labels,
        rotation=45,
        ha="right",
        fontsize=8,
    )

    ax.set_ylabel("Power-to-TDP Multiple")

    if ref_line is not None:
        ax.axhline(ref_line, linestyle="--", linewidth=1)

    ax.grid(axis="y", linestyle="--", alpha=0.5)

    if y_limits is not None:
        ax.set_ylim(y_limits)

    ymin, ymax = ax.get_ylim()
    y_range = ymax - ymin

    count_y = ymin - count_offset * y_range
    dataset_y = ymin - dataset_offset * y_range

    for x, label in zip(positions, count_labels):
        ax.text(
            x,
            count_y,
            label,
            ha="center",
            va="top",
            fontsize=8,
            clip_on=False,
        )

    for dataset_label in [label_inf, label_trn]:
        if dataset_label in block_midpoints:
            ax.text(
                block_midpoints[dataset_label],
                dataset_y,
                dataset_label,
                ha="center",
                va="top",
                fontsize=8,
                fontweight="bold",
                clip_on=False,
            )

    fig.subplots_adjust(bottom=bottom_margin)

    if output_path is not None:
        fig.savefig(output_path, dpi=600, bbox_inches="tight")

    plt.show()

    return df_plot


# ============================================================
# Histogram function
# ============================================================

def plot_inf_vs_trn_hist(
    df_inf,
    df_trn,
    value_col: str,
    bins: int = 10,
    figsize=(7.0, 2.8),
    density: bool = False,
    ref_line: float = 1.0,
    output_path=None,
):
    inf = pd.to_numeric(df_inf[value_col], errors="coerce").dropna()
    trn = pd.to_numeric(df_trn[value_col], errors="coerce").dropna()

    if inf.empty or trn.empty:
        raise ValueError("One of the datasets is empty after cleaning.")

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)

    axes[0].hist(inf, bins=bins, density=density)
    axes[0].axvline(ref_line, linestyle="--", linewidth=1)
    axes[0].set_xlabel("Power-to-TDP Multiple")
    axes[0].set_ylabel("Density" if density else "Frequency")
    axes[0].set_title("Inference", fontsize=8)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    axes[1].hist(trn, bins=bins, density=density)
    axes[1].axvline(ref_line, linestyle="--", linewidth=1)
    axes[1].set_xlabel("Power-to-TDP Multiple")
    axes[1].set_title("Training", fontsize=8)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=600, bbox_inches="tight")

    plt.show()


# ============================================================
# Main script
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

mlperf_inf_input = read_mlperf_csv(INPUT_INF_FILE)
mlperf_trn_input = read_mlperf_csv(INPUT_TRN_FILE)

value_col = "Watts / CPU+GPU TDP"


# ============================================================
# Shared settings for aligned panel figures
# ============================================================

COMMON_FIGSIZE = (7.0, 6.0)
COMMON_COUNT_OFFSET = 0.3
COMMON_DATASET_OFFSET = 0.2
COMMON_BOTTOM_MARGIN = 0.2
COMMON_Y_LIMITS = (0, 1.5)
ONECOL_COMMON_FIGSIZE = (14.0, 6.0)

# ============================================================
# Figure: By GPU
# ============================================================

GPU_OUTPUT_FILE = OUTPUT_DIR / "By_GPU.png"

GPU_plot = plot_box_by_group_then_dataset(
    mlperf_inf_input,
    mlperf_trn_input,
    group_col="GPU",
    value_col=value_col,
    ref_line=1.0,
    output_path=GPU_OUTPUT_FILE,
    order_type="alphabetical",
    figsize=COMMON_FIGSIZE,
    count_offset=COMMON_COUNT_OFFSET,
    dataset_offset=COMMON_DATASET_OFFSET,
    bottom_margin=COMMON_BOTTOM_MARGIN,
    y_limits=COMMON_Y_LIMITS,
)

GPU_plot.to_csv(OUTPUT_DIR / "By_GPU.csv", index=False)


# ============================================================
# Figure: By number of GPUs
# ============================================================

NUM_OUTPUT_FILE = OUTPUT_DIR / "By_num_GPUs.png"

NUM_plot = plot_box_by_group_then_dataset(
    mlperf_inf_input,
    mlperf_trn_input,
    group_col="Total Accelerators",
    value_col=value_col,
    ref_line=1.0,
    output_path=NUM_OUTPUT_FILE,
    order_type="numeric",
    figsize=COMMON_FIGSIZE,
    count_offset=COMMON_COUNT_OFFSET,
    dataset_offset=COMMON_DATASET_OFFSET,
    bottom_margin=COMMON_BOTTOM_MARGIN,
    y_limits=COMMON_Y_LIMITS,
)

NUM_plot.to_csv(OUTPUT_DIR / "By_num_GPUs.csv", index=False)


# ============================================================
# Figure: Histogram
# ============================================================

HIST_OUTPUT_FILE = OUTPUT_DIR / "Hist.png"

plot_inf_vs_trn_hist(
    mlperf_inf_input,
    mlperf_trn_input,
    value_col=value_col,
    bins=10,
    output_path=HIST_OUTPUT_FILE,
    density=False,
)


# ============================================================
# Figure: By benchmark model
# ============================================================

MOD_OUTPUT_FILE = OUTPUT_DIR / "By_Benchmark_model.png"

MOD_plot = plot_box_by_group_then_dataset(
    mlperf_inf_input,
    mlperf_trn_input,
    group_col="Benchmark Model",
    value_col=value_col,
    ref_line=1.0,
    output_path=MOD_OUTPUT_FILE,
    order_type="alphabetical",
    figsize=ONECOL_COMMON_FIGSIZE,
    count_offset=COMMON_COUNT_OFFSET,
    dataset_offset=COMMON_DATASET_OFFSET,
    bottom_margin=COMMON_BOTTOM_MARGIN,
    y_limits=COMMON_Y_LIMITS,
)

MOD_plot.to_csv(OUTPUT_DIR / "By_Benchmark_model.csv", index=False)


# ============================================================
# Figure: By benchmark model group
# ============================================================

mlperf_inf_input["Benchmark Model Group"] = (
    mlperf_inf_input["Benchmark Model"].apply(classify_benchmark_model)
)

mlperf_trn_input["Benchmark Model Group"] = (
    mlperf_trn_input["Benchmark Model"].apply(classify_benchmark_model)
)

MOD_GROUP_OUTPUT_FILE = OUTPUT_DIR / "By_Benchmark_model_group.png"

MOD_group_plot = plot_box_by_group_then_dataset(
    mlperf_inf_input,
    mlperf_trn_input,
    group_col="Benchmark Model Group",
    value_col=value_col,
    ref_line=1.0,
    output_path=MOD_GROUP_OUTPUT_FILE,
    order_type="alphabetical",
    figsize=ONECOL_COMMON_FIGSIZE,
    count_offset=COMMON_COUNT_OFFSET,
    dataset_offset=COMMON_DATASET_OFFSET,
    bottom_margin=COMMON_BOTTOM_MARGIN,
    y_limits=COMMON_Y_LIMITS,
)

MOD_group_plot.to_csv(OUTPUT_DIR / "By_Benchmark_model_group.csv", index=False)


# ============================================================
# Percentile summary
# ============================================================

def percentile_summary(df, dataset_name, value_col=value_col):
    x = pd.to_numeric(df[value_col], errors="coerce").dropna()

    return pd.DataFrame({
        "Dataset": [dataset_name],
        "n": [len(x)],
        "P10": [x.quantile(0.10)],
        "P50": [x.quantile(0.50)],
        "P90": [x.quantile(0.90)],
    })


pct_summary = pd.concat(
    [
        percentile_summary(mlperf_inf_input, "Inference"),
        percentile_summary(mlperf_trn_input, "Training"),
    ],
    ignore_index=True,
)

print(pct_summary)

pct_summary.to_csv(
    OUTPUT_DIR / "Power_to_TDP_percentiles.csv",
    index=False,
)


# ============================================================
# Completion messages
# ============================================================

print(f"Saved GPU plot to: {GPU_OUTPUT_FILE}")
print(f"Saved GPU data to: {OUTPUT_DIR / 'By_GPU.csv'}")
print(f"Saved accelerator-count plot to: {NUM_OUTPUT_FILE}")
print(f"Saved accelerator-count data to: {OUTPUT_DIR / 'By_num_GPUs.csv'}")
print(f"Saved histogram to: {HIST_OUTPUT_FILE}")
print(f"Saved benchmark model plot to: {MOD_OUTPUT_FILE}")
print(f"Saved benchmark model group plot to: {MOD_GROUP_OUTPUT_FILE}")
print(f"Saved percentile summary to: {OUTPUT_DIR / 'Power_to_TDP_percentiles.csv'}")