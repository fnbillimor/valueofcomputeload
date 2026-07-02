from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "instances" / "graphs"

TOTAL_LOAD_GW = 80.0

SERVICE_TYPE_SHARES = {
    "On-demand": 0.90,
    "Interruptible": 0.10,
}

AI_WORKLOAD_SHARES = {
    "Emerging / Frontier AI": 0.50,
    "Mainstream Modern AI": 0.40,
    "Transitional / Early AI": 0.10,
}

MEDIAN_VOCL_USD_PER_MWH = {
    "On-demand": {
        "Legacy / Older Vintage": 2600,
        "Transitional / Early AI": 8100,
        "Mainstream Modern AI": 8300,
        "Emerging / Frontier AI": 11600,
    },
    "Spot": {
        "Legacy / Older Vintage": 1100,
        "Transitional / Early AI": 2500,
        "Mainstream Modern AI": 3300,
        "Emerging / Frontier AI": 2800,
    },
}

SERVICE_TYPE_TO_VOCL_BASIS = {
    "On-demand": "On-demand",
    "Interruptible": "Spot",
}

SERVICE_TYPE_COLORS = {
    "On-demand": "#2f6f8f",
    "Interruptible": "#c76e3b",
}

WORKLOAD_COLORS = {
    "Emerging / Frontier AI": "#7b4ab2",
    "Mainstream Modern AI": "#ff7f0e",
    "Transitional / Early AI": "#3fa34d",
}

SERVICE_TYPE_LINESTYLES = {
    "On-demand": "-",
    "Interruptible": "--",
}

WORKLOAD_SHORT_LABELS = {
    "Emerging / Frontier AI": "Frontier AI",
    "Mainstream Modern AI": "Mainstream AI",
    "Transitional / Early AI": "Early AI",
}


@dataclass(frozen=True)
class DemandBlock:
    service_type: str
    vocl_basis: str
    workload: str
    service_share: float
    workload_share_within_service: float
    load_gw: float
    median_vocl_usd_per_mwh: float


def build_bid_blocks() -> pd.DataFrame:
    blocks: list[DemandBlock] = []
    for service_type, service_share in SERVICE_TYPE_SHARES.items():
        vocl_basis = SERVICE_TYPE_TO_VOCL_BASIS[service_type]
        for workload, workload_share in AI_WORKLOAD_SHARES.items():
            load_gw = TOTAL_LOAD_GW * service_share * workload_share
            blocks.append(
                DemandBlock(
                    service_type=service_type,
                    vocl_basis=vocl_basis,
                    workload=workload,
                    service_share=service_share,
                    workload_share_within_service=workload_share,
                    load_gw=load_gw,
                    median_vocl_usd_per_mwh=MEDIAN_VOCL_USD_PER_MWH[vocl_basis][workload],
                )
            )

    df = pd.DataFrame([block.__dict__ for block in blocks])
    df["total_load_share"] = df["load_gw"] / TOTAL_LOAD_GW
    return df[
        [
            "service_type",
            "vocl_basis",
            "workload",
            "service_share",
            "workload_share_within_service",
            "total_load_share",
            "load_gw",
            "median_vocl_usd_per_mwh",
        ]
    ]


def build_sorted_bid_curve(blocks: pd.DataFrame) -> pd.DataFrame:
    curve = blocks.sort_values(
        ["median_vocl_usd_per_mwh", "service_type", "workload"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    curve["cumulative_load_start_gw"] = curve["load_gw"].cumsum().shift(fill_value=0.0)
    curve["cumulative_load_end_gw"] = curve["load_gw"].cumsum()
    curve["bid_rank"] = range(1, len(curve) + 1)
    return curve[
        [
            "bid_rank",
            "service_type",
            "vocl_basis",
            "workload",
            "load_gw",
            "cumulative_load_start_gw",
            "cumulative_load_end_gw",
            "median_vocl_usd_per_mwh",
        ]
    ]


def plot_bid_curve(curve: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    label_fontsize = 16
    tick_fontsize = 14
    legend_fontsize = 14
    low_price_cap = 3000
    high_price_cap = 20000

    previous_y: float | None = None
    for _, row in curve.iterrows():
        color = WORKLOAD_COLORS[row["workload"]]
        linestyle = SERVICE_TYPE_LINESTYLES[row["service_type"]]
        x_start = row["cumulative_load_start_gw"]
        x_end = row["cumulative_load_end_gw"]
        y = row["median_vocl_usd_per_mwh"]

        if previous_y is not None and previous_y != y:
            ax.vlines(
                x=x_start,
                ymin=min(previous_y, y),
                ymax=max(previous_y, y),
                colors=color,
                linestyles=linestyle,
                linewidth=1.5,
                alpha=0.8,
            )

        ax.plot(
            [x_start, x_end],
            [y, y],
            color=color,
            linestyle=linestyle,
            linewidth=2.2,
            marker="o",
            markersize=4,
        )
        previous_y = y

    handles = [
        plt.Line2D([0], [0], color=color, linewidth=2.4, label=label)
        for label, color in WORKLOAD_COLORS.items()
    ]
    handles += [
        plt.Line2D(
            [0],
            [0],
            color="#333333",
            linewidth=2.4,
            linestyle=linestyle,
            label=service_type,
        )
        for service_type, linestyle in SERVICE_TYPE_LINESTYLES.items()
    ]

    ax.axhline(
        low_price_cap,
        color="#555555",
        linestyle=":",
        linewidth=1.4,
        alpha=0.85,
        label="_nolegend_",
    )
    ax.axhline(
        high_price_cap,
        color="#555555",
        linestyle=":",
        linewidth=1.4,
        alpha=0.85,
        label="_nolegend_",
    )
    ax.text(
        1.0,
        low_price_cap,
        "$3,000/MWh low price cap",
        ha="left",
        va="bottom",
        fontsize=12,
        color="#555555",
    )
    ax.text(
        1.0,
        high_price_cap,
        "$20,000/MWh high price cap",
        ha="left",
        va="bottom",
        fontsize=12,
        color="#555555",
    )

    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1.0, 0.92),
        ncol=2,
        fontsize=legend_fontsize,
    )

    ax.set_xlabel("Cumulative Demand Bid Quantities (GW)", fontsize=label_fontsize)
    ax.set_ylabel("Demand Bid Price ($/MWh)", fontsize=label_fontsize)
    ax.set_xlim(0, TOTAL_LOAD_GW)
    ax.set_ylim(0, high_price_cap * 1.08)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    blocks = build_bid_blocks()
    curve = build_sorted_bid_curve(blocks)

    blocks_path = OUTPUT_DIR / "hypothetical_datacenter_demand_blocks.csv"
    curve_path = OUTPUT_DIR / "hypothetical_datacenter_demand_bid_curve.csv"
    graph_path = OUTPUT_DIR / "hypothetical_datacenter_demand_bid_curve.png"

    blocks.to_csv(blocks_path, index=False)
    curve.to_csv(curve_path, index=False)
    plot_bid_curve(curve, graph_path)

    print(f"Saved demand blocks: {blocks_path}")
    print(f"Saved bid curve table: {curve_path}")
    print(f"Saved bid curve graph: {graph_path}")
    print()
    print(curve.to_string(index=False))


if __name__ == "__main__":
    main()
