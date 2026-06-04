from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


def build_relations_matrix(df: pd.DataFrame, countries: list[str]) -> pd.DataFrame:
    totals = (
        df.groupby(["From", "To"], dropna=False)["Points"]
        .sum()
        .unstack(fill_value=0)
    )
    totals = totals.reindex(index=countries, columns=countries, fill_value=0)
    return totals - totals.T


def plot_relations_matrix(net: pd.DataFrame, output_path: Path, min_year: int) -> None:
    cmap = LinearSegmentedColormap.from_list(
        "red_orange_yellow", ["#b30000", "#f16913", "#ffd53d"]
    )
    abs_vals = net.abs().stack()
    vmax = float(abs_vals.quantile(0.90))
    if vmax == 0:
        vmax = float(abs_vals.max())
    if vmax == 0:
        vmax = 1.0

    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(
        net,
        cmap=cmap,
        center=0,
        vmin=-vmax,
        vmax=vmax,
        cbar_kws={"label": "Net points (A→B minus B→A)"},
    )
    ax.set_xlabel("To")
    ax.set_ylabel("From")
    ax.set_title(f"Net points between selected countries (years ≥ {min_year})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
