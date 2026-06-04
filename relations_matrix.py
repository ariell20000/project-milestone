from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def build_relations_matrix(df: pd.DataFrame, countries: list[str]) -> pd.DataFrame:
    yearly = (
        df.groupby(["Year", "From", "To"])["Points"]
        .sum()
        .unstack("To", fill_value=0)
        .reindex(columns=countries, fill_value=0)
    )

    yearly_nets = []
    for _, group in yearly.groupby(level="Year"):
        mat = group.droplevel("Year").reindex(index=countries, fill_value=0)
        yearly_nets.append(mat - mat.T)

    if not yearly_nets:
        return pd.DataFrame(0.0, index=countries, columns=countries)

    return sum(yearly_nets) / len(yearly_nets)


def plot_relations_matrix(avg_net: pd.DataFrame, output_path: Path, min_year: int) -> None:
    # mask lower triangle + diagonal — show each pair once
    mask = np.tril(np.ones(avg_net.shape, dtype=bool))

    avg_abs = avg_net.abs()
    upper_vals = avg_abs.where(~mask).stack()
    vmax = float(upper_vals.quantile(0.90))
    if vmax == 0:
        vmax = float(upper_vals.max())
    if vmax == 0:
        vmax = 1.0

    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(
        avg_abs,           # color intensity = absolute value
        mask=mask,
        cmap="YlOrRd",
        vmin=0,
        vmax=vmax,
        annot=avg_net,     # annotation text = signed value
        fmt="+.1f",
        cbar_kws={"label": "Avg yearly |net points|"},
    )
    ax.set_xlabel("To")
    ax.set_ylabel("From")
    ax.set_title(
        f"Avg yearly net points between selected countries (years ≥ {min_year})\n"
        "Color = magnitude  |  Sign = row gave more (+) or column gave more (−)"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
