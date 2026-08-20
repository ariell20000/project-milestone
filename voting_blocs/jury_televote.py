from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

JURY_COL = "Points given by the jury"
TELEVOTE_COL = "Points given by televoters"


def build_jury_televote_diff(df: pd.DataFrame, countries: list[str]) -> pd.Series:
    pivot = df.pivot_table(
        index="To",
        columns="Points type",
        values="Points",
        aggfunc="sum",
        fill_value=0,
    )
    for col in (JURY_COL, TELEVOTE_COL):
        if col not in pivot.columns:
            pivot[col] = 0

    diff = pivot[JURY_COL] - pivot[TELEVOTE_COL]
    return diff.reindex(countries, fill_value=0).sort_values()


def plot_jury_televote_diff(
    diff: pd.Series, output_path: Path, years_label: str
) -> None:
    colors = diff.apply(
        lambda v: "#d73027" if v > 0 else "#2166ac" if v < 0 else "#888888"
    )

    plt.figure(figsize=(14, 6))
    ax = diff.plot(kind="bar", color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Jury − Televote points")
    ax.set_xlabel("Country")
    ax.set_title(
        f"Jury vs Televote points received (selected countries, {years_label})"
    )
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
