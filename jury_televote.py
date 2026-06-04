from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

JURY_COL = "Points given by the jury"
TELEVOTE_COL = "Points given by televoters"


def validate_coverage(df: pd.DataFrame) -> None:
    all_countries = set(df["To"].unique()) | set(df["From"].unique())
    split_years = sorted(df["Year"].unique().astype(int).tolist())

    missing_received: dict[str, list[int]] = {}
    missing_voted: dict[str, list[int]] = {}

    for country in sorted(all_countries):
        absent = [y for y in split_years if country not in df[df["Year"] == y]["To"].values]
        if absent:
            missing_received[country] = absent
        did_not_vote = [y for y in split_years if country not in df[df["Year"] == y]["From"].values]
        if did_not_vote:
            missing_voted[country] = did_not_vote

    if not missing_received and not missing_voted:
        print("Coverage OK: all countries received votes and voted in every year.")
        return

    print("=== Jury/Televote coverage report ===")
    if missing_received:
        print("\nMissing as vote recipient:")
        for country, years in missing_received.items():
            print(f"  {country}: {years}")
    if missing_voted:
        print("\nDid not vote:")
        for country, years in missing_voted.items():
            print(f"  {country}: {years}")
    print()


def build_jury_televote_diff(df: pd.DataFrame) -> pd.Series:
    # sum per year per (To, Points type), then average across years
    yearly = (
        df.groupby(["Year", "To", "Points type"])["Points"]
        .sum()
        .unstack("Points type", fill_value=0)
    )
    for col in (JURY_COL, TELEVOTE_COL):
        if col not in yearly.columns:
            yearly[col] = 0

    avg = yearly.groupby("To")[[JURY_COL, TELEVOTE_COL]].mean()
    return (avg[JURY_COL] - avg[TELEVOTE_COL]).sort_values()


def plot_jury_televote_diff(
    diff: pd.Series, output_path: Path, years_label: str
) -> None:
    colors = diff.apply(
        lambda v: "#d73027" if v > 0 else "#2166ac" if v < 0 else "#888888"
    )

    n = len(diff)
    plt.figure(figsize=(max(14, n * 0.4), 6))
    ax = diff.plot(kind="bar", color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Avg yearly Jury − Televote points")
    ax.set_xlabel("Country")
    ax.set_title(
        f"Avg yearly Jury vs Televote points received (all countries, {years_label})"
    )
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
