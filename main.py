from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
MIN_YEAR = 2000
VALID_POINTS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12}
JURY_COL = "Points given by the jury"
TELEVOTE_COL = "Points given by televoters"
SPLIT_POINTS_TYPES = {JURY_COL, TELEVOTE_COL}
ALLOWED_POINTS_TYPES = {"Points given"} | SPLIT_POINTS_TYPES
TARGET_COUNTRIES_RAW = [
    "Israel",
    "serbia",
    "France",
    "Germany",
    "england",
    "malta",
    "switzerland",
    "sweeden",
    "ukraine",
    "greece",
    "cyprus",
    "slovenia",
]
COUNTRY_ALIASES = {
    "aerbaijan": "Azerbaijan",
    "azerbaijan": "Azerbaijan",
    "england": "United Kingdom",
    "united kingdom": "United Kingdom",
    "sweeden": "Sweden",
    "sweden": "Sweden",
    "bosnia": "Bosnia and Herzegovina",
    "bosnia and herzegovina": "Bosnia and Herzegovina",
}


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.columns[0] == "":
        df = df.drop(columns=df.columns[0])
    return df


def normalize_country(value: object) -> str:
    if pd.isna(value):
        return ""
    cleaned = re.sub(r"\s+", " ", str(value)).strip().lower()
    if not cleaned:
        return ""
    if cleaned in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[cleaned]
    return cleaned.title()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Points"] = pd.to_numeric(df["Points"], errors="coerce")
    df["From"] = df["From"].map(normalize_country)
    df["To"] = df["To"].map(normalize_country)
    df["Points type"] = df["Points type"].astype(str).str.strip()

    df = df.dropna(subset=["Year", "Points"])
    df = df[(df["From"] != "") & (df["To"] != "")]
    df = df[df["From"] != df["To"]]
    df = df[df["Points"].isin(VALID_POINTS)]
    df = df[df["Points type"].isin(ALLOWED_POINTS_TYPES)]
    df = df.drop_duplicates(
        subset=["Edition", "Year", "Points type", "From", "To", "Points"]
    )
    return df


def years_with_split_votes(df: pd.DataFrame) -> list[int]:
    split = df[df["Points type"].isin(SPLIT_POINTS_TYPES)]
    counts = (
        split.groupby(["Year", "Points type"])
        .size()
        .unstack(fill_value=0)
    )
    has_both = (counts.get(JURY_COL, 0) > 0) & (
        counts.get(TELEVOTE_COL, 0) > 0
    )
    return sorted(counts[has_both].index.astype(int).tolist())


def get_target_countries() -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for raw in TARGET_COUNTRIES_RAW:
        country = normalize_country(raw)
        if country and country not in seen:
            ordered.append(country)
            seen.add(country)
    return ordered


def build_relations_matrix(
    df: pd.DataFrame, countries: list[str]
) -> pd.DataFrame:
    totals = (
        df.groupby(["From", "To"], dropna=False)["Points"]
        .sum()
        .unstack(fill_value=0)
    )
    totals = totals.reindex(index=countries, columns=countries, fill_value=0)
    net = totals - totals.T
    return net


def plot_relations_matrix(net: pd.DataFrame, output_path: Path) -> None:
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
    ax.set_title(f"Net points between selected countries (years ≥ {MIN_YEAR})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def build_jury_televote_diff(
    df: pd.DataFrame, countries: list[str]
) -> pd.Series:
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
    diff = diff.reindex(countries, fill_value=0)
    return diff.sort_values()


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
        "Jury vs Televote points received "
        f"(selected countries, {years_label})"
    )
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "eurovision_1957-2021.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    sns.set_theme(style="whitegrid")

    df = load_data(data_path)
    df = clean_data(df)
    df = df[df["Year"] >= MIN_YEAR].copy()
    target_countries = get_target_countries()
    available = set(df["From"]).union(df["To"])
    missing = [c for c in target_countries if c not in available]
    if missing:
        print("Warning: countries not found in filtered data:", ", ".join(missing))

    output_dir = base_dir / "outputs"
    output_dir.mkdir(exist_ok=True)

    df_matrix = df[
        df["From"].isin(target_countries) & df["To"].isin(target_countries)
    ].copy()
    net = build_relations_matrix(df_matrix, target_countries)
    plot_relations_matrix(net, output_dir / "relations_matrix.png")

    split_years = years_with_split_votes(df)
    if split_years:
        years_label = f"years {split_years[0]}–{split_years[-1]}"
        df_split = df[
            df["Year"].isin(split_years)
            & df["Points type"].isin(SPLIT_POINTS_TYPES)
        ].copy()
    else:
        years_label = f"years ≥ {MIN_YEAR}"
        df_split = df[df["Points type"].isin(SPLIT_POINTS_TYPES)].copy()

    df_to_targets = df_split[df_split["To"].isin(target_countries)].copy()
    diff = build_jury_televote_diff(df_to_targets, target_countries)
    plot_jury_televote_diff(diff, output_dir / "jury_vs_televote.png", years_label)

    print("Saved:")
    print(f"- {output_dir / 'relations_matrix.png'}")
    print(f"- {output_dir / 'jury_vs_televote.png'}")


if __name__ == "__main__":
    main()
