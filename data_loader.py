from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
import seaborn as sns

from jury_televote import JURY_COL, TELEVOTE_COL

MIN_YEAR = 2000
VALID_POINTS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12}
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

# Path to the CSV dataset sitting one level above this package
DATA_PATH = Path(__file__).resolve().parent.parent / "eurovision_1957-2021.csv"


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.columns[0] == "":
        df = df.drop(columns=df.columns[0])
    return df


def normalize_country(value: object) -> str:
    if pd.isna(value):
        return ""
    cleaned = re.sub(r"\s+", " ", str(value).replace("-", " ")).strip().lower()
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
    has_both = (counts.get(JURY_COL, 0) > 0) & (counts.get(TELEVOTE_COL, 0) > 0)
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


def prepare_data() -> pd.DataFrame:
    """Load, clean, and filter the dataset. Returns a ready-to-use DataFrame."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    sns.set_theme(style="whitegrid")

    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = df[df["Year"] >= MIN_YEAR].copy()

    target_countries = get_target_countries()
    available = set(df["From"]).union(df["To"])
    missing = [c for c in target_countries if c not in available]
    if missing:
        print("Warning: countries not found in filtered data:", ", ".join(missing))

    return df
