from __future__ import annotations

import bisect
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "eurovision_enriched.csv"

# How many of the most recent grand-final years to use as the reference
# distribution. Recent years reflect the current voting/scoring system;
# older eras used different scoring rules and aren't comparable.
RECENT_YEARS_COUNT = 8
DEFAULT_FIELD_SIZE = 26


@lru_cache(maxsize=1)
def load_historical_scores(path: Path = DATA_PATH) -> np.ndarray:
    df = pd.read_csv(path)
    if df.columns[0] == "":
        df = df.drop(columns=df.columns[0])

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Place"] = pd.to_numeric(df["Place"], errors="coerce")
    df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
    # Only rows with a real Place are grand-final results (semi-final-only
    # entries have "-" for Place in this dataset).
    df = df.dropna(subset=["Year", "Place", "Score"])

    recent_years = sorted(df["Year"].unique())[-RECENT_YEARS_COUNT:]
    recent = df[df["Year"].isin(recent_years)]
    return np.sort(recent["Score"].to_numpy())


def estimate_place(
    predicted_score: float, field_size: int = DEFAULT_FIELD_SIZE
) -> tuple[int, float]:
    """Estimate a rough place out of `field_size`, based on where
    `predicted_score` falls in the historical score distribution.

    This is an approximation against historical results, not a simulated
    placement against a specific field of competing songs.
    """
    scores = load_historical_scores()
    n = len(scores)
    below_or_equal = bisect.bisect_right(scores, predicted_score)
    percentile = 100.0 * below_or_equal / n

    place = 1 + round((1 - percentile / 100.0) * (field_size - 1))
    place = max(1, min(field_size, place))
    return place, percentile
