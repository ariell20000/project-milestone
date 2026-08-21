from __future__ import annotations

from pathlib import Path

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


