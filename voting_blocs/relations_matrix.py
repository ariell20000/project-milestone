from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_relations_matrix(df: pd.DataFrame, countries: list[str]) -> pd.DataFrame:
    totals = (
        df.groupby(["From", "To"], dropna=False)["Points"]
        .sum()
        .unstack(fill_value=0)
    )
    totals = totals.reindex(index=countries, columns=countries, fill_value=0)
    return totals - totals.T


