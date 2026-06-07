"""Generate the jury-vs-televote difference bar chart."""
from __future__ import annotations

from pathlib import Path

from data_loader import MIN_YEAR, SPLIT_POINTS_TYPES, prepare_data, years_with_split_votes
from jury_televote import build_jury_televote_diff, plot_jury_televote_diff, validate_coverage


def main() -> None:
    df = prepare_data()

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)

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

    validate_coverage(df_split)
    diff = build_jury_televote_diff(df_split)
    plot_jury_televote_diff(diff, output_dir / "jury_vs_televote.png", years_label)

    print(f"Saved: {output_dir / 'jury_vs_televote.png'}")


if __name__ == "__main__":
    main()
