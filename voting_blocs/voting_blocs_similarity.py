from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

from jury_televote import JURY_COL, TELEVOTE_COL
from main import clean_data, load_data

BASE_DIR = Path(__file__).resolve().parent
VOTES_PATH = BASE_DIR.parent / "eurovision_1957-2021.csv"  # shared repo-root dataset
VOTES_PATH_EXTENSION = BASE_DIR / "eurovision_2022-2026.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

SIMILARITY_PATH = OUTPUT_DIR / "voting_blocs_similarity_matrix.csv"
POINTS_MATRIX_PATH = OUTPUT_DIR / "voting_blocs_points_matrix.csv"
JURY_MATRIX_PATH = OUTPUT_DIR / "voting_blocs_jury_matrix.csv"
TELEVOTE_MATRIX_PATH = OUTPUT_DIR / "voting_blocs_televote_matrix.csv"
HEATMAP_PATH = OUTPUT_DIR / "voting_blocs_similarity_heatmap.png"

MIN_YEAR = 2004
MAX_YEAR = 2026
SPLIT_MIN_YEAR = 2016

# Countries whose outgoing row rests on only a handful of editions produce
# similarity values dominated by which contests they happened to attend rather
# than by who they favour, so they are kept out of the clustered space.
MIN_VOTING_EDITIONS = 4

# A pair counts as "voted for" for the Jaccard baseline once it clears one
# point per shared edition, i.e. roughly one mid-table vote every year.
JACCARD_THRESHOLD = 1.0

FROM_COL = "From"
TO_COL = "To"
POINTS_COL = "Points"

# The votes file spells the same country two ways depending on whether it shows
# up as a voter ("United-Kingdom") or as a recipient ("United Kingdom"), which
# would otherwise split one country across two rows/columns of the matrix.
COUNTRY_ALIASES = {
    "Czech-Republic": "Czech Republic",
    "San-Marino": "San Marino",
    "The-Netherlands": "Netherlands",
    "The Netherlands": "Netherlands",
    "United-Kingdom": "United Kingdom",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Serbia & Montenegro": "Serbia and Montenegro",
}


def harmonize_country(name: str) -> str:
    return COUNTRY_ALIASES.get(name, name)


def load_votes(
    min_year: int = MIN_YEAR,
    max_year: int = MAX_YEAR,
) -> pd.DataFrame:
    """Merges the 1957-2021 votes file with the 2022+ extension (same schema,
    fetched separately since Eurovision started publishing per-year) before
    cleaning - mirrors build_jury_televote_split.py's old+new concat, so
    every consumer of this module sees the full 2004-2026 history rather
    than stopping at the original file's 2021 cutoff."""
    raw = pd.concat([load_data(VOTES_PATH), load_data(VOTES_PATH_EXTENSION)], ignore_index=True)
    votes = clean_data(raw)
    votes = votes[votes["Year"].between(min_year, max_year)].copy()
    votes[FROM_COL] = votes[FROM_COL].map(harmonize_country)
    votes[TO_COL] = votes[TO_COL].map(harmonize_country)
    votes["Year"] = votes["Year"].astype(int)
    return votes.reset_index(drop=True)


def all_countries(votes: pd.DataFrame) -> list[str]:
    return sorted(set(votes[FROM_COL]) | set(votes[TO_COL]))


def voting_editions(votes: pd.DataFrame) -> pd.Series:
    return votes.groupby(FROM_COL)["Year"].nunique().sort_values(ascending=False)


def eligible_voters(
    votes: pd.DataFrame, min_editions: int = MIN_VOTING_EDITIONS
) -> list[str]:
    editions = voting_editions(votes)
    return sorted(editions[editions >= min_editions].index)


def build_points_matrix(votes: pd.DataFrame, countries: list[str]) -> pd.DataFrame:
    totals = (
        votes.groupby([FROM_COL, TO_COL])[POINTS_COL]
        .sum()
        .unstack(fill_value=0)
        .reindex(index=countries, columns=countries, fill_value=0)
    )
    return totals.astype(float)


def build_opportunity_matrix(votes: pd.DataFrame, countries: list[str]) -> pd.DataFrame:
    """Per (voter, recipient) pair, the number of editions in which the voter
    actually voted and the recipient actually competed."""
    position = {country: i for i, country in enumerate(countries)}
    counts = np.zeros((len(countries), len(countries)), dtype=float)
    for _year, edition in votes.groupby("Year"):
        voters = [position[c] for c in set(edition[FROM_COL]) if c in position]
        recipients = [position[c] for c in set(edition[TO_COL]) if c in position]
        counts[np.ix_(voters, recipients)] += 1.0
    np.fill_diagonal(counts, 0.0)
    return pd.DataFrame(counts, index=countries, columns=countries)


def build_rate_matrix(votes: pd.DataFrame, countries: list[str]) -> pd.DataFrame:
    """Average points given per shared edition.

    Raw sums conflate affinity with co-participation: Montenegro and the UK can
    hand out very different totals to the same neighbour purely because one
    voted in 2 finals and the other in 17. Dividing by the number of editions
    the pair actually shared turns every cell into "points per chance to vote",
    which is comparable across countries with different attendance records.
    """
    points = build_points_matrix(votes, countries)
    opportunity = build_opportunity_matrix(votes, countries)
    return points.div(opportunity.where(opportunity > 0)).fillna(0.0)


def center_by_recipient(
    rates: pd.DataFrame, opportunity: pd.DataFrame
) -> pd.DataFrame:
    """Subtract each recipient's average received rate, over the voters who
    could actually vote for it.

    Without this, two countries look similar simply because they both gave
    points to the songs everyone liked - which is the musical-taste signal, not
    the bloc signal. Centering leaves the residual "how far above or below the
    field did this voter rate that recipient", so cosine on the residuals asks
    whether two countries deviate from the consensus in the same direction.
    Cells where a pair never shared an edition stay at 0 rather than becoming a
    spurious negative preference.
    """
    values = rates.to_numpy(dtype=float)
    mask = opportunity.loc[rates.index, rates.columns].to_numpy() > 0
    totals = np.where(mask, values, 0.0).sum(axis=0)
    counts = mask.sum(axis=0)
    means = np.divide(totals, counts, out=np.zeros_like(totals), where=counts > 0)
    centered = np.where(mask, values - means, 0.0)
    return pd.DataFrame(centered, index=rates.index, columns=rates.columns)


def cosine_similarity_matrix(profiles: pd.DataFrame) -> pd.DataFrame:
    similarity = cosine_similarity(profiles.to_numpy())
    np.fill_diagonal(similarity, 1.0)
    return pd.DataFrame(similarity, index=profiles.index, columns=profiles.index)


def jaccard_similarity_matrix(
    rates: pd.DataFrame, threshold: float = JACCARD_THRESHOLD
) -> pd.DataFrame:
    binary = (rates.to_numpy() >= threshold).astype(float)
    intersection = binary @ binary.T
    sizes = binary.sum(axis=1)
    union = sizes[:, None] + sizes[None, :] - intersection
    similarity = np.divide(
        intersection, union, out=np.zeros_like(intersection), where=union > 0
    )
    np.fill_diagonal(similarity, 1.0)
    return pd.DataFrame(similarity, index=rates.index, columns=rates.index)


def build_profiles(
    min_year: int = MIN_YEAR,
    max_year: int = MAX_YEAR,
    points_types: set[str] | None = None,
    min_editions: int = MIN_VOTING_EDITIONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Opportunity-normalized outgoing vote rates and their recipient-centered
    residuals, restricted to countries with enough voting editions.

    `points_types` restricts to a subset of the "Points type" column (e.g.
    just JURY_COL or TELEVOTE_COL) so the same pipeline can build jury-only
    and televote-only profiles, not just the combined one."""
    votes = load_votes(min_year=min_year, max_year=max_year)
    if points_types is not None:
        votes = votes[votes["Points type"].isin(points_types)]
    countries = all_countries(votes)
    voters = eligible_voters(votes, min_editions=min_editions)
    opportunity = build_opportunity_matrix(votes, countries)
    rates = build_rate_matrix(votes, countries).loc[voters]
    return rates, center_by_recipient(rates, opportunity.loc[voters])


def build_similarity(
    min_year: int = MIN_YEAR,
    max_year: int = MAX_YEAR,
    points_types: set[str] | None = None,
    min_editions: int = MIN_VOTING_EDITIONS,
) -> pd.DataFrame:
    _rates, centered = build_profiles(
        min_year=min_year,
        max_year=max_year,
        points_types=points_types,
        min_editions=min_editions,
    )
    return cosine_similarity_matrix(centered)


def build_split_matrices(min_year: int = SPLIT_MIN_YEAR) -> tuple[pd.DataFrame, pd.DataFrame]:
    votes = load_votes(min_year=min_year)
    countries = all_countries(votes)
    return (
        build_points_matrix(votes[votes["Points type"] == JURY_COL], countries),
        build_points_matrix(votes[votes["Points type"] == TELEVOTE_COL], countries),
    )


def plot_similarity_heatmap(
    similarity: pd.DataFrame,
    output_path: Path,
    order: list[str] | None = None,
    title: str | None = None,
) -> None:
    ordered = similarity if order is None else similarity.loc[order, order]
    limit = float(np.abs(ordered.to_numpy()[~np.eye(len(ordered), dtype=bool)]).max())
    plt.figure(figsize=(16, 14))
    ax = sns.heatmap(
        ordered,
        cmap="vlag",
        center=0.0,
        vmin=-limit,
        vmax=limit,
        square=True,
        cbar_kws={"label": "Cosine similarity of centered outgoing vote profiles"},
    )
    ax.set_xlabel("Country")
    ax.set_ylabel("Country")
    ax.set_title(
        title or f"Similarity of country voting profiles ({MIN_YEAR}–{MAX_YEAR})"
    )
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    sns.set_theme(style="whitegrid")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    votes = load_votes()
    countries = all_countries(votes)
    voters = eligible_voters(votes)
    excluded = sorted(set(countries) - set(voters))

    points = build_points_matrix(votes, countries)
    _rates, centered = build_profiles()
    similarity = cosine_similarity_matrix(centered)
    jury, televote = build_split_matrices()

    points.to_csv(POINTS_MATRIX_PATH)
    jury.to_csv(JURY_MATRIX_PATH)
    televote.to_csv(TELEVOTE_MATRIX_PATH)
    similarity.to_csv(SIMILARITY_PATH)
    plot_similarity_heatmap(similarity, HEATMAP_PATH)

    print(f"Votes {MIN_YEAR}-{MAX_YEAR}: {len(votes)} rows, {len(countries)} countries")
    print(f"Clustered voters (>= {MIN_VOTING_EDITIONS} editions): {len(voters)}")
    print(f"Excluded (too few voting editions / recipient-only): {', '.join(excluded)}")
    print(f"Jury {jury.shape} and televote {televote.shape} matrices ({SPLIT_MIN_YEAR}+)")
    print("Saved:")
    for path in (
        POINTS_MATRIX_PATH,
        JURY_MATRIX_PATH,
        TELEVOTE_MATRIX_PATH,
        SIMILARITY_PATH,
        HEATMAP_PATH,
    ):
        print(f"  - {path}")


if __name__ == "__main__":
    main()
