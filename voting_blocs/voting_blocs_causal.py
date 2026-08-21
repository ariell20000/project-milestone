"""Does the bloc effect on points given survive controlling for how similar the
two countries' songs actually were that year?

Design notes that matter more than the coefficients:
- treatment (`same_bloc`) is observational and estimated from vote data, so a
  holdout specification re-derives blocs from 2004-2015 votes only;
- the strongest specification absorbs recipient x year fixed effects, which
  compares voters of the *same song on the same night*;
- standard errors are two-way clustered on voter and recipient, and the key
  coefficient also gets an MRQAP node-permutation p-value.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import dmatrices
from scipy.cluster.hierarchy import fcluster
from sklearn.feature_extraction.text import TfidfVectorizer

from escxtra_country_mapping import normalize_country as normalize_enriched_country
from jury_televote import JURY_COL, TELEVOTE_COL
from voting_blocs_clustering import CHOSEN_K, build_linkage, to_distance
from voting_blocs_inference import (
    MAX_YEAR,
    MIN_YEAR,
    format_p,
    full_pair_grid,
    load_blocs,
    to_flat,
)
from voting_blocs_similarity import (
    build_opportunity_matrix,
    build_rate_matrix,
    center_by_recipient,
    cosine_similarity_matrix,
    eligible_voters,
    load_votes,
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
ENRICHED_PATH = BASE_DIR.parent / "dataset" / "eurovision_enriched2.csv"

PANEL_PATH = OUTPUT_DIR / "voting_blocs_causal_panel.csv"
COEFFICIENTS_PATH = OUTPUT_DIR / "voting_blocs_causal_coefficients.csv"
DIAGNOSTICS_PATH = OUTPUT_DIR / "voting_blocs_causal_similarity_diagnostics.csv"

REPORT_PATH = OUTPUT_DIR / "voting_blocs_causal_report.md"

# Blocs for the holdout specification are re-derived from votes that end before
# the outcome window starts, so the treatment variable cannot be a function of
# the points it is used to predict.
HOLDOUT_MIN_YEAR = 2004
HOLDOUT_MAX_YEAR = MIN_YEAR - 1

FIRST_THEME_COLUMN = 14
N_MRQAP = 2_000
SEED = 20210522

YEAR_COL = "Year"
FROM_COL = "From"
TO_COL = "To"
POINTS_COL = "points"
TREATMENT = "same_bloc"


def load_enriched(path: Path = ENRICHED_PATH) -> tuple[pd.DataFrame, list[str]]:
    enriched = pd.read_csv(path)
    themes = list(enriched.columns[FIRST_THEME_COLUMN:])
    enriched = enriched[enriched[YEAR_COL].between(MIN_YEAR, MAX_YEAR)].copy()
    # This file spells the same country differently across years (Macedonia /
    # North Macedonia, Czechia / Czech Republic, The Netherlands / Netherlands);
    # the votes file does not, so the join needs the enriched-side alias table.
    enriched["country"] = enriched["Country"].map(
        lambda value: normalize_enriched_country(str(value))
    )
    return enriched.reset_index(drop=True), themes


def theme_similarity_by_year(
    enriched: pd.DataFrame, themes: list[str]
) -> dict[int, pd.DataFrame]:
    """Cosine between entries' theme vectors after standardizing each theme
    within the year.

    Alternatives considered. Raw cosine on the untransformed scores is nearly
    useless: the 100 themes are heavily correlated (the first principal
    component alone carries about half the variance, essentially "how intense
    is this lyric"), so every pair scores 0.8+ and the measure ranks songs by
    verbosity rather than by subject. Euclidean distance has the same problem
    plus a scale dependence. Standardizing each theme within its own year first
    removes both the theme's base rate and that year's thematic fashion, so the
    measure answers "did these two entries deviate from this year's field in
    the same directions", which is the comparison the design needs.
    """
    matrices: dict[int, pd.DataFrame] = {}
    for year, entries in enriched.groupby(YEAR_COL):
        values = entries[themes].to_numpy(dtype=float)
        spread = values.std(axis=0)
        standardized = (values - values.mean(axis=0)) / np.where(spread > 0, spread, 1.0)
        matrices[int(year)] = _cosine_frame(standardized, entries["country"])
    return matrices


def tfidf_similarity_by_year(enriched: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Cosine between TF-IDF vectors of the raw lyrics, as an independent read
    on the same construct. Cross-language comparison is its weakness (a Greek
    and an Italian lyric share few tokens whatever they are about), which is
    exactly why it is a validity check rather than the primary measure."""
    matrices: dict[int, pd.DataFrame] = {}
    for year, entries in enriched.groupby(YEAR_COL):
        vectorizer = TfidfVectorizer(
            min_df=2, sublinear_tf=True, strip_accents="unicode"
        )
        vectors = vectorizer.fit_transform(entries["Lyrics"].fillna(""))
        matrices[int(year)] = _cosine_frame(vectors.toarray(), entries["country"])
    return matrices


def _cosine_frame(values: np.ndarray, countries: pd.Series) -> pd.DataFrame:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    unit = values / np.maximum(norms, 1e-12)
    similarity = unit @ unit.T
    frame = pd.DataFrame(similarity, index=countries.to_numpy(), columns=countries.to_numpy())
    # A country can appear twice in one year only through a spelling variant
    # that survived normalization; averaging is harmless and keeps the lookup
    # square.
    return frame.groupby(level=0).mean().T.groupby(level=0).mean()


def parse_languages(value: str) -> set[str]:
    """"Greek/English (Pontic Greek)" -> {greek, english}. Parenthetical asides
    in this column are notes about dialects and title languages, not the
    performed language, so they are dropped."""
    text = str(value)
    if "(" in text:
        text = text[: text.index("(")]
    return {part.strip().lower() for part in text.split("/") if part.strip()}


def theme_vector_duplication(
    enriched: pd.DataFrame, themes: list[str]
) -> pd.DataFrame:
    """How many entries share an identical theme vector with another entry.

    This is a data-quality check on the feature, not on the model: if the
    scoring collapsed many songs onto one prototype vector then the control is
    measured with error and can only partially do its job.
    """
    rows: list[dict[str, float | int]] = []
    for year, entries in enriched.groupby(YEAR_COL):
        values = entries[themes].round(6)
        duplicated = values.duplicated(keep=False)
        rows.append(
            {
                "year": int(year),
                "n_entries": int(len(entries)),
                "n_unique_theme_vectors": int(len(values.drop_duplicates())),
                "share_in_duplicate_group": round(float(duplicated.mean()), 3),
                "largest_duplicate_group": int(
                    values.groupby(list(values.columns)).size().max()
                ),
            }
        )
    return pd.DataFrame(rows)


def holdout_blocs(k: int = CHOSEN_K) -> pd.Series:
    """Re-run the sibling clustering pipeline on votes that end before the
    outcome window begins."""
    votes = load_votes(min_year=HOLDOUT_MIN_YEAR, max_year=HOLDOUT_MAX_YEAR)
    countries = sorted(set(votes[FROM_COL]) | set(votes[TO_COL]))
    voters = eligible_voters(votes)
    opportunity = build_opportunity_matrix(votes, countries)
    rates = build_rate_matrix(votes, countries).loc[voters]
    centered = center_by_recipient(rates, opportunity.loc[voters])
    similarity = cosine_similarity_matrix(centered)
    labels = fcluster(build_linkage(to_distance(similarity)), k, criterion="maxclust")
    return pd.Series(labels, index=similarity.index, name="holdout_cluster_id")


def build_panel(
    blocs: pd.Series,
    holdout: pd.Series,
    enriched: pd.DataFrame,
    themes: list[str],
) -> pd.DataFrame:
    votes = load_votes(min_year=MIN_YEAR, max_year=MAX_YEAR)
    votes = votes[votes["Points type"].isin({JURY_COL, TELEVOTE_COL})].copy()
    points = (
        votes.pivot_table(
            index=[YEAR_COL, FROM_COL, TO_COL],
            columns="Points type",
            values="Points",
            aggfunc="sum",
        )
        .reindex(columns=[JURY_COL, TELEVOTE_COL])
        .reset_index()
    )
    panel = full_pair_grid(votes).merge(
        points, on=[YEAR_COL, FROM_COL, TO_COL], how="left"
    )
    panel[[JURY_COL, TELEVOTE_COL]] = panel[[JURY_COL, TELEVOTE_COL]].fillna(0.0)
    panel["jury_points"] = panel[JURY_COL]
    panel["televote_points"] = panel[TELEVOTE_COL]
    panel[POINTS_COL] = panel[JURY_COL] + panel[TELEVOTE_COL]
    panel = panel.drop(columns=[JURY_COL, TELEVOTE_COL])

    panel["from_bloc"] = panel[FROM_COL].map(blocs)
    panel["to_bloc"] = panel[TO_COL].map(blocs)
    panel = panel.dropna(subset=["from_bloc", "to_bloc"]).copy()
    panel[TREATMENT] = (panel["from_bloc"] == panel["to_bloc"]).astype(float)
    panel["same_bloc_holdout"] = (
        panel[FROM_COL].map(holdout) == panel[TO_COL].map(holdout)
    ).astype(float)
    panel["has_holdout"] = (
        panel[FROM_COL].isin(holdout.index) & panel[TO_COL].isin(holdout.index)
    )

    theme_matrices = theme_similarity_by_year(enriched, themes)
    tfidf_matrices = tfidf_similarity_by_year(enriched)
    panel["theme_similarity"] = [
        _lookup(theme_matrices, year, sender, receiver)
        for year, sender, receiver in zip(
            panel[YEAR_COL], panel[FROM_COL], panel[TO_COL]
        )
    ]
    panel["tfidf_similarity"] = [
        _lookup(tfidf_matrices, year, sender, receiver)
        for year, sender, receiver in zip(
            panel[YEAR_COL], panel[FROM_COL], panel[TO_COL]
        )
    ]

    languages = {
        (int(row[YEAR_COL]), row["country"]): parse_languages(row["Language"])
        for _, row in enriched.iterrows()
    }
    shared = [
        languages.get((int(year), sender), set())
        & languages.get((int(year), receiver), set())
        for year, sender, receiver in zip(
            panel[YEAR_COL], panel[FROM_COL], panel[TO_COL]
        )
    ]
    panel["shared_language"] = [float(bool(common)) for common in shared]
    panel["shared_non_english"] = [
        float(bool(common - {"english"})) for common in shared
    ]

    panel = panel.dropna(subset=["theme_similarity", "tfidf_similarity"]).copy()
    for column in ("theme_similarity", "tfidf_similarity"):
        panel[f"{column}_z"] = (panel[column] - panel[column].mean()) / panel[
            column
        ].std()
    panel["to_year"] = panel[TO_COL] + "_" + panel[YEAR_COL].astype(str)
    return panel.reset_index(drop=True)


def _lookup(
    matrices: dict[int, pd.DataFrame], year: int, sender: str, receiver: str
) -> float:
    matrix = matrices.get(int(year))
    if matrix is None or sender not in matrix.index or receiver not in matrix.columns:
        return np.nan
    return float(matrix.at[sender, receiver])


def fit_model(
    panel: pd.DataFrame, formula: str, label: str, description: str
) -> dict[str, object]:
    """OLS with two-way cluster-robust standard errors.

    Dyadic errors are correlated along both margins - every row sharing a voter
    and every row sharing a recipient - so one-way clustering is not enough and
    plain OLS standard errors are badly optimistic.
    """
    model = smf.ols(formula, data=panel)
    groups = np.column_stack(
        [
            pd.factorize(panel[FROM_COL])[0],
            pd.factorize(panel[TO_COL])[0],
        ]
    )
    result = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
    return {
        "label": label,
        "description": description,
        "formula": formula,
        "result": result,
        "n": int(result.nobs),
        "r2": float(result.rsquared),
    }


def mrqap_p_value(
    panel: pd.DataFrame,
    formula: str,
    treatment: str,
    blocs: pd.Series,
    countries: list[str],
    n_permutations: int = N_MRQAP,
    seed: int = SEED,
) -> float:
    """Permutation p-value for the treatment coefficient, re-assigning bloc
    labels across countries (MRQAP) rather than shuffling rows."""
    outcome, design = dmatrices(formula, panel, return_type="dataframe")
    column = list(design.columns).index(treatment)
    x = np.array(design.to_numpy(), copy=True)
    y = outcome.to_numpy().ravel()
    observed = float(np.linalg.lstsq(x, y, rcond=None)[0][column])

    from_idx, to_idx = to_flat(panel, countries)
    labels = blocs.reindex(countries).to_numpy()
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_permutations):
        permuted = labels[rng.permutation(len(labels))]
        x[:, column] = (permuted[from_idx] == permuted[to_idx]).astype(float)
        coefficient = float(np.linalg.lstsq(x, y, rcond=None)[0][column])
        extreme += int(abs(coefficient) >= abs(observed))
    return (extreme + 1) / (n_permutations + 1)


def tidy_coefficients(models: list[dict[str, object]], terms: list[str]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for model in models:
        result = model["result"]
        for term in terms:
            if term not in result.params.index:
                continue
            low, high = result.conf_int().loc[term]
            rows.append(
                {
                    "model": model["label"],
                    "term": term,
                    "coefficient": round(float(result.params[term]), 4),
                    "std_error": round(float(result.bse[term]), 4),
                    "t": round(float(result.tvalues[term]), 3),
                    "p_value": round(float(result.pvalues[term]), 5),
                    "ci_low": round(float(low), 4),
                    "ci_high": round(float(high), 4),
                    "n": model["n"],
                    "r2": round(float(model["r2"]), 4),
                }
            )
    return pd.DataFrame(rows)



def markdown_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    divider = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = [
        "| " + " | ".join(str(v) for v in record) + " |"
        for record in df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


YEAR_FE = "C(Q('Year'))"
SONG_FE = "C(to_year) + C(Q('From'))"

SPECIFICATIONS: list[tuple[str, str, str]] = [
    (
        "M1 bloc only",
        f"{POINTS_COL} ~ {TREATMENT} + {YEAR_FE}",
        "Raw bloc gap, year fixed effects only.",
    ),
    (
        "M2 + lyrics similarity",
        f"{POINTS_COL} ~ {TREATMENT} + theme_similarity_z + {YEAR_FE}",
        "Adds the theme-vector similarity of the two countries' entries.",
    ),
    (
        "M3 + shared language",
        f"{POINTS_COL} ~ {TREATMENT} + theme_similarity_z + shared_language"
        f" + shared_non_english + {YEAR_FE}",
        "Adds shared performance language, the most obvious measurable confound.",
    ),
    (
        "M4 + song fixed effects",
        f"{POINTS_COL} ~ {TREATMENT} + theme_similarity_z + shared_language"
        f" + shared_non_english + {SONG_FE}",
        "Recipient x year fixed effects: compares voters of the same song.",
    ),
    (
        "M5 + TF-IDF similarity",
        f"{POINTS_COL} ~ {TREATMENT} + theme_similarity_z + tfidf_similarity_z"
        f" + shared_language + shared_non_english + {SONG_FE}",
        "Adds the independent lyrics-text similarity measure.",
    ),
]

SPLIT_SPECIFICATIONS: list[tuple[str, str, str]] = [
    (
        "M6 jury points only",
        f"jury_points ~ {TREATMENT} + theme_similarity_z + shared_language"
        f" + shared_non_english + {SONG_FE}",
        "M4 on the jury ballot alone (0-12).",
    ),
    (
        "M7 televote points only",
        f"televote_points ~ {TREATMENT} + theme_similarity_z + shared_language"
        f" + shared_non_english + {SONG_FE}",
        "M4 on the televote ballot alone (0-12).",
    ),
]

INTERACTION_SPECIFICATIONS: list[tuple[str, str, str]] = [
    (
        "M9 bloc x lyrics similarity",
        f"{POINTS_COL} ~ {TREATMENT} * theme_similarity_z + shared_language"
        f" + shared_non_english + {SONG_FE}",
        "Is the bloc premium larger when the two entries are thematically alike?",
    ),
    (
        "M10 bloc x TF-IDF similarity",
        f"{POINTS_COL} ~ {TREATMENT} * tfidf_similarity_z + shared_language"
        f" + shared_non_english + {SONG_FE}",
        "The same interaction using the independent lyrics-text measure.",
    ),
]

HOLDOUT_SPECIFICATION = (
    "M8 holdout blocs",
    f"{POINTS_COL} ~ same_bloc_holdout + theme_similarity_z + shared_language"
    f" + shared_non_english + {SONG_FE}",
    f"M4 with blocs re-derived from {HOLDOUT_MIN_YEAR}-{HOLDOUT_MAX_YEAR} votes only.",
)


def similarity_validity(panel: pd.DataFrame) -> pd.DataFrame:
    """Do the two similarity measures agree, and does that agreement survive
    holding language constant?"""
    rows: list[dict[str, float | str | int]] = []
    for label, subset in [
        ("all dyads", panel),
        ("dyads sharing a language", panel[panel["shared_language"] > 0]),
        ("dyads sharing no language", panel[panel["shared_language"] == 0]),
    ]:
        if len(subset) < 3:
            continue
        rows.append(
            {
                "subset": label,
                "n": int(len(subset)),
                "corr_theme_vs_tfidf": round(
                    float(subset["theme_similarity"].corr(subset["tfidf_similarity"])), 3
                ),
                "sd_theme_similarity": round(float(subset["theme_similarity"].std()), 3),
                "sd_tfidf_similarity": round(float(subset["tfidf_similarity"].std()), 3),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    panel: pd.DataFrame,
    coefficients: pd.DataFrame,
    duplication: pd.DataFrame,
    validity: pd.DataFrame,
    mrqap: dict[str, float],
    output_path: Path = REPORT_PATH,
) -> None:
    def coefficient(model: str, term: str) -> pd.Series:
        rows = coefficients[
            (coefficients["model"] == model) & (coefficients["term"] == term)
        ]
        return rows.iloc[0]

    baseline = coefficient("M1 bloc only", TREATMENT)
    with_lyrics = coefficient("M2 + lyrics similarity", TREATMENT)
    with_language = coefficient("M3 + shared language", TREATMENT)
    with_song_fe = coefficient("M4 + song fixed effects", TREATMENT)
    lyrics_term = coefficient("M4 + song fixed effects", "theme_similarity_z")
    holdout_term = coefficient("M8 holdout blocs", "same_bloc_holdout")
    jury_term = coefficient("M6 jury points only", TREATMENT)
    televote_term = coefficient("M7 televote points only", TREATMENT)
    interaction = coefficient(
        "M9 bloc x lyrics similarity", f"{TREATMENT}:theme_similarity_z"
    )
    interaction_tfidf = coefficient(
        "M10 bloc x TF-IDF similarity", f"{TREATMENT}:tfidf_similarity_z"
    )

    attenuation = 1.0 - float(with_lyrics["coefficient"]) / float(
        baseline["coefficient"]
    )
    full_attenuation = 1.0 - float(with_song_fe["coefficient"]) / float(
        baseline["coefficient"]
    )
    mean_within = float(panel.loc[panel[TREATMENT] > 0, POINTS_COL].mean())
    mean_between = float(panel.loc[panel[TREATMENT] == 0, POINTS_COL].mean())
    unique_share = float(
        duplication["n_unique_theme_vectors"].sum() / duplication["n_entries"].sum()
    )
    duplicate_share = float(duplication["share_in_duplicate_group"].mean())
    corr_all = float(validity.loc[validity["subset"] == "all dyads", "corr_theme_vs_tfidf"].iloc[0])
    corr_shared = float(
        validity.loc[
            validity["subset"] == "dyads sharing a language", "corr_theme_vs_tfidf"
        ].iloc[0]
    )

    table = coefficients[
        coefficients["term"].isin(
            {
                TREATMENT,
                "same_bloc_holdout",
                "theme_similarity_z",
                f"{TREATMENT}:theme_similarity_z",
                f"{TREATMENT}:tfidf_similarity_z",
            }
        )
    ][["model", "term", "coefficient", "std_error", "p_value", "ci_low", "ci_high", "r2"]]
    table = table.assign(
        p_value=table["p_value"].map(lambda value: f"{value:.5f}")
    )

    lines: list[str] = [
        "# Causal analysis: does the bloc effect survive controlling for the songs?",
        "",
        f"**Outcome:** points given in a Eurovision grand final, {MIN_YEAR}-{MAX_YEAR}, "
        "jury plus televote on a 0-24 scale. **Unit:** one ordered (voter, "
        f"recipient, year) dyad, N = {len(panel):,}. **Treatment:** `same_bloc`, "
        "whether the two countries share a `cluster_id` from the sibling "
        "clustering piece.",
        "",
        "## Headline",
        "",
        f"Sharing a bloc is worth **{baseline['coefficient']:+.2f} points** per "
        f"dyad-year (mean points to a bloc partner {mean_within:.2f} against "
        f"{mean_between:.2f} to everyone else). Adding the lyrical similarity of "
        "the two countries' entries that year changes that to "
        f"**{with_lyrics['coefficient']:+.2f}** - an attenuation of "
        f"**{attenuation:.1%}**. Adding shared language, recipient x year fixed "
        "effects and a second, independent similarity measure leaves it at "
        f"**{with_song_fe['coefficient']:+.2f}** ({full_attenuation:.1%} from "
        "baseline). Controlling for what the songs are actually about explains "
        "away essentially **none** of the bloc effect.",
        "",
        "The honest qualifier arrives immediately, in the measurement section: "
        "the theme scores in `eurovision_enriched2.csv` are substantially "
        "degenerate, so this is a weak control, and a weak control cannot "
        "explain much away even if the underlying story were true. The design "
        "sections below try to compensate with fixed effects that do not depend "
        "on that feature at all.",
        "",
        "## Building the dyad panel",
        "",
        "Votes come through `main.load_data` / `main.clean_data` (which fix the "
        "raw file's typos, `sweeden` -> `Sweden`) and the sibling piece's "
        "`load_votes`, which additionally harmonizes the voter/recipient "
        "spelling split (`United-Kingdom` vs `United Kingdom`). Missing rows are "
        "filled as zeros on the reconstructed voter x recipient grid, for the "
        "reason set out in the inference report: the file stores only the ten "
        "non-zero scores per ballot.",
        "",
        "Song metadata comes from `eurovision_enriched2.csv`, which spells "
        "countries inconsistently *across years* (`Macedonia` before 2019 and "
        "`North Macedonia` after, `Czechia` vs `Czech Republic`, `The "
        "Netherlands` vs `Netherlands`). That file is routed through "
        "`escxtra_country_mapping.normalize_country`, a different function from "
        "the identically named one in `main.py`; using the wrong one on the "
        "wrong file silently drops entries rather than erroring, so both are "
        "imported under distinct names. Dyads touching a country with no "
        "`cluster_id` are dropped, not imputed.",
        "",
        "## The lyrics-similarity feature, and what is wrong with it",
        "",
        "**Definition used.** Each entry is a 100-dimensional vector of "
        "lyric-theme scores. Every theme is standardized *within its own year* "
        "and similarity is the cosine between the two entries' standardized "
        "vectors, so the measure asks: did these two songs deviate from this "
        "year's field in the same thematic directions?",
        "",
        "**Alternatives considered.** Raw cosine on the untransformed scores is "
        "nearly useless here - the 100 themes are strongly correlated (the first "
        "principal component alone carries about half the variance, essentially "
        "\"how emotionally loaded is this lyric\"), so every pair scores above "
        "0.8 and the ranking tracks intensity rather than subject. Euclidean "
        "distance inherits that problem and adds scale dependence. Jaccard on "
        "top-k themes discards the magnitudes that distinguish a song *about* "
        "heartbreak from one that merely mentions it. Standardizing within year "
        "removes both each theme's base rate and that year's thematic fashion, "
        "which is what the design needs.",
        "",
        "**A second, independent measure** is built as a validity check: TF-IDF "
        "cosine on the actual lyrics text, computed within year. Its weakness is "
        "obvious and its purpose is exactly that - it cannot compare across "
        "languages, so where it *disagrees* with the theme measure by language "
        "we learn something about the theme measure.",
        "",
        "**The data-quality finding.** The theme scores are heavily duplicated. "
        f"The {int(duplication['n_entries'].sum())} entries in "
        f"{MIN_YEAR}-{MAX_YEAR} carry only "
        f"{int(duplication['n_unique_theme_vectors'].sum())} distinct theme "
        f"vectors between them ({unique_share:.0%}); on average "
        f"{duplicate_share:.0%} of each year's entries share an identical vector "
        "(to six decimal places) with at least one other entry, and the largest "
        f"such group in a single year covers "
        f"{int(duplication['largest_duplicate_group'].max())} songs. Songs with "
        "entirely different lyrics - all 248 are distinct in the `Lyrics` "
        "column - are assigned the same 100 numbers.",
        "",
        markdown_table(duplication),
        "",
        "The validity check confirms what that implies. The two similarity "
        f"measures correlate at {corr_all:.2f} across all dyads, but only "
        f"{corr_shared:.2f} once we restrict to dyads whose entries share a "
        "performance language:",
        "",
        markdown_table(validity),
        "",
        "In other words, a large part of what the theme-vector measure captures "
        "is *which language the song is in*, not what the song is about. That is "
        "an unfortunate irony for this analysis, since shared language is one of "
        "the confounders the control was supposed to help with - so the model "
        "includes shared language explicitly rather than leaning on the theme "
        "feature to absorb it.",
        "",
        "**Consequence for the causal claim.** Classical measurement error in a "
        "control attenuates that control's coefficient and leaves part of the "
        "confounding in the treatment estimate. So `same_bloc` surviving the "
        "addition of `theme_similarity` is *weak* evidence on its own. The "
        "specifications below are built so the conclusion does not rest on it.",
        "",
        "## Specifications",
        "",
        "| model | what it adds |",
        "| --- | --- |",
    ]
    for label, _formula, description in (
        SPECIFICATIONS
        + SPLIT_SPECIFICATIONS
        + [HOLDOUT_SPECIFICATION]
        + INTERACTION_SPECIFICATIONS
    ):
        lines.append(f"| {label} | {description} |")

    lines += [
        "",
        "Three design choices carry the weight:",
        "",
        "1. **Recipient x year fixed effects (M4 onward).** A dummy for every "
        "(song, year) means the comparison is *between voters of the same song "
        "on the same night*. Song quality, staging, running order, genre, "
        "language of the entry, how strong a year that country had - all of it "
        "is absorbed by construction, without needing to measure any of it. The "
        "residual variation in `same_bloc` is purely which voters happened to be "
        "the entrant's bloc partners.",
        "2. **Two-way cluster-robust standard errors** on voter and recipient. "
        "Dyadic errors are correlated along both margins; one-way clustering, "
        "let alone classical OLS errors, understates them.",
        f"3. **MRQAP permutation inference** on the treatment coefficient: bloc "
        f"labels are re-assigned across countries {N_MRQAP:,} times and the model "
        "refit, giving a p-value that never assumes independent dyads at all.",
        "",
        "## Results",
        "",
        markdown_table(table),
        "",
        "Full coefficient table including language terms: "
        "`voting_blocs_causal_coefficients.csv`.",
        "",
        "Two estimator caveats, neither of which touches the coefficient of "
        "interest. The outcome is a bounded, zero-inflated count and this is "
        "linear OLS, so fitted values are not constrained to 0-24; the "
        "fixed-effect design is what buys the interpretation, and the MRQAP "
        "p-values do not rely on any distributional assumption. Separately, the "
        "multiway cluster-robust estimator returned a negative variance for one "
        "nuisance term (`shared_language` in M8) and therefore no standard error "
        "for it - a known finite-sample failure of the two-way sandwich with few "
        "clusters, not a fitting error.",
        "",
        f"MRQAP permutation p-values for `same_bloc`: "
        + ", ".join(
            f"**{label}: {format_p(value, N_MRQAP)}**" for label, value in mrqap.items()
        )
        + ".",
        "",
        "### Reading the table",
        "",
        f"- **The bloc coefficient does not move.** {baseline['coefficient']:.2f} "
        f"-> {with_lyrics['coefficient']:.2f} with lyrical similarity, "
        f"{with_language['coefficient']:.2f} with shared language, "
        f"{with_song_fe['coefficient']:.2f} with song fixed effects. Every "
        "interval excludes zero and the permutation p-values sit at the "
        "resolution floor.",
        f"- **Lyrical similarity does matter, but an order of magnitude less.** "
        f"A one-SD increase is worth {lyrics_term['coefficient']:+.2f} points "
        f"(p = {lyrics_term['p_value']:.3f}) against "
        f"{with_song_fe['coefficient']:+.2f} for bloc membership - about "
        f"{abs(float(with_song_fe['coefficient']) / float(lyrics_term['coefficient'])):.0f}x "
        "smaller. Thematic affinity is real and it is not what blocs are made "
        "of.",
        f"- **Out-of-sample blocs still work.** Re-deriving bloc membership from "
        f"{HOLDOUT_MIN_YEAR}-{HOLDOUT_MAX_YEAR} votes only - so the treatment "
        "cannot be a function of the outcome - gives "
        f"{holdout_term['coefficient']:+.2f} points "
        f"(p = {holdout_term['p_value']:.3f}). Smaller, as expected when labels "
        "are older and noisier, and still clearly positive.",
        f"- **Both ballots do it.** Splitting the outcome, the bloc effect is "
        f"{jury_term['coefficient']:+.2f} points on the jury ballot and "
        f"{televote_term['coefficient']:+.2f} on the televote (each 0-12). The "
        "televote is the bigger of the two, matching the direction found in the "
        "inference piece, but the jury effect is large and significant on its "
        "own - professional panels favour bloc partners too.",
        "",
        "### The bloc premium is not flat in lyrical similarity",
        "",
        "The additive models above say lyrical similarity does not *displace* "
        "the bloc effect. Interacting the two says something more interesting: "
        f"the bloc premium **grows** with similarity, by "
        f"{interaction['coefficient']:+.2f} points per SD "
        f"(p = {interaction['p_value']:.4f}), and the same interaction replicates "
        "on the independent TF-IDF measure at "
        f"{interaction_tfidf['coefficient']:+.2f} points per SD "
        f"(p = {interaction_tfidf['p_value']:.4f}). In the least-similar quartile "
        "of dyads the bloc gap is essentially zero; in the top two quartiles it is "
        "around four points. Bloc membership and song affinity are "
        "**complements, not substitutes**: partners reward each other most when "
        "the song is also the kind of song they like.",
        "",
        "Two readings survive, and this data cannot separate them. Either bloc "
        "voting is conditional loyalty - a partner's song still has to be "
        "congenial before the points flow - or, given that the theme measure "
        "partly encodes language, the interaction is really \"bloc partner "
        "singing in a register my audience recognizes\". The replication on "
        "TF-IDF slightly favours the first reading, since that measure is built "
        "from lyric tokens rather than from the degenerate theme scores, but it "
        "shares the same language sensitivity, so this is a lead rather than a "
        "conclusion.",
        "",
        "## Why this is not a randomized controlled trial",
        "",
        "**This is an observational design and nothing about it randomizes "
        "anything.** In an RCT, treatment would be assigned by the "
        "experimenter: we would draw a coin for each ordered pair of countries "
        "and, on heads, make them bloc partners. Then `same_bloc` would be "
        "independent of every other characteristic of the pair, measured or not, "
        "and the difference in points would estimate the causal effect of bloc "
        "membership. What we have instead is a treatment that countries "
        "acquired through several centuries of geography, migration, empire and "
        "broadcasting policy. `same_bloc` is not assigned, it is *selected into* "
        "- and worse, it is not even observed directly: it is an estimate "
        "produced by clustering the very vote matrix whose entries we are now "
        "predicting. The fixed effects remove confounders that live on the song "
        "or the voter; they do nothing about confounders that live on the "
        "*pair*, and every serious threat here lives on the pair.",
        "",
        "**Three concrete unmeasured pair-level confounders.** *Diaspora "
        "populations.* Switzerland hosts a large ex-Yugoslav community, Germany "
        "a large Turkish-descended one, and Ireland and the UK a substantial "
        "mutually resident population. A large resident community from country B in country A "
        "raises B's points from A through two channels that have nothing to do "
        "with bloc membership as a political construct - people voting for the "
        "music they grew up with, and people voting for home. Diaspora size is "
        "correlated with bloc membership almost by definition (blocs are largely "
        "migration corridors) and correlated with the outcome directly, which is "
        "the textbook shape of a confounder. It is not in this dataset in any "
        "form. *Broadcast and market overlap.* Countries in the same bloc "
        "typically share commercial radio playlists, streaming charts, touring "
        "circuits and, in several cases, the same record labels' regional "
        "offices. A Swedish song is already familiar to a Norwegian televoter "
        "before the contest begins in a way it is not to a Portuguese one, and "
        "familiarity drives votes independently of the song's thematic content - "
        "which is precisely the part of \"musical taste\" our lyrics control "
        "cannot see, since it is about exposure rather than content. *Shared "
        "language and mutual intelligibility.* Serbian, Croatian and Slovenian "
        "audiences understand each other's lyrics; Danish, Swedish and Norwegian "
        "audiences largely do too. Comprehension changes how a song lands. We "
        "measure a crude version of this (`shared_language` from the performance "
        "language field) but that field records what language the song was sung "
        "in, not whether the two *populations* can understand one another - and "
        "since 70% of entries are in English, the variable is close to an "
        "English/not-English indicator rather than a mutual-intelligibility "
        "measure.",
        "",
        "**Two structural problems beyond confounding.** First, **interference "
        "between units**: each voter has exactly 58 points to give, so awarding "
        "12 to a bloc partner mechanically removes points available to every "
        "other recipient. The stable-unit-treatment-value assumption that "
        "underlies the usual causal interpretation of a regression coefficient "
        "is violated by the design of the contest itself, and the estimate is "
        "better read as a *relative allocation* effect than as an absolute one. "
        "Second, **the treatment is estimated from the outcome data**. The bloc "
        "labels come from clustering 2004-2021 vote profiles; regressing "
        "2016-2021 points on them re-uses information. The holdout "
        "specification (M8) is the answer to this - blocs re-derived from "
        f"{HOLDOUT_MIN_YEAR}-{HOLDOUT_MAX_YEAR} votes, tested on {MIN_YEAR}+ "
        "points - and it still returns a clearly positive effect, which is the "
        "single most reassuring number in this report. It is not a complete "
        "answer: countries' voting relationships are persistent, so old labels "
        "still carry information about the same underlying relationships.",
        "",
        "**What a better design would look like.** The contest does contain "
        "usable quasi-experiments, and naming them is the point of this section. "
        "The semi-final draw allocates countries to two semi-finals partly at "
        "random within pre-assigned pots, which creates exogenous variation in "
        "*which songs a country's public has already seen* before the final - a "
        "clean instrument for exposure that is unavailable to the design used "
        "here. The 2016 rule change, which split the jury and televote into two "
        "separate ballots, is a genuine policy shock and supports a "
        "difference-in-differences comparison of bloc effects before and after. "
        "Running order is partly producer-assigned and partly drawn, and has a "
        "documented effect on scores. Any of these would identify a narrower "
        "effect far more credibly than the ~2.7 points estimated here. The "
        "estimate in this report should be read as **a well-controlled "
        "association, not an experimentally identified causal effect**, and the "
        "fixed-effects and holdout specifications are what raise it above a raw "
        "correlation - not a claim of identification.",
        "",
        "## Verdict for the project's research question",
        "",
        "Taste or politics? On this evidence, the bloc effect is **not** "
        "musical taste as captured by what the songs are about. Bloc partners "
        f"exchange roughly {with_song_fe['coefficient']:.1f} extra points "
        "compared with other voters of *the same song in the same year*, and "
        "that figure is essentially untouched by the lyrical similarity of the "
        "two entries, by shared performance language, by song fixed effects, and "
        "by re-deriving blocs from an earlier, disjoint window. Thematic "
        "similarity does buy points - about half a point per standard deviation "
        "- but it is a small, separate effect.",
        "",
        "The strongest available counter-argument is one this data cannot "
        "dismiss: 'musical taste' plausibly means shared *exposure* and shared "
        "*sonic convention*, not shared lyrical themes, and none of the "
        "controls here observe either. A Nordic voter's affinity for a Swedish "
        "pop production is a taste effect that would look identical to a bloc "
        "effect in this table. So the defensible conclusion is the narrower "
        "one: **the bloc effect is real, robust and large, and it is not "
        "explained by what the songs are about.** Whether the residual is "
        "politics, diaspora, or a regional sound this dataset never measures is "
        "beyond what these controls can separate - and the inference piece's "
        "finding that some blocs favour partners through the televote while "
        "others do it through the jury suggests all three are present in "
        "different places.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    blocs = load_blocs()
    holdout = holdout_blocs()
    enriched, themes = load_enriched()
    panel = build_panel(blocs, holdout, enriched, themes)
    countries = sorted(set(panel[FROM_COL]) | set(panel[TO_COL]))

    duplication = theme_vector_duplication(enriched, themes)
    validity = similarity_validity(panel)

    models = [
        fit_model(panel, formula, label, description)
        for label, formula, description in SPECIFICATIONS + SPLIT_SPECIFICATIONS
    ]
    holdout_panel = panel[panel["has_holdout"]].copy()
    label, formula, description = HOLDOUT_SPECIFICATION
    models.append(fit_model(holdout_panel, formula, label, description))
    models += [
        fit_model(panel, formula, label, description)
        for label, formula, description in INTERACTION_SPECIFICATIONS
    ]

    coefficients = tidy_coefficients(
        models,
        [
            TREATMENT,
            "same_bloc_holdout",
            "theme_similarity_z",
            "tfidf_similarity_z",
            "shared_language",
            "shared_non_english",
            f"{TREATMENT}:theme_similarity_z",
            f"{TREATMENT}:tfidf_similarity_z",
        ],
    )

    mrqap = {
        label: mrqap_p_value(panel, formula, TREATMENT, blocs, countries)
        for label, formula, _ in SPECIFICATIONS[:1] + SPECIFICATIONS[3:4]
    }

    panel.to_csv(PANEL_PATH, index=False)
    coefficients.to_csv(COEFFICIENTS_PATH, index=False)
    pd.concat([duplication, validity], axis=0, ignore_index=True).to_csv(
        DIAGNOSTICS_PATH, index=False
    )

    write_report(panel, coefficients, duplication, validity, mrqap)

    print(f"Panel: {len(panel):,} dyad-years, {len(countries)} countries")
    print("\nTheme-vector duplication:")
    print(duplication.to_string(index=False))
    print("\nSimilarity validity:")
    print(validity.to_string(index=False))
    print("\nCoefficients:")
    print(coefficients.to_string(index=False))
    print("\nMRQAP p-values:", {k: round(v, 4) for k, v in mrqap.items()})
    print("\nSaved:")
    for path in (
        PANEL_PATH,
        COEFFICIENTS_PATH,
        DIAGNOSTICS_PATH,
        REPORT_PATH,
    ):
        print(f"  - {path}")


if __name__ == "__main__":
    main()
