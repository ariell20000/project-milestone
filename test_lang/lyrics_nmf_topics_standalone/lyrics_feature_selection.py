"""Feature selection over the 100 lyrics semantic-theme scores in
eurovision_enriched1.csv, ahead of feeding them into score_model/pairwise_vote_model.

Stage 1 (redundancy): cluster features whose pairwise |correlation| exceeds
REDUNDANCY_THRESHOLD (hierarchical clustering on the 1-|corr| distance, not a
naive order-dependent greedy pass) and keep only the cluster member with the
strongest |correlation| to the target - no averaging, so inference still only
has to extract the surviving features.

Stage 2 (relevance): drop whatever's left whose |correlation| with the target
is below RELEVANCE_THRESHOLD.

Both stages run separately inside every training fold of a GroupKFold grouped
by Year (never on the full dataset), and a feature only survives if it clears
both bars in at least MIN_FOLD_SURVIVAL of the folds - a single lucky/unlucky
year split can't decide the final list. If too many features still survive,
the thresholds tighten automatically and the whole thing reruns.

Interaction search (stage 3) is intentionally not here yet - it should run on
whatever this script keeps, once we see how many that is.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.model_selection import GroupKFold

from score_model import COUNTRY_COL, TARGET_COL, clean_text


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "eurovision_enriched1.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_TARGET_COL = "Score_norm"

FEATURE_COLS = [
    "love",
    "hope",
    "heartbreak",
    "party / celebration",
    "war or conflict",
    "nostalgia",
    "empowerment",
    "loneliness",
    "nature",
    "spirituality",
    "joy / happiness",
    "grief / mourning",
    "anger / rage",
    "fear / anxiety",
    "jealousy",
    "longing / yearning",
    "regret",
    "gratitude",
    "pride",
    "shame / guilt",
    "desperation",
    "relief",
    "excitement / thrill",
    "numbness / emptiness",
    "confusion",
    "bitterness",
    "tenderness",
    "euphoria",
    "melancholy",
    "defiance",
    "friendship",
    "betrayal",
    "family",
    "unrequited love",
    "reunion",
    "breakup",
    "flirtation / seduction",
    "commitment / devotion",
    "infidelity",
    "long-distance relationship",
    "forgiveness",
    "jealous ex-lover",
    "new beginnings in love",
    "codependency",
    "heartbreak recovery",
    "freedom",
    "unity / solidarity",
    "identity",
    "LGBTQ+ pride",
    "feminism / women's empowerment",
    "immigration / displacement",
    "social justice",
    "protest / rebellion",
    "patriotism / national pride",
    "peace",
    "human rights",
    "equality",
    "discrimination / prejudice",
    "war refugee experience",
    "revolution",
    "self-discovery",
    "self-acceptance",
    "mental health struggle",
    "resilience / perseverance",
    "ambition / dreams",
    "escapism",
    "personal transformation",
    "coming of age",
    "individuality",
    "vulnerability",
    "inner strength",
    "self-doubt",
    "healing",
    "courage",
    "authenticity",
    "dance",
    "nightlife",
    "alcohol / drinking",
    "fame / stardom",
    "wealth / materialism",
    "hedonism",
    "youth",
    "summer",
    "glamour",
    "freedom through music",
    "the ocean / sea",
    "stars / sky",
    "fire",
    "water",
    "seasons changing",
    "death / mortality",
    "afterlife",
    "fate / destiny",
    "magic / mysticism",
    "light versus darkness",
    "chaos versus order",
    "time passing",
    "dreams (sleep / visions)",
    "storytelling / fairy tale",
    "music as escape / power of song",
]

N_FOLDS = 5
REDUNDANCY_THRESHOLD = 0.8
RELEVANCE_THRESHOLD = 0.05
MIN_FOLD_SURVIVAL = 0.6
TARGET_MAX_FEATURES = 25
MAX_TIGHTEN_ITERS = 6
TIGHTEN_STEP = 0.05


def load_and_clean_data(csv_path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.columns[0] == "":
        df = df.drop(columns=df.columns[0])

    df = df.copy()
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df[COUNTRY_COL] = df[COUNTRY_COL].map(clean_text)
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[TARGET_COL, "Year", *FEATURE_COLS])
    df = df[df[COUNTRY_COL] != ""].copy()

    # Same reasoning as score_model.load_and_clean_data: raw Score isn't
    # comparable across years, so the target is each song's share of its
    # own year's total.
    year_totals = df.groupby("Year")[TARGET_COL].transform("sum")
    df[MODEL_TARGET_COL] = df[TARGET_COL] / year_totals

    df = df.reset_index(drop=True)
    return df


def cluster_redundant_features(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    threshold: float = REDUNDANCY_THRESHOLD,
) -> list[list[str]]:
    """Group features whose pairwise |correlation| exceeds `threshold` into
    clusters via average-linkage hierarchical clustering on the 1-|corr|
    distance - not a greedy pairwise pass, which would be order-dependent
    and wouldn't merge chains of >2 mutually-similar features consistently."""
    corr = train_df[feature_cols].corr().abs()
    distance = (1 - corr).clip(lower=0)
    condensed = squareform(distance.to_numpy(), checks=False)
    links = linkage(condensed, method="average")
    cluster_ids = fcluster(links, t=1 - threshold, criterion="distance")

    clusters: dict[int, list[str]] = {}
    for col, cluster_id in zip(feature_cols, cluster_ids):
        clusters.setdefault(cluster_id, []).append(col)
    return list(clusters.values())


def pick_cluster_representatives(
    train_df: pd.DataFrame,
    clusters: list[list[str]],
    target_col: str,
) -> list[str]:
    """Within each redundant cluster, keep the member most correlated with
    the target - not the one with the highest variance, since variance
    doesn't say anything about which twin is actually predictive."""
    representatives = []
    for cluster in clusters:
        if len(cluster) == 1:
            representatives.append(cluster[0])
            continue
        target_corr = {
            col: abs(train_df[col].corr(train_df[target_col])) for col in cluster
        }
        representatives.append(max(target_corr, key=target_corr.get))
    return representatives


def filter_by_relevance(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    threshold: float = RELEVANCE_THRESHOLD,
) -> list[str]:
    return [
        col
        for col in feature_cols
        if abs(train_df[col].corr(train_df[target_col])) >= threshold
    ]


def select_features_single_fold(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    redundancy_threshold: float,
    relevance_threshold: float,
) -> list[str]:
    clusters = cluster_redundant_features(train_df, feature_cols, redundancy_threshold)
    deduped = pick_cluster_representatives(train_df, clusters, target_col)
    return filter_by_relevance(train_df, deduped, target_col, relevance_threshold)


def select_top_n_by_correlation_single_fold(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    n: int,
) -> list[str]:
    """Alternative to select_features_single_fold: skips the redundancy
    clustering entirely and just keeps whichever `n` features have the
    highest |correlation| with the target - for comparing against the
    cluster-then-filter approach."""
    target_corr = {col: abs(train_df[col].corr(train_df[target_col])) for col in feature_cols}
    ranked = sorted(target_corr, key=target_corr.get, reverse=True)
    return ranked[:n]


def run_selector_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_folds: int,
    min_fold_survival: float,
    single_fold_selector,
) -> pd.DataFrame:
    """Applies `single_fold_selector(train_df) -> list[str]` inside every
    training fold of a GroupKFold grouped by Year, so selection never sees a
    row that ends up in that fold's held-out year. A feature is only kept
    overall if it was selected in at least `min_fold_survival` of the folds."""
    gkf = GroupKFold(n_splits=n_folds)
    survival_counts = {col: 0 for col in feature_cols}

    for train_idx, _ in gkf.split(df, groups=df["Year"]):
        train_df = df.iloc[train_idx]
        for col in single_fold_selector(train_df):
            survival_counts[col] += 1

    result = pd.DataFrame(
        {
            "feature": feature_cols,
            "n_folds_survived": [survival_counts[col] for col in feature_cols],
        }
    )
    result["fold_survival_rate"] = result["n_folds_survived"] / n_folds
    result["kept"] = result["fold_survival_rate"] >= min_fold_survival
    return result.sort_values("fold_survival_rate", ascending=False).reset_index(drop=True)


def select_features_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    n_folds: int,
    redundancy_threshold: float,
    relevance_threshold: float,
    min_fold_survival: float,
) -> pd.DataFrame:
    """Runs stage 1+2 (redundancy clustering, then relevance filter)."""
    return run_selector_cv(
        df,
        feature_cols,
        n_folds,
        min_fold_survival,
        lambda train_df: select_features_single_fold(
            train_df, feature_cols, target_col, redundancy_threshold, relevance_threshold
        ),
    )


def select_top_n_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    n_folds: int,
    n: int,
    min_fold_survival: float,
) -> pd.DataFrame:
    """No clustering - just the `n` features most correlated with the target
    in each fold."""
    return run_selector_cv(
        df,
        feature_cols,
        n_folds,
        min_fold_survival,
        lambda train_df: select_top_n_by_correlation_single_fold(
            train_df, feature_cols, target_col, n
        ),
    )


def select_features_with_tightening(
    df: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLS,
    target_col: str = MODEL_TARGET_COL,
    n_folds: int = N_FOLDS,
    redundancy_threshold: float = REDUNDANCY_THRESHOLD,
    relevance_threshold: float = RELEVANCE_THRESHOLD,
    min_fold_survival: float = MIN_FOLD_SURVIVAL,
    target_max_features: int = TARGET_MAX_FEATURES,
    max_iterations: int = MAX_TIGHTEN_ITERS,
) -> tuple[pd.DataFrame, list[dict[str, float]]]:
    history: list[dict[str, float]] = []
    result = pd.DataFrame()

    for iteration in range(1, max_iterations + 1):
        result = select_features_cv(
            df,
            feature_cols,
            target_col,
            n_folds,
            redundancy_threshold,
            relevance_threshold,
            min_fold_survival,
        )
        n_kept = int(result["kept"].sum())
        history.append(
            {
                "iteration": iteration,
                "redundancy_threshold": redundancy_threshold,
                "relevance_threshold": relevance_threshold,
                "n_kept": n_kept,
            }
        )
        if n_kept <= target_max_features:
            break
        redundancy_threshold = max(0.5, redundancy_threshold - TIGHTEN_STEP)
        relevance_threshold += TIGHTEN_STEP / 5

    return result, history


def main() -> None:
    df = load_and_clean_data()
    result, history = select_features_with_tightening(df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_DIR / "lyrics_feature_selection.csv", index=False)
    pd.DataFrame(history).to_csv(
        OUTPUT_DIR / "lyrics_feature_selection_tightening_history.csv", index=False
    )

    kept = result.loc[result["kept"], "feature"].tolist()
    print(f"{len(df)} rows, {len(FEATURE_COLS)} candidate features")
    print(f"\nTightening history ({len(history)} iteration(s)):")
    print(pd.DataFrame(history).to_string(index=False))
    print(f"\n{len(kept)} features kept (survived >= {int(MIN_FOLD_SURVIVAL * 100)}% of {N_FOLDS} folds):")
    for col in kept:
        print(f"  - {col}")
    print(f"\nFull per-feature breakdown saved to {OUTPUT_DIR / 'lyrics_feature_selection.csv'}")


if __name__ == "__main__":
    main()
