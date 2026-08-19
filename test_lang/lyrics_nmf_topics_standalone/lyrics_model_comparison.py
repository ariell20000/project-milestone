"""Does the stage-1/2 feature selection in lyrics_feature_selection.py
actually help? Compares four lyrics-feature sets - the original 10 themes,
the 16 lyrics_feature_selection.py kept (cluster-then-filter), the same
number picked by plain top-N-by-correlation (no redundancy clustering), and
all 100 candidates - using the exact same score_model Ridge pipelines and the
exact same rows/folds (GroupKFold by Year on eurovision_enriched1.csv), so
the feature set is the only thing that differs between runs.

A single train/test split can be lucky or unlucky (we saw this directly with
the pairwise model), so this evaluates every feature set across all 5 folds
and reports mean +/- std R2, not one number.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold

from lyrics_feature_selection import FEATURE_COLS as ALL_100_FEATURES
from lyrics_feature_selection import MIN_FOLD_SURVIVAL
from lyrics_feature_selection import MODEL_TARGET_COL
from lyrics_feature_selection import N_FOLDS as SELECTION_N_FOLDS
from lyrics_feature_selection import load_and_clean_data
from lyrics_feature_selection import select_top_n_cv
from score_model import COUNTRY_COL
from score_model import THEME_COLS as ORIGINAL_10_FEATURES
from score_model import evaluate_models


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
SELECTION_PATH = OUTPUT_DIR / "lyrics_feature_selection.csv"
TOP_N_SELECTION_PATH = OUTPUT_DIR / "lyrics_feature_selection_top_n.csv"

N_FOLDS = 5


def load_selected_16_features(selection_path: Path = SELECTION_PATH) -> list[str]:
    result = pd.read_csv(selection_path)
    return result.loc[result["kept"], "feature"].tolist()


def select_top_n_features(
    df: pd.DataFrame, n: int, save_path: Path = TOP_N_SELECTION_PATH
) -> list[str]:
    """Same GroupKFold-by-Year + cross-fold-survival discipline as
    lyrics_feature_selection.py, but skipping the redundancy-clustering step
    entirely - just the `n` features most correlated with the target in each
    fold, kept only if that held in >= MIN_FOLD_SURVIVAL of the folds."""
    result = select_top_n_cv(
        df,
        feature_cols=ALL_100_FEATURES,
        target_col=MODEL_TARGET_COL,
        n_folds=SELECTION_N_FOLDS,
        n=n,
        min_fold_survival=MIN_FOLD_SURVIVAL,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(save_path, index=False)
    return result.loc[result["kept"], "feature"].tolist()


def evaluate_feature_set(
    df: pd.DataFrame, feature_cols: list[str], n_folds: int = N_FOLDS
) -> pd.DataFrame:
    gkf = GroupKFold(n_splits=n_folds)
    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(gkf.split(df, groups=df["Year"])):
        X_train = df.iloc[train_idx][feature_cols + [COUNTRY_COL]].copy()
        X_test = df.iloc[test_idx][feature_cols + [COUNTRY_COL]].copy()
        y_train = df.iloc[train_idx][MODEL_TARGET_COL].copy()
        y_test = df.iloc[test_idx][MODEL_TARGET_COL].copy()

        metrics = evaluate_models(X_train, X_test, y_train, y_test, feature_cols=feature_cols)
        metrics.insert(0, "fold", fold)
        fold_results.append(metrics)
    return pd.concat(fold_results, ignore_index=True)


def summarize(all_folds: pd.DataFrame) -> pd.DataFrame:
    summary = (
        all_folds.groupby("model")["r2"]
        .agg(mean_r2="mean", std_r2="std")
        .reset_index()
    )
    return summary


def main() -> None:
    df = load_and_clean_data()
    selected_16 = load_selected_16_features()
    top_n_by_corr = select_top_n_features(df, n=len(selected_16))

    feature_sets = {
        "original_10": ORIGINAL_10_FEATURES,
        "selected_16 (cluster-then-filter)": selected_16,
        f"top_{len(selected_16)}_by_corr (no clustering)": top_n_by_corr,
        "all_100": ALL_100_FEATURES,
    }

    all_summaries = []
    all_folds_by_set = []
    for set_name, feature_cols in feature_sets.items():
        all_folds = evaluate_feature_set(df, feature_cols)
        all_folds.insert(0, "feature_set", set_name)
        all_folds_by_set.append(all_folds)

        summary = summarize(all_folds)
        summary.insert(0, "feature_set", set_name)
        summary.insert(2, "n_features", len(feature_cols))
        all_summaries.append(summary)

    detail = pd.concat(all_folds_by_set, ignore_index=True)
    summary = pd.concat(all_summaries, ignore_index=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detail.to_csv(OUTPUT_DIR / "lyrics_feature_set_comparison_by_fold.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "lyrics_feature_set_comparison_summary.csv", index=False)

    print(f"{len(df)} rows, {N_FOLDS}-fold GroupKFold by Year\n")
    print("Mean +/- std R2 across folds, by feature set and model:")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
