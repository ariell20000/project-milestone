from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from main import clean_data as clean_votes
from main import load_data as load_votes_raw
from main import normalize_country
from score_model import COUNTRY_COL
from score_model import TARGET_COL as SCORE_COL
from score_model import THEME_COLS
from score_model import load_and_clean_data as load_enriched_data
from score_model import make_combined_model as make_old_combined_model
from score_model import split_data as split_songs


BASE_DIR = Path(__file__).resolve().parent
VOTES_PATH = BASE_DIR / "eurovision_1957-2021.csv"
ENRICHED_PATH = BASE_DIR / "eurovision_enriched.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_PATH = ARTIFACT_DIR / "pairwise_vote_model.joblib"

FROM_COL = "From"
TO_COL = "To"
PAIR_COL = "Pair"
TARGET_COL = "Points"

# A pair only gets its own coefficient if it shows up often enough AND scores
# high enough on average - otherwise a single lucky 12-pointer (n=1) would look
# like a "relationship". These thresholds land on ~30 pairs, which line up with
# known Eurovision voting blocks (Cyprus<->Greece, Moldova<->Romania, the
# post-Soviet cluster, the Baltics, ...).
MIN_PAIR_OBSERVATIONS = 5
MIN_PAIR_AVG_POINTS = 10.0
OTHER_PAIR_LABEL = "Other"

# main.normalize_country() doesn't catch these votes-file spellings
# ("the netherlands", "czech-republic", "north macedonia" ...), so without this
# the merge against eurovision_enriched.csv would silently drop those countries.
EXTRA_COUNTRY_ALIASES = {
    "The Netherlands": "Netherlands",
    "Czech Republic": "Czechia",
    "Czech-Republic": "Czechia",
    "North Macedonia": "Macedonia",
}


def harmonize_country(name: str) -> str:
    return EXTRA_COUNTRY_ALIASES.get(name, name)


def _load_clean_votes_and_songs(
    votes_path: Path = VOTES_PATH, enriched_path: Path = ENRICHED_PATH
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    votes = clean_votes(load_votes_raw(votes_path))
    votes[FROM_COL] = votes[FROM_COL].map(harmonize_country)
    votes[TO_COL] = votes[TO_COL].map(harmonize_country)
    votes_agg = (
        votes.groupby(["Year", FROM_COL, TO_COL], as_index=False)[TARGET_COL].sum()
    )

    songs = load_enriched_data(enriched_path)
    songs = songs.copy()
    songs["Country"] = songs["Country"].map(normalize_country).map(harmonize_country)
    songs = songs.drop_duplicates(subset=["Year", "Country"])

    return votes, votes_agg, songs


def build_pairwise_dataset(
    votes_path: Path = VOTES_PATH, enriched_path: Path = ENRICHED_PATH
) -> pd.DataFrame:
    """Only the (Year, From, To) pairs that actually appear in the votes file,
    i.e. From actually gave To some points. Useful for asking "given that a
    voter awarded points, how many" - NOT for estimating a country's total
    score, since it silently assumes every pair received a vote (see
    build_full_voting_grid for that)."""
    _votes, votes_agg, songs = _load_clean_votes_and_songs(votes_path, enriched_path)

    merged = votes_agg.merge(
        songs[["Year", "Country", *THEME_COLS]],
        left_on=["Year", TO_COL],
        right_on=["Year", "Country"],
        how="inner",
    ).drop(columns="Country")

    # Explicit From-To pair identity, so a linear model can learn a
    # country-pair-specific coefficient instead of only additive From/To effects.
    merged[PAIR_COL] = merged[FROM_COL] + " -> " + merged[TO_COL]

    return merged.reset_index(drop=True)


def build_full_voting_grid(
    votes_path: Path = VOTES_PATH, enriched_path: Path = ENRICHED_PATH
) -> pd.DataFrame:
    """Every (Year, From, To) combination where From was an eligible voter that
    year and To competed that year - not just the pairs where From actually
    gave To points. Eurovision voters only report their top ~10 picks, so any
    combination missing from the votes file received 0 points, not "unknown".
    This is what a fair total-score estimate has to sum over: the known roster
    of possible voters, not the (outcome-dependent) list of who actually voted
    for a given song."""
    votes, votes_agg, songs = _load_clean_votes_and_songs(votes_path, enriched_path)

    grids = []
    for year, year_votes in votes.groupby("Year"):
        voters = year_votes[FROM_COL].unique()
        competitors = songs.loc[songs["Year"] == year, "Country"].unique()
        if len(voters) == 0 or len(competitors) == 0:
            continue
        grid = pd.MultiIndex.from_product(
            [voters, competitors], names=[FROM_COL, TO_COL]
        ).to_frame(index=False)
        grid = grid[grid[FROM_COL] != grid[TO_COL]].copy()
        grid["Year"] = year
        grids.append(grid)
    full_grid = pd.concat(grids, ignore_index=True)

    full_grid = full_grid.merge(votes_agg, on=["Year", FROM_COL, TO_COL], how="left")
    full_grid[TARGET_COL] = full_grid[TARGET_COL].fillna(0.0)

    full_grid = full_grid.merge(
        songs[["Year", "Country", *THEME_COLS]],
        left_on=["Year", TO_COL],
        right_on=["Year", "Country"],
        how="inner",
    ).drop(columns="Country")

    full_grid[PAIR_COL] = full_grid[FROM_COL] + " -> " + full_grid[TO_COL]

    return full_grid.reset_index(drop=True)


def compute_stable_pairs(
    df: pd.DataFrame,
    min_observations: int = MIN_PAIR_OBSERVATIONS,
    min_avg_points: float = MIN_PAIR_AVG_POINTS,
) -> set[str]:
    stats = df.groupby(PAIR_COL)[TARGET_COL].agg(["mean", "count"])
    stable = stats[(stats["count"] >= min_observations) & (stats["mean"] >= min_avg_points)]
    return set(stable.index)


def bucket_rare_pairs(df: pd.DataFrame, stable_pairs: set[str]) -> pd.DataFrame:
    df = df.copy()
    df[PAIR_COL] = df[PAIR_COL].where(df[PAIR_COL].isin(stable_pairs), OTHER_PAIR_LABEL)
    return df


def make_baseline_model() -> DummyRegressor:
    return DummyRegressor(strategy="mean")


def make_theme_model() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocessor",
                ColumnTransformer(transformers=[("themes", "passthrough", THEME_COLS)]),
            ),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )


def make_from_country_model() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocessor",
                ColumnTransformer(
                    transformers=[
                        ("from_country", OneHotEncoder(handle_unknown="ignore"), [FROM_COL]),
                    ]
                ),
            ),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )


def make_to_country_model() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocessor",
                ColumnTransformer(
                    transformers=[
                        ("to_country", OneHotEncoder(handle_unknown="ignore"), [TO_COL]),
                    ]
                ),
            ),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )


def make_countries_model() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocessor",
                ColumnTransformer(
                    transformers=[
                        ("from_country", OneHotEncoder(handle_unknown="ignore"), [FROM_COL]),
                        ("to_country", OneHotEncoder(handle_unknown="ignore"), [TO_COL]),
                    ]
                ),
            ),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )


def make_combined_model() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocessor",
                ColumnTransformer(
                    transformers=[
                        ("themes", "passthrough", THEME_COLS),
                        ("from_country", OneHotEncoder(handle_unknown="ignore"), [FROM_COL]),
                        ("to_country", OneHotEncoder(handle_unknown="ignore"), [TO_COL]),
                    ]
                ),
            ),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )


def make_pair_model() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocessor",
                ColumnTransformer(
                    transformers=[
                        ("pair", OneHotEncoder(handle_unknown="ignore"), [PAIR_COL]),
                    ]
                ),
            ),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )


def make_combined_with_pair_model() -> Pipeline:
    # Same as make_combined_model, plus an explicit country-pair dummy so the
    # (still fully linear/interpretable) model can pick up pair-specific effects
    # like geopolitical voting blocks, on top of the additive From/To effects.
    return Pipeline(
        steps=[
            (
                "preprocessor",
                ColumnTransformer(
                    transformers=[
                        ("themes", "passthrough", THEME_COLS),
                        ("from_country", OneHotEncoder(handle_unknown="ignore"), [FROM_COL]),
                        ("to_country", OneHotEncoder(handle_unknown="ignore"), [TO_COL]),
                        ("pair", OneHotEncoder(handle_unknown="ignore"), [PAIR_COL]),
                    ]
                ),
            ),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    features = df[THEME_COLS + [FROM_COL, TO_COL, PAIR_COL]].copy()
    target = df[TARGET_COL].copy()
    return train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )


def evaluate_predictions(y_true: pd.Series, y_pred: Iterable[float]) -> dict[str, float]:
    preds = list(y_pred)
    return {
        "mse": float(mean_squared_error(y_true, preds)),
        "r2": float(r2_score(y_true, preds)),
    }


def evaluate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame:
    stable_pairs = compute_stable_pairs(X_train.assign(**{TARGET_COL: y_train}))
    X_train_bucketed = bucket_rare_pairs(X_train, stable_pairs)
    X_test_bucketed = bucket_rare_pairs(X_test, stable_pairs)

    models: list[tuple[str, object, list[str], pd.DataFrame, pd.DataFrame]] = [
        ("baseline_mean", make_baseline_model(), [], X_train, X_test),
        ("themes_only", make_theme_model(), THEME_COLS, X_train, X_test),
        ("from_country_only", make_from_country_model(), [FROM_COL], X_train, X_test),
        ("to_country_only", make_to_country_model(), [TO_COL], X_train, X_test),
        ("countries_only", make_countries_model(), [FROM_COL, TO_COL], X_train, X_test),
        ("combined", make_combined_model(), THEME_COLS + [FROM_COL, TO_COL], X_train, X_test),
        ("pair_only", make_pair_model(), [PAIR_COL], X_train_bucketed, X_test_bucketed),
        (
            "combined_with_pair",
            make_combined_with_pair_model(),
            THEME_COLS + [FROM_COL, TO_COL, PAIR_COL],
            X_train_bucketed,
            X_test_bucketed,
        ),
    ]

    rows: list[dict[str, float | str]] = []
    for name, model, cols, Xtr_source, Xte_source in models:
        Xtr = Xtr_source if not cols else Xtr_source[cols].copy()
        Xte = Xte_source if not cols else Xte_source[cols].copy()
        model.fit(Xtr, y_train)
        metrics = evaluate_predictions(y_test, model.predict(Xte))
        rows.append({"model": name, **metrics})
    return pd.DataFrame(rows)


def group_permutation_importance(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    columns: list[str],
    n_repeats: int = 5,
) -> dict[str, float]:
    baseline = evaluate_predictions(y_test, model.predict(X_test))
    deltas: list[float] = []
    for repeat in range(n_repeats):
        shuffled = X_test.copy()
        for col in columns:
            shuffled[col] = X_test[col].sample(frac=1.0, random_state=42 + repeat).to_numpy()
        permuted = evaluate_predictions(y_test, model.predict(shuffled))
        deltas.append(permuted["mse"] - baseline["mse"])
    return {
        "baseline_mse": baseline["mse"],
        "mse_increase": float(sum(deltas) / len(deltas)),
    }


def train_final_model(df: pd.DataFrame) -> Pipeline:
    stable_pairs = compute_stable_pairs(df)
    df = bucket_rare_pairs(df, stable_pairs)
    X = df[THEME_COLS + [FROM_COL, TO_COL, PAIR_COL]].copy()
    y = df[TARGET_COL].copy()
    model = make_combined_with_pair_model()
    model.fit(X, y)
    return model


def save_artifact(model: Pipeline, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_artifact(path: Path = MODEL_PATH) -> Pipeline:
    return joblib.load(path)


def extract_coefficients(model: Pipeline, top_n: int = 20) -> pd.DataFrame:
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    coefs = model.named_steps["regressor"].coef_.ravel()
    df = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
    df["abs"] = df["coefficient"].abs()
    return df.sort_values("abs", ascending=False).head(top_n).drop(columns="abs")


def save_grouped_coefficients(model: Pipeline, output_dir: Path = OUTPUT_DIR) -> None:
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    coefs = model.named_steps["regressor"].coef_.ravel()
    df = pd.DataFrame({"feature": feature_names, "coefficient": coefs})

    groups = {
        "themes": "pairwise_top_theme_coefficients.csv",
        "from_country": "pairwise_top_from_country_coefficients.csv",
        "to_country": "pairwise_top_to_country_coefficients.csv",
        "pair": "pairwise_top_pair_coefficients.csv",
    }
    for prefix, filename in groups.items():
        group_df = (
            df[df["feature"].str.startswith(f"{prefix}__")]
            .assign(abs=lambda frame: frame["coefficient"].abs())
            .sort_values("abs", ascending=False)
            .drop(columns="abs")
            .head(20)
        )
        group_df.to_csv(output_dir / filename, index=False)


def predict_pairwise_score(
    model: Pipeline, from_country: str, to_country: str, lyrics: str
) -> float:
    sample = pd.DataFrame([build_sample_from_lyrics(lyrics, from_country, to_country)])
    return max(0.0, float(model.predict(sample)[0]))


def build_sample_from_lyrics(lyrics: str, from_country: str, to_country: str) -> dict[str, str]:
    # Same simplification as score_model.build_sample_from_lyrics: themes are not
    # inferred from raw lyrics text here, so they default to 0.
    sample = {col: 0.0 for col in THEME_COLS}
    sample[FROM_COL] = normalize_country(from_country)
    sample[TO_COL] = normalize_country(to_country)
    # A single ad-hoc prediction has no reliable way to know if this pair was
    # one of the stable pairs baked into the trained model, so fall back to
    # the generic "Other" bucket rather than a pair string the model never saw.
    sample[PAIR_COL] = OTHER_PAIR_LABEL
    return sample


def evaluate_against_old_model() -> pd.DataFrame:
    """Aggregate this model's per-pair predictions into a predicted total score
    per (Year, Country) and compare it to score_model's direct Score prediction,
    on the exact same held-out songs."""
    songs_df = load_enriched_data()
    songs_df["Country_norm"] = songs_df[COUNTRY_COL].map(normalize_country).map(harmonize_country)
    X_train, X_test, y_train, y_test = split_songs(songs_df)

    train_keys = set(
        zip(songs_df.loc[X_train.index, "Year"], songs_df.loc[X_train.index, "Country_norm"])
    )
    test_keys = set(
        zip(songs_df.loc[X_test.index, "Year"], songs_df.loc[X_test.index, "Country_norm"])
    )

    # Full voter x competitor grid (including implicit zeros) - a fair total-score
    # estimate has to sum over every country that *could* have voted, not just the
    # ones who, as it turned out, actually gave this song points.
    pairwise_df = build_full_voting_grid()
    pairwise_df["_key"] = list(zip(pairwise_df["Year"], pairwise_df[TO_COL]))
    train_edges = pairwise_df[pairwise_df["_key"].isin(train_keys)].drop(columns="_key")
    test_edges = pairwise_df[pairwise_df["_key"].isin(test_keys)].drop(columns="_key")

    stable_pairs = compute_stable_pairs(train_edges)
    train_edges = bucket_rare_pairs(train_edges, stable_pairs)
    test_edges = bucket_rare_pairs(test_edges, stable_pairs)

    pairwise_model = make_combined_with_pair_model()
    pairwise_model.fit(
        train_edges[THEME_COLS + [FROM_COL, TO_COL, PAIR_COL]], train_edges[TARGET_COL]
    )
    # Points can't be negative; Ridge trained on a lot of true zeros will predict some.
    test_edges["predicted_points"] = pairwise_model.predict(
        test_edges[THEME_COLS + [FROM_COL, TO_COL, PAIR_COL]]
    ).clip(min=0)
    predicted_total = (
        test_edges.groupby(["Year", TO_COL])["predicted_points"]
        .sum()
        .reset_index()
        .rename(columns={TO_COL: "Country", "predicted_points": "predicted_score"})
    )

    old_model = make_old_combined_model()
    old_model.fit(X_train, y_train)

    comparison = songs_df.loc[X_test.index, ["Year", "Country_norm", SCORE_COL]].rename(
        columns={"Country_norm": "Country", SCORE_COL: "actual_score"}
    )
    comparison["old_model_score"] = old_model.predict(X_test)
    comparison = comparison.merge(predicted_total, on=["Year", "Country"], how="left")
    comparison.to_csv(OUTPUT_DIR / "predicted_vs_actual_total_score.csv", index=False)

    covered = comparison.dropna(subset=["predicted_score"])
    rows = [
        {
            "model": "old_score_model (all test songs)",
            "n": len(comparison),
            **evaluate_predictions(comparison["actual_score"], comparison["old_model_score"]),
        },
        {
            "model": "old_score_model (songs with vote coverage)",
            "n": len(covered),
            **evaluate_predictions(covered["actual_score"], covered["old_model_score"]),
        },
        {
            "model": "pairwise_aggregated (songs with vote coverage)",
            "n": len(covered),
            **evaluate_predictions(covered["actual_score"], covered["predicted_score"]),
        },
    ]
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_DIR / "pairwise_vs_score_model_comparison.csv", index=False)
    return result


def main() -> None:
    df = build_pairwise_dataset()
    X_train, X_test, y_train, y_test = split_data(df)

    metrics = evaluate_models(X_train, X_test, y_train, y_test)

    stable_pairs = compute_stable_pairs(X_train.assign(**{TARGET_COL: y_train}))
    X_train_bucketed = bucket_rare_pairs(X_train, stable_pairs)
    X_test_bucketed = bucket_rare_pairs(X_test, stable_pairs)

    combined = make_combined_with_pair_model()
    combined.fit(X_train_bucketed, y_train)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_artifact(train_final_model(df))
    metrics.to_csv(OUTPUT_DIR / "pairwise_model_metrics.csv", index=False)
    extract_coefficients(combined).to_csv(
        OUTPUT_DIR / "pairwise_top_coefficients.csv", index=False
    )
    save_grouped_coefficients(combined)

    theme_importance = group_permutation_importance(combined, X_test_bucketed, y_test, THEME_COLS)
    from_importance = group_permutation_importance(combined, X_test_bucketed, y_test, [FROM_COL])
    to_importance = group_permutation_importance(combined, X_test_bucketed, y_test, [TO_COL])
    pair_importance = group_permutation_importance(combined, X_test_bucketed, y_test, [PAIR_COL])
    importance_df = pd.DataFrame(
        [
            {"group": "themes", **theme_importance},
            {"group": "from_country", **from_importance},
            {"group": "to_country", **to_importance},
            {"group": "pair", **pair_importance},
        ]
    )
    importance_df.to_csv(OUTPUT_DIR / "pairwise_group_importance.csv", index=False)

    comparison = evaluate_against_old_model()

    print(f"Pairwise dataset size: {len(df)} rows")
    print("\nModel metrics:")
    print(metrics.to_string(index=False))
    print("\nGroup importance:")
    print(importance_df.to_string(index=False))
    print("\nAggregated total score vs. old score_model (same held-out songs):")
    print(comparison.to_string(index=False))
    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
