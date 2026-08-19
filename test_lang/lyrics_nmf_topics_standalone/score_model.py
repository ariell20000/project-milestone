from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "eurovision_enriched1.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_PATH = ARTIFACT_DIR / "score_model.joblib"

COUNTRY_COL = "Country"
TARGET_COL = "Score"
MODEL_TARGET_COL = "Score_norm"

THEME_COLS = [
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
]


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text in {"", "-"} else text


def load_and_clean_data(csv_path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.columns[0] == "":
        df = df.drop(columns=df.columns[0])

    df = df.copy()
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df[COUNTRY_COL] = df[COUNTRY_COL].map(clean_text)
    for col in THEME_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)

    df = df.dropna(subset=[TARGET_COL, "Year"])
    df = df[df[COUNTRY_COL] != ""]

    # Total points awarded per year scales with how many countries were
    # voting that year (~150 max in the 1970s vs ~580 max in the 2020s), so
    # raw Score isn't comparable across years - the model target is each
    # song's share of its own year's total instead.
    year_totals = df.groupby("Year")[TARGET_COL].transform("sum")
    df[MODEL_TARGET_COL] = df[TARGET_COL] / year_totals

    df = df.reset_index(drop=True)
    return df


def make_baseline_model() -> DummyRegressor:
    return DummyRegressor(strategy="mean")


def make_theme_model(feature_cols: list[str] = THEME_COLS) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocessor",
                ColumnTransformer(
                    transformers=[
                        ("themes", "passthrough", feature_cols),
                    ]
                ),
            ),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )


def make_country_model() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocessor",
                ColumnTransformer(
                    transformers=[
                        ("country", OneHotEncoder(handle_unknown="ignore"), [COUNTRY_COL]),
                    ]
                ),
            ),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )


def make_combined_model(feature_cols: list[str] = THEME_COLS) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocessor",
                ColumnTransformer(
                    transformers=[
                        ("themes", "passthrough", feature_cols),
                        ("country", OneHotEncoder(handle_unknown="ignore"), [COUNTRY_COL]),
                    ]
                ),
            ),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )


def split_data(
    df: pd.DataFrame,
    feature_cols: list[str] = THEME_COLS,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # Grouped by Year (not a plain row-level split): a whole contest year goes
    # entirely to train or entirely to test, so a song can't leak into both
    # sides. Which years land in test is still randomized via random_state,
    # not a fixed/chronological holdout.
    features = df[feature_cols + [COUNTRY_COL]].copy()
    target = df[MODEL_TARGET_COL].copy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(features, groups=df["Year"]))
    return (
        features.iloc[train_idx],
        features.iloc[test_idx],
        target.iloc[train_idx],
        target.iloc[test_idx],
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
    feature_cols: list[str] = THEME_COLS,
) -> pd.DataFrame:
    models: list[tuple[str, object, list[str]]] = [
        ("baseline_mean", make_baseline_model(), []),
        ("themes_only", make_theme_model(feature_cols), feature_cols),
        ("country_only", make_country_model(), [COUNTRY_COL]),
        ("combined", make_combined_model(feature_cols), feature_cols + [COUNTRY_COL]),
    ]

    rows: list[dict[str, float | str]] = []
    for name, model, cols in models:
        Xtr = X_train if not cols else X_train[cols].copy()
        Xte = X_test if not cols else X_test[cols].copy()
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


def train_final_model(df: pd.DataFrame, feature_cols: list[str] = THEME_COLS) -> Pipeline:
    X = df[feature_cols + [COUNTRY_COL]].copy()
    y = df[MODEL_TARGET_COL].copy()
    model = make_combined_model(feature_cols)
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

    theme_df = (
        df[df["feature"].str.startswith("themes__")]
        .assign(abs=lambda frame: frame["coefficient"].abs())
        .sort_values("abs", ascending=False)
        .drop(columns="abs")
        .head(20)
    )
    country_df = (
        df[df["feature"].str.startswith("country__")]
        .assign(abs=lambda frame: frame["coefficient"].abs())
        .sort_values("abs", ascending=False)
        .drop(columns="abs")
        .head(20)
    )
    theme_df.to_csv(output_dir / "top_theme_coefficients.csv", index=False)
    country_df.to_csv(output_dir / "top_country_coefficients.csv", index=False)


def predict_score(model: Pipeline, lyrics: str, country: str) -> float:
    sample = pd.DataFrame([build_sample_from_lyrics(lyrics, country)])
    return max(0.0, float(model.predict(sample)[0]))


def build_sample_from_lyrics(lyrics: str, country: str) -> dict[str, str]:
    # Build sample using theme columns inferred in the CSV (continuous values)
    # Here we will simple set themes to 0 and not attempt to infer from lyrics text
    sample = {col: 0.0 for col in THEME_COLS}
    sample[COUNTRY_COL] = clean_text(country)
    return sample


def main() -> None:
    df = load_and_clean_data()
    X_train, X_test, y_train, y_test = split_data(df)

    metrics = evaluate_models(X_train, X_test, y_train, y_test)
    combined = make_combined_model()
    combined.fit(X_train, y_train)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_artifact(train_final_model(df))
    metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    extract_coefficients(combined).to_csv(OUTPUT_DIR / "top_coefficients.csv", index=False)
    save_grouped_coefficients(combined)

    theme_importance = group_permutation_importance(combined, X_test, y_test, THEME_COLS)
    country_importance = group_permutation_importance(combined, X_test, y_test, [COUNTRY_COL])
    importance_df = pd.DataFrame(
        [
            {"group": "themes", **theme_importance},
            {"group": "country", **country_importance},
        ]
    )
    importance_df.to_csv(OUTPUT_DIR / "group_importance.csv", index=False)

    print("Model metrics:")
    print(metrics.to_string(index=False))
    print("\nGroup importance:")
    print(importance_df.to_string(index=False))
    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
