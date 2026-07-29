from __future__ import annotations

from functools import lru_cache

import pandas as pd

import rank_estimator
import score_model


@lru_cache(maxsize=1)
def load_model():
    return score_model.load_artifact()


def known_countries() -> list[str]:
    df = score_model.load_and_clean_data()
    return sorted(df[score_model.COUNTRY_COL].unique())


def _contribution_breakdown(model, sample: dict[str, object]) -> dict[str, float]:
    preprocessor = model.named_steps["preprocessor"]
    regressor = model.named_steps["regressor"]
    feature_names = preprocessor.get_feature_names_out()
    coefs = regressor.coef_.ravel()

    country_feature = f"country__{score_model.COUNTRY_COL}_{sample[score_model.COUNTRY_COL]}"

    theme_contribution = 0.0
    country_contribution = 0.0
    for name, coef in zip(feature_names, coefs):
        if name.startswith("themes__"):
            theme = name.removeprefix("themes__")
            theme_contribution += coef * sample.get(theme, 0.0)
        elif name == country_feature:
            country_contribution = float(coef)

    return {
        "theme_contribution": float(theme_contribution),
        "country_contribution": float(country_contribution),
        "intercept": float(regressor.intercept_),
    }


def predict(lyrics: str, country: str) -> dict[str, object]:
    model = load_model()
    sample = score_model.build_sample_from_lyrics(lyrics, country)

    predicted_score = max(0.0, float(model.predict(pd.DataFrame([sample]))[0]))
    place, percentile = rank_estimator.estimate_place(predicted_score)
    breakdown = _contribution_breakdown(model, sample)

    theme_scores = {col: sample[col] for col in score_model.THEME_COLS}

    return {
        "predicted_score": predicted_score,
        "theme_scores": theme_scores,
        "estimated_place": place,
        "percentile": percentile,
        **breakdown,
    }
