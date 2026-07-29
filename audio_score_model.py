from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
#not relevant file, not found enough data

BASE_DIR = Path(__file__).resolve().parent
LYRICS_PATH = BASE_DIR / "eurovision-lyrics-2025.json"
# Two general-purpose Spotify audio-features datasets, combined to maximize
# how many Eurovision entries we can find audio features for. They only
# overlap on ~400 tracks out of ~1.15M combined, so using both roughly
# doubles the number of Eurovision matches versus either alone. Neither file
# is Eurovision-specific and neither carries a contest Score.
TRACKS_FEATURES_PATH = BASE_DIR / "tracks_features.csv"
UNIVERSAL_TOP_SONGS_PATH = BASE_DIR / "universal_top_spotify_songs.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_PATH = ARTIFACT_DIR / "audio_score_model.joblib"

TARGET_COL = "Score"
AUDIO_COLS = ["danceability", "energy", "loudness", "tempo"]

# Minimum number of merged rows needed before we bother holding out a test split.
MIN_ROWS_FOR_SPLIT = 20


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()


def load_lyrics_scores(path: Path = LYRICS_PATH) -> pd.DataFrame:
    """Load the eurovision-lyrics-2025 dataset, which holds the actual contest
    Score for each entry (audio_features.csv has no such column)."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    df = pd.DataFrame(list(raw.values()))
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["artist_norm"] = df["Artist"].map(normalize_text)
    df["song_norm"] = df["Song"].map(normalize_text)
    df = df.dropna(subset=[TARGET_COL])
    df = df[(df["artist_norm"] != "") & (df["song_norm"] != "")]
    return df


def load_tracks_features(path: Path = TRACKS_FEATURES_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["id", "name", "artists"] + AUDIO_COLS)
    df = df.drop_duplicates(subset="id")
    for col in AUDIO_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["artist_norm"] = df["artists"].map(normalize_text)
    df["song_norm"] = df["name"].map(normalize_text)
    return df.dropna(subset=AUDIO_COLS)


def load_universal_top_songs(path: Path = UNIVERSAL_TOP_SONGS_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["spotify_id", "name", "artists"] + AUDIO_COLS)
    df = df.drop_duplicates(subset="spotify_id")
    for col in AUDIO_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["artist_norm"] = df["artists"].map(normalize_text)
    df["song_norm"] = df["name"].map(normalize_text)
    return df.dropna(subset=AUDIO_COLS)


def load_audio_features() -> pd.DataFrame:
    """Combine both general-purpose Spotify audio-features datasets. Most rows
    in either file are not Eurovision entries at all - they only become
    useful once merged against the lyrics/score dataset below. The two files
    barely overlap, so combining them roughly doubles the number of
    Eurovision entries we can find audio features for."""
    tracks = load_tracks_features()[["artist_norm", "song_norm"] + AUDIO_COLS]
    universal = load_universal_top_songs()[["artist_norm", "song_norm"] + AUDIO_COLS]
    combined = pd.concat([tracks, universal], ignore_index=True)
    return combined.drop_duplicates(subset=["artist_norm", "song_norm"])


def merge_datasets(
    lyrics_df: pd.DataFrame, audio_df: pd.DataFrame
) -> pd.DataFrame:
    """Inner-join on normalized (artist, song) so we keep only the Eurovision
    entries that we could actually find audio features for, paired with their
    real contest Score."""
    merged = lyrics_df.merge(
        audio_df, on=["artist_norm", "song_norm"], how="inner", suffixes=("", "_audio")
    )
    merged = merged.drop_duplicates(subset=["artist_norm", "song_norm"])
    merged = merged.dropna(subset=[TARGET_COL] + AUDIO_COLS)
    return merged.reset_index(drop=True)


def make_baseline_model() -> DummyRegressor:
    return DummyRegressor(strategy="mean")


def make_audio_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )


def evaluate_predictions(y_true: pd.Series, y_pred: Iterable[float]) -> dict[str, float]:
    preds = list(y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, preds)),
        "mse": float(mean_squared_error(y_true, preds)),
        "r2": float(r2_score(y_true, preds)),
    }


def evaluate_models(df: pd.DataFrame) -> pd.DataFrame:
    X = df[AUDIO_COLS].copy()
    y = df[TARGET_COL].copy()

    if len(df) < MIN_ROWS_FOR_SPLIT:
        print(
            f"Only {len(df)} merged rows found (fewer than {MIN_ROWS_FOR_SPLIT}); "
            "evaluating in-sample instead of holding out a test split."
        )
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    rows: list[dict[str, float | str]] = []
    for name, model in [
        ("baseline_mean", make_baseline_model()),
        ("audio_features", make_audio_model()),
    ]:
        model.fit(X_train, y_train)
        metrics = evaluate_predictions(y_test, model.predict(X_test))
        rows.append({"model": name, **metrics})
    return pd.DataFrame(rows)


def train_final_model(df: pd.DataFrame) -> Pipeline:
    X = df[AUDIO_COLS].copy()
    y = df[TARGET_COL].copy()
    model = make_audio_model()
    model.fit(X, y)
    return model


def save_artifact(model: Pipeline, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_artifact(path: Path = MODEL_PATH) -> Pipeline:
    return joblib.load(path)


def extract_coefficients(model: Pipeline) -> pd.DataFrame:
    coefs = model.named_steps["regressor"].coef_.ravel()
    return pd.DataFrame({"feature": AUDIO_COLS, "coefficient": coefs}).sort_values(
        "coefficient", key=lambda s: s.abs(), ascending=False
    )


def predict_score(
    model: Pipeline, danceability: float, energy: float, loudness: float, tempo: float
) -> float:
    sample = pd.DataFrame(
        [{"danceability": danceability, "energy": energy, "loudness": loudness, "tempo": tempo}]
    )
    return max(0.0, float(model.predict(sample)[0]))


def main() -> None:
    lyrics_df = load_lyrics_scores()
    audio_df = load_audio_features()
    merged = merge_datasets(lyrics_df, audio_df)

    print(
        f"Matched {len(merged)} Eurovision entries to audio features "
        f"(out of {len(lyrics_df)} scored entries and {len(audio_df)} audio tracks)."
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    merged.to_csv(OUTPUT_DIR / "audio_merged_dataset.csv", index=False)

    metrics = evaluate_models(merged)
    metrics.to_csv(OUTPUT_DIR / "audio_model_metrics.csv", index=False)

    final_model = train_final_model(merged)
    save_artifact(final_model)
    extract_coefficients(final_model).to_csv(
        OUTPUT_DIR / "audio_coefficients.csv", index=False
    )

    print("\nModel metrics:")
    print(metrics.to_string(index=False))
    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
