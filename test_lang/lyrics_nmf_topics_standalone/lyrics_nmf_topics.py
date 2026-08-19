"""Stage A: NMF topic-mapping (Levy, Granot & Peres 2024, Stages 1-3 -
see nmf_lyrics_topic_algorithm.md) applied to the Eurovision lyrics corpus.
Stage B: feeds the resulting per-song topic-weight vector into the exact
same GroupKFold-by-Year predictive pipeline used for the NLI-based theme
features (lyrics_model_comparison.py), to see whether an unsupervised,
corpus-driven topic space predicts Score_norm any better than the top-down,
hand-labeled NLI themes did.

gensim (the library the paper used for coherence-based K selection) doesn't
build on this machine's Python (3.14) - its Cython extensions rely on
removed CPython 3.13- internals. UMass coherence (Mimno et al. 2011) is
reimplemented directly here instead: it only needs document co-occurrence
counts of each topic's top words, which is a few lines over the count
matrix - no gensim required, and it's one of the two standard coherence
measures in the topic-modeling literature (the other, UCI/NPMI, needs an
external reference corpus that we don't have).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from scipy import sparse
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from lyrics_model_comparison import evaluate_feature_set, summarize
from score_model import COUNTRY_COL, TARGET_COL, clean_text


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "eurovision_enriched1.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

MIN_TRANSLATION_LEN = 30
K_RANGE = range(2, 31)  # same sweep range as the paper (they tested 1-30)
REPORT_K_VALUES = [5, 10, 15, 20, 25]  # evaluated for prediction regardless of the coherence argmax
PRIMARY_K = 15  # matches the paper's own choice; used for the saved topic-word/song-topic mapping output
TOP_WORDS_PER_TOPIC = 10
RANDOM_STATE = 42

STOPWORDS = set(stopwords.words("english"))
STEMMER = SnowballStemmer("english")
WORD_RE = re.compile(r"[a-zA-Z]+")


def build_english_text(df: pd.DataFrame) -> pd.Series:
    """`Lyrics translation` is a placeholder (just the word "English") for
    777/1795 rows - any song whose Language field involves English at all
    got no real translation. Those rows' `Lyrics` column already IS English
    text, so falling back to it recovers the full 1795-row corpus instead
    of silently dropping 43% of it."""
    translation = df["Lyrics translation"].fillna("")
    original = df["Lyrics"].fillna("")
    return translation.where(translation.str.len() >= MIN_TRANSLATION_LEN, original)


def strip_artist_name(text: str, artist: str) -> str:
    if not isinstance(artist, str) or not artist.strip():
        return text
    for name_part in re.split(r"[ &,/]+", artist):
        name_part = name_part.strip()
        if len(name_part) < 2:
            continue
        text = re.sub(re.escape(name_part), " ", text, flags=re.IGNORECASE)
    return text


def clean_and_stem(text: str, artist: str) -> str:
    """Paper's Stage 1 (see nmf_lyrics_topic_algorithm.md): stopwords,
    artist name, punctuation and numbers removed, then stemmed. Per user
    decision, "non-word syllable" filtering (haa/wawa - the paper doesn't
    define how it detected these) is skipped rather than guessed at."""
    text = strip_artist_name(text, artist)
    tokens = WORD_RE.findall(text.lower())
    stems = [STEMMER.stem(tok) for tok in tokens if tok not in STOPWORDS and len(tok) > 2]
    return " ".join(stems)


def umass_coherence(top_words: list[str], binary_doc_term: sparse.csr_matrix, vocab_index: dict[str, int]) -> float:
    """Mimno et al. (2011) UMass coherence for one topic's top words: high
    when the topic's words tend to co-occur in the same documents."""
    idxs = [vocab_index[w] for w in top_words if w in vocab_index]
    if len(idxs) < 2:
        return float("nan")
    doc_freq = np.asarray(binary_doc_term.sum(axis=0)).flatten()
    score, n_pairs = 0.0, 0
    for i in range(1, len(idxs)):
        for j in range(i):
            wi, wj = idxs[i], idxs[j]
            co_doc = binary_doc_term[:, wi].multiply(binary_doc_term[:, wj]).sum()
            score += np.log((co_doc + 1) / doc_freq[wj])
            n_pairs += 1
    return score / n_pairs


def top_words_for_topic(components_row: np.ndarray, vocab: np.ndarray, n: int = TOP_WORDS_PER_TOPIC) -> list[str]:
    top_idx = np.argsort(components_row)[::-1][:n]
    return [vocab[i] for i in top_idx]


def sweep_k(tfidf: sparse.csr_matrix, binary_doc_term: sparse.csr_matrix, vocab: np.ndarray, vocab_index: dict[str, int]) -> pd.DataFrame:
    rows = []
    for k in K_RANGE:
        model = NMF(n_components=k, init="nndsvd", random_state=RANDOM_STATE, max_iter=500)
        model.fit(tfidf)
        topic_coherences = [
            umass_coherence(top_words_for_topic(row, vocab), binary_doc_term, vocab_index)
            for row in model.components_
        ]
        rows.append({"k": k, "mean_umass_coherence": float(np.nanmean(topic_coherences))})
    return pd.DataFrame(rows)


def fit_final_nmf(tfidf: sparse.csr_matrix, k: int) -> NMF:
    model = NMF(n_components=k, init="nndsvd", random_state=RANDOM_STATE, max_iter=500)
    model.fit(tfidf)
    return model


def doc_topic_dataframe(model: NMF, tfidf: sparse.csr_matrix, base_df: pd.DataFrame, k: int) -> tuple[pd.DataFrame, list[str]]:
    doc_topic = model.transform(tfidf)
    doc_topic_norm = doc_topic / doc_topic.sum(axis=1, keepdims=True).clip(min=1e-12)
    topic_cols = [f"nmf_topic_{i:02d}" for i in range(k)]
    topics_df = pd.DataFrame(doc_topic_norm, columns=topic_cols)
    result = pd.concat(
        [base_df[["Song", "Artist", "Year", COUNTRY_COL, TARGET_COL]].reset_index(drop=True), topics_df],
        axis=1,
    )
    return result, topic_cols


def evaluate_topic_set(result: pd.DataFrame, topic_cols: list[str], label: str) -> pd.DataFrame:
    """Same GroupKFold-by-Year prediction pipeline used for the NLI-based
    theme features (lyrics_model_comparison.evaluate_feature_set) - just a
    different feature set, so the two are directly comparable."""
    pred_df = result.copy()
    pred_df[TARGET_COL] = pd.to_numeric(pred_df[TARGET_COL], errors="coerce")
    pred_df["Year"] = pd.to_numeric(pred_df["Year"], errors="coerce")
    pred_df[COUNTRY_COL] = pred_df[COUNTRY_COL].map(clean_text)
    pred_df = pred_df.dropna(subset=[TARGET_COL, "Year", *topic_cols])
    pred_df = pred_df[pred_df[COUNTRY_COL] != ""].copy()
    year_totals = pred_df.groupby("Year")[TARGET_COL].transform("sum")
    pred_df["Score_norm"] = pred_df[TARGET_COL] / year_totals
    pred_df = pred_df.reset_index(drop=True)

    folds = evaluate_feature_set(pred_df, topic_cols)
    summary = summarize(folds)
    summary.insert(0, "feature_set", label)
    summary.insert(2, "n_features", len(topic_cols))
    return summary


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    if df.columns[0] == "":
        df = df.drop(columns=df.columns[0])
    df = df.copy()

    df["English_text"] = build_english_text(df)
    df["cleaned_text"] = [
        clean_and_stem(text, artist) for text, artist in zip(df["English_text"], df["Artist"])
    ]
    has_text = df["cleaned_text"].str.strip().str.len() > 0
    print(f"{len(df)} rows total, {has_text.sum()} with usable cleaned text after preprocessing")
    df = df[has_text].reset_index(drop=True)

    tfidf_vectorizer = TfidfVectorizer(min_df=3, max_df=0.7)
    tfidf = tfidf_vectorizer.fit_transform(df["cleaned_text"])
    vocab = tfidf_vectorizer.get_feature_names_out()
    vocab_index = {w: i for i, w in enumerate(vocab)}
    print(f"vocabulary size after cleaning: {len(vocab)} stems")

    count_vectorizer = CountVectorizer(vocabulary=vocab_index)
    binary_doc_term = (count_vectorizer.fit_transform(df["cleaned_text"]) > 0).astype(int)

    print(f"\nSweeping K in {K_RANGE.start}..{K_RANGE.stop - 1}, computing UMass coherence per K...")
    coherence_curve = sweep_k(tfidf, binary_doc_term, vocab, vocab_index)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    coherence_curve.to_csv(OUTPUT_DIR / "lyrics_nmf_coherence_by_k.csv", index=False)
    print(coherence_curve.to_string(index=False))
    print(
        "\nNote: UMass coherence here decreases roughly monotonically with K (no clear elbow), "
        f"so its raw argmax (K={int(coherence_curve.loc[coherence_curve['mean_umass_coherence'].idxmax(), 'k'])}) "
        "isn't a meaningful topic count on its own - same caveat the paper itself flagged by picking K "
        "from visual inspection of the curve rather than a blind max. The mapping/prediction steps below "
        f"instead use K={PRIMARY_K} (the paper's own choice) as the primary topic count, and separately test "
        "prediction across a spread of K values."
    )

    # Primary mapping output: fixed at the paper's own K=15, saved for
    # descriptive use (what themes appear in Eurovision lyrics) independent
    # of whether it turns out to predict anything.
    primary_model = fit_final_nmf(tfidf, PRIMARY_K)
    primary_result, primary_topic_cols = doc_topic_dataframe(primary_model, tfidf, df, PRIMARY_K)
    primary_result.to_csv(OUTPUT_DIR / "lyrics_nmf_topics.csv", index=False)

    word_rows = []
    for topic_idx, component_row in enumerate(primary_model.components_):
        for word in top_words_for_topic(component_row, vocab):
            word_rows.append({"topic": f"nmf_topic_{topic_idx:02d}", "word": word})
    pd.DataFrame(word_rows).to_csv(OUTPUT_DIR / "lyrics_nmf_topic_words.csv", index=False)
    print(f"\nTop words per topic (K={PRIMARY_K}, saved to outputs/lyrics_nmf_topics.csv / _topic_words.csv):")
    for topic_idx, component_row in enumerate(primary_model.components_):
        words = top_words_for_topic(component_row, vocab)
        print(f"  nmf_topic_{topic_idx:02d}: {', '.join(words)}")

    # Stage B: does the NMF topic-weight vector predict Score_norm any
    # better than the NLI-based theme features did, at any topic count?
    # Same evaluate_feature_set GroupKFold-by-Year pipeline as the NLI
    # comparison, just a different feature set each time.
    summaries = []
    for k in sorted(set(REPORT_K_VALUES) | {PRIMARY_K}):
        model = primary_model if k == PRIMARY_K else fit_final_nmf(tfidf, k)
        result, topic_cols = doc_topic_dataframe(model, tfidf, df, k)
        summaries.append(evaluate_topic_set(result, topic_cols, f"nmf_topics_k{k}"))

    nmf_summary = pd.concat(summaries, ignore_index=True)
    nmf_summary.to_csv(OUTPUT_DIR / "lyrics_nmf_prediction_summary.csv", index=False)

    print(f"\n5-fold GroupKFold by Year, across K = {sorted(set(REPORT_K_VALUES) | {PRIMARY_K})}:")
    print(nmf_summary[nmf_summary["model"].isin(["themes_only", "combined"])].round(4).to_string(index=False))

    comparison_path = OUTPUT_DIR / "lyrics_feature_set_comparison_summary.csv"
    if comparison_path.exists():
        prior = pd.read_csv(comparison_path)
        combined = pd.concat([prior, nmf_summary], ignore_index=True)
        print("\nFor context, alongside the previously evaluated NLI-based feature sets:")
        print(combined[combined["model"].isin(["themes_only", "combined"])].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
