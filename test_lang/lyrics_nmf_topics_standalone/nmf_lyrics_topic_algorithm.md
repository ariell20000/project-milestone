# NMF Lyrics-Topic Algorithm — extracted from Levy, Granot & Peres (2024)

**Source:** Levy, S., Granot, R., & Peres, R. (2024). "Lyrics do matter: how 'coping songs' relate to well-being goals. The COVID pandemic case." *Frontiers in Psychology.*

**What the paper actually does, in one sentence:** it (1) turns each song's lyrics into a bag of cleaned, stemmed words, (2) factorizes the resulting corpus with NMF into a small number of latent topics, (3) names those topics via a human/LLM pipeline, and (4) regresses each song's topic-weights against the *listener's* self-reported well-being goals to see which topics people needing which kind of psychological support gravitate toward.

**What it is *not*:** the paper never tries to predict an external, objective outcome (chart position, competition score, etc.) from the topics. The "prediction" in the paper runs the other direction — from person-level psychological traits to song-topic choice, not from song content to song outcome. This matters for how we'd reuse it (see the consult section in chat).

The pipeline has two genuinely separable halves:
- **Part A (Stages 1–3): unsupervised topic extraction** — this is the reusable, corpus-only part.
- **Part B (Stages 4–6): topic naming + the well-being regression** — this is specific to their study design (self-report goals, MTurk, placebo test) and largely doesn't transfer to us as-is.

---

## Stage 0 — Data acquisition (context, not part of the reusable algorithm)

- 2,804 "coping songs" nominated by 5,619 respondents across 11 countries during COVID.
- Lyrics scraped from YouTube's video-info panel via a custom browser plugin.
- Non-English lyrics (953 of 2,804 songs) machine-translated to English via Google Translate.
- Genre metadata pulled separately (PyTube → video title → ChatGPT extracts artist/song name → ChatGPT again infers genre). Not used by the topic model itself.

Not directly relevant to us — we already have a clean `Lyrics` / `Lyrics translation` column per song, no scraping needed.

---

## Stage 1 — Text preprocessing

Applied to every song's lyric text before modeling:

1. **Stopword removal** — generic function words ("is", "the", "chorus", etc.). *Paper does not name a specific stopword list.*
2. **Artist-name removal** — the performing artist's name(s) stripped out of the lyric text (presumably to stop it dominating a topic just because it's repeated/printed in the lyric block). *Not a generic library operation — needs the artist name per song as an input.*
3. **Non-word syllable removal** — filler vocalizations like "haa," "wawa" that aren't real words. *Paper gives no formal definition or method for detecting these — this is the vaguest step in their description.*
4. **Punctuation removal.**
5. **Number removal.**
6. **Stemming** — inflected words collapsed to a root form (e.g., "likes"/"liking" → "like"). *Paper does not name the stemmer (Porter vs. Snowball vs. Lancaster, etc.).*
7. Result: a corpus-wide lexicon of **~6,000 words**.
8. **Deduplication** — when the same song was nominated by multiple respondents, it's counted once, not once per respondent (avoids popular songs dominating the topic model). Corpus shrinks from 2,804 nominations to **2,386 unique songs**. *Not relevant to us — our dataset is already one row per song, no repeat-nomination structure.*

## Stage 2 — Document-term matrix

The cleaned, stemmed lyrics are converted into a document–term matrix (songs × vocabulary) for NMF to factorize. **The paper does not explicitly state whether this is raw term counts or TF‑IDF weighted** — TF‑IDF is the near-universal default for NMF topic modeling in the literature (raw counts let long/repetitive songs and generic-but-frequent words dominate), so it's a reasonable assumption but not a stated fact from the paper.

## Stage 3 — NMF topic modeling (the core, reusable algorithm)

- **Method:** Non-negative Matrix Factorization. The paper's stated rationale for NMF over LDA specifically for song lyrics: LDA is a probabilistic (Dirichlet) model that "often fail[s] to model the topic structure in corpuses of songs" because song lyrics have non-standard grammar, are short, and are highly repetitive (choruses) — properties that violate LDA's implicit assumptions about document generation. NMF, being a purely linear-algebra factorization, degrades more gracefully on short/sparse/repetitive text.
- **What it produces**, given a document-term matrix V (songs × vocabulary) and a chosen topic count K:
  - **W** (songs × K): each song's weight/membership across the K topics.
  - **H** (K × vocabulary): each topic's weight across every word in the vocabulary.
  - V ≈ W·H, with the non-negativity constraint on both W and H (unlike PCA/SVD, no negative weights — every topic contributes non-negatively to every document, which is part of why NMF topics tend to be more human-interpretable as "themes").
- **Choosing K (number of topics):** they swept K from 1 to 30, computing a **coherence score** for each K using **`gensim.models.CoherenceModel`**, then picked K based on inspecting the coherence-vs-K curve (a qualitative "where does it plateau/peak" read, not a hard automatic rule). They landed on **K = 15**.
- **Topic content:** for each of the 15 topics, the top 10 highest-weighted words (from that topic's row in H) characterize it — reported in their Table 1.
- **Per-song topic distribution:** for interpretability/reporting, a song's raw W-row is normalized so its topic weights sum to 100% (e.g., their Figure 3B example: "Forever Young" = 63% one topic, 15% another topic, etc.) — i.e., L1-normalize each row of W. This turns each song into a probability-like distribution over the 15 topics, which is the natural per-song feature vector this stage produces.

**This is the part of the algorithm most directly reusable for us** — the output of Stage 3 is, for every song, a 15-dimensional (or however many K we pick) numeric feature vector, structurally identical in shape to the 100 NLI-based theme scores we've already been feeding into `score_model.py` / `lyrics_feature_selection.py`.

## Stage 4 — Topic naming (human-in-the-loop; optional for us)

Purely to attach a human-readable label to each of the 15 numbered topics, not required to *use* the topic vectors:

1. Top-10-words list per topic shown to 40 MTurk workers (paid $1 each), asked to propose a title capturing the commonality among the words.
2. The same 10-word list separately given to ChatGPT with a "find the common denominator between these words" prompt.
3. MTurk responses + ChatGPT response integrated (manually, by the researchers) into one final title per topic.
4. **Validation:** 210 different MTurk respondents shown a topic's top-10 words plus two candidate titles — the real one and a random title stolen from a different topic — and asked to pick the one that fits. Used to statistically confirm the assigned titles are meaningfully better than chance.

Since our downstream goal is prediction, not producing a human-readable report, this stage is **skippable** — a numbered "Topic 7" feature column works exactly as well as a named one for a Ridge/GroupKFold pipeline. Worth doing later only if we want a plain-English write-up of what the topics "mean."

## Stage 5 — Downstream regression: topic weight ~ listener's well-being goals

For **each of the 15 topics separately** (15 independent OLS regressions), on the respondent-level data:

y_ij = α₀ + ᾱ·goal_j + β̄·(goal_j × music_importance_j) + γ̄·X_j + ε

- **y_ij**: respondent *j*'s chosen song's weight on topic *i* (i.e., the corresponding entry of the L1-normalized W row for that song).
- **goal_j**: the respondent's 0–4 Likert ratings across 5 predefined well-being goals (the paper's independent variable of primary interest).
- **goal_j × music_importance_j**: interaction of each goal rating with the respondent's self-rated importance of music in their life.
- **X_j**: control variables — age, gender, number of children, relationship/personal status, religiosity, spirituality.
- Significance assessed at p < 0.05, coefficients reported in a color-coded significance matrix (their Tables 2/3) — topics × goals grid, colored by sign/significance.

**This entire stage is specific to their study** — it needs individual-respondent psychological self-report data (goals, music-importance, demographics) that has no Eurovision analogue. Not reusable for us in this form. (See consult section below for what *would* be the structurally analogous thing for us.)

## Stage 6 — Statistical validation: placebo permutation test

To rule out the Stage 5 regression coefficients being spurious/overfit:

- For each respondent, their 5 goal ratings are **randomly permuted among themselves**, preserving that respondent's own mean rating level (so a respondent who rates everything "3" on average still rates everything "3" on average post-shuffle, just reassigned to different goals).
- The 15 regressions are rerun on this permuted (placebo) data.
- Expectation confirmed: coefficients on the placebo data are statistically null, unlike the real coefficients — supporting that the real associations aren't an artifact of the regression setup or multiple comparisons. (Method attributed to Abadie & Gardeazabal 2003.)

This is a generic, reusable **validation technique** (not dependent on their specific variables) — worth keeping in mind for *any* regression we run against a topic-weight target, including a Eurovision one, as a sanity check against spurious "signal."

## Stage 7 — Parallel acoustic-feature analysis (not applicable to us)

Alongside lyrics, they ran the identical regression design (Stage 5 structure, without the music-importance interaction term) using **acoustic** features — Loudness, Mode, Tempo, Harmony, Timbre — extracted from the audio (WAV files) via the **Essentia** library. We have no audio files in our Eurovision dataset, so this stage has no counterpart for us — noted here only for completeness of the paper's pipeline.

---

## Full pipeline at a glance

| Stage | Purpose | Reusable for us? |
|---|---|---|
| 0. Data acquisition | Scrape lyrics + translate | No — we already have lyrics/translations |
| 1. Preprocessing | Clean + stem lyric text | **Yes** (standard NLP text-cleaning) |
| 2. Document-term matrix | Vectorize corpus | **Yes** (standard vectorization) |
| 3. NMF topic modeling | Extract K latent topics, per-song topic-weight vector | **Yes — this is the core algorithm** |
| 4. Topic naming | Human-readable topic labels | Optional, skippable for prediction |
| 5. Topic ~ goals regression | Their specific research question | No direct analogue (no listener self-report data) |
| 6. Placebo permutation test | Validate regression isn't spurious | **Yes, as a general validation technique** |
| 7. Acoustic features | Parallel audio-based analysis | No — no audio files available |
