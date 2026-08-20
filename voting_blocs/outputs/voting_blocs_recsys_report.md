# Piece C — Recommender system for Eurovision voting blocs

**Question this piece answers:** how much of a country's ballot can be predicted from
*who is voting* (bloc affinity) rather than from *what is being voted on* (the song's
general appeal)? A recommender is the natural tool: it forces the two effects into
separate, measurable terms.

Script: `voting_blocs_recsys.py` · Window: contest years **2004–2021** ·
Source: `eurovision_1957-2021.csv`, cleaned with `main.clean_data`.

---

## 1. Problem framing

The interaction grain is **(voter country, competing entry)**, where an *entry* is a
specific `(Year, To)` song — not a country. This matters: at the country-to-country
grain the exercise collapses into a static similarity matrix (which the
clustering/graph pieces already build). At the entry grain the model has to answer a
genuine recommender question: *given everything this voter has rewarded in past
contests, which of this year's 25 songs will it hand its 12 points to?*

**Implicit zeros are constructed carefully.** A Eurovision ballot only reports its ten
favourite entries, so an absent `(From, To, Year)` triple usually means "zero points" —
but only if that country actually cast a ballot in that year and round. The grid is
therefore built per `(Year, Points type)` group from the voters and recipients that
really appear in that group, never from the union over all years
(`build_interactions`). Treating "never had the chance to vote" as a zero would invent
tens of thousands of fake negative observations and bias every country that entered the
contest late (Australia, San Marino) or missed a final.

Resulting dataset:

| Quantity | Value |
| --- | --- |
| Ballot × candidate cells | 12,800 |
| Ballots (Year × round × voter) | 543 |
| Voter-years | 414 |
| Voter countries / entries | 43 / 420 |
| Non-zero cells | 5,430 (42.4%) |

Two data quirks drive several design choices:

1. **Fixed budget.** Every ballot awards exactly 1,2,…,8,10,12 = 58 points to exactly
   ten entries. Points are a *ranking allocation*, not an independent rating, and a
   voter's mean points-per-candidate is fixed by the size of the field, not by taste.
2. **Two rounds since 2016.** From 2016 each country files a separate jury ballot and
   televote ballot. Both are kept as observed ballots; for the factorization the two
   rounds of one voter-year are averaged, which puts every year on one comparable 0–12
   scale (`aggregate_cells`).

---

## 2. Method research and choice

Research (web) on collaborative filtering for implicit, sparse, small-count feedback
converged on one distinction, from Hu, Koren & Volinsky (2008), *Collaborative
Filtering for Implicit Feedback Datasets*: **explicit feedback encodes preference,
implicit feedback encodes confidence.** With 1–5 star data the value *is* the target,
so a plain SVD/least-squares fit of observed ratings is appropriate. With implicit
counts (plays, clicks, purchases — or here, points) the value tells you how *sure* you
are that a positive preference exists, and the zeros carry information too. The
practical consequences reported in that literature are: (a) prefer the implicit-MF /
ALS family over vanilla SVD, (b) never factorize a zero-filled matrix as if the zeros
were observed ratings, and (c) evaluate on ranking metrics, since the numeric scale is
not really a rating scale.

Eurovision points sit *between* the two regimes, which is why this piece fits and
compares several models rather than asserting one:

* points are counts on a fixed budget (implicit-like: 12 means "most preferred", not
  "12 units of quality"), **but**
* the observation mask is fully known — we know exactly which voter could have rewarded
  which entry — so we do **not** have HKV's core problem of unknown negatives.

**Chosen primary model: biased matrix factorization fitted by masked ALS**
(`MatrixFactorizationRecommender`), i.e.

```
points(voter u, entry i) ≈ mu + b_u + b_i + p_u · q_i
```

with `b_i` = entry appeal, `b_u` = voter offset, and `p_u · q_i` the latent bloc term.
ALS runs **only over cells inside the ballot grid**, so entries a voter never had a
chance to reward contribute nothing to the loss. This is the key reason `svds` on a
zero-filled matrix was rejected: `scipy.sparse.linalg.svds` cannot distinguish a
structural blank from a real zero, and here 57% of the grid is real, informative zeros
while everything outside the grid is meaningless. Damped/regularized biases and an L2
penalty handle the small sample (43 voters, 420 entries).

Three comparison models were fitted with the same protocol:

* **`implicit_als_hkv`** — the literal Hu–Koren–Volinsky formulation: binary preference
  `p = 1{points > 0}` with confidence `c = 1 + α·points`, solved with the same weighted
  ALS routine. Its output is a ranking score, so a linear calibration fitted on the
  training cells maps it back to the 0–12 scale before RMSE.
* **`neighborhood_cf`** — user-based nearest-neighbour CF over voter countries
  (shrunk cosine similarity on bias-residuals): the classic non-factorization baseline.
* **`bias_only_entry_quality`** — the same model with the latent term removed. This is
  the scientifically interesting comparison: it is a pure "how good was the song"
  model with no voter-specific preference at all.

Plus two naive baselines: global mean, and each voter's historical average points
given.

The `implicit` package was not used — it is not installed in this environment and
requires a native build; the HKV objective is 30 lines of NumPy at this data size
(12.8k cells), so it is implemented directly in `weighted_als` and the script stays on
the repo's existing dependency set.

---

## 3. Evaluation protocol

* **Split:** 20% of **whole `(From, Year)` voter-years** held out (`split_by_voter_year`)
  — 105 test ballots / 2,445 cells. Holding out individual cells would leak badly: the
  other nine awards on the same ballot, and (post-2016) the same country's other round,
  nearly determine what is left. Hyper-parameters (`n_factors`, `reg`, `alpha`) are
  tuned on a *second* voter-year split carved out of the training set only
  (`outputs/voting_blocs_recsys_tuning.csv`); best: 4 factors, λ=20 for the MF model,
  16 factors / λ=20 / α=1 for HKV.
* **Error metrics:** RMSE over the whole held-out grid, plus RMSE restricted to cells
  that actually received points.
* **Ranking metrics:** for each held-out ballot, rank all of that year's candidates and
  compare with the ballot's real top-5 — precision@5 and NDCG@5/@10 (gain = actual
  points, so a wrongly-ordered 12 costs more than a wrongly-ordered 1). Ties are broken
  by seeded jitter, so a constant-prediction baseline scores like a random ranking.
* **Robustness:** every model is re-run over 5 different held-out splits
  (mean/σ columns in the metrics CSV), and paired t-tests over the 105 held-out
  ballots test the model-vs-reference differences
  (`outputs/voting_blocs_recsys_significance.csv`). Ballots, not cells, are the unit of
  the test — the ten awards on one ballot sum to 58 by construction and are anything
  but independent.

---

## 4. Results

Held-out performance (primary split; `outputs/voting_blocs_recsys_metrics.csv`):

| Model | RMSE | RMSE (awarded only) | P@5 | NDCG@5 | NDCG@10 | RMSE mean ± σ over 5 splits |
| --- | --- | --- | --- | --- | --- | --- |
| baseline_global_mean | 3.611 | 4.729 | 0.194 | 0.266 | 0.354 | 3.603 ± 0.005 |
| baseline_voter_mean | 3.611 | 4.736 | 0.194 | 0.266 | 0.354 | 3.604 ± 0.005 |
| bias_only_entry_quality | 3.130 | 4.105 | 0.509 | 0.617 | 0.664 | 3.138 ± 0.043 |
| implicit_als_hkv | 3.214 | 4.102 | **0.541** | 0.657 | 0.697 | 3.211 ± 0.056 |
| neighborhood_cf | 3.015 | **3.893** | 0.535 | **0.663** | **0.704** | 3.003 ± 0.051 |
| **matrix_factorization_als** | **2.992** | 3.897 | 0.528 | 0.651 | 0.700 | **2.988 ± 0.067** |

Paired ballot-level tests (105 ballots):

| Comparison | Metric | Difference | t | p |
| --- | --- | --- | --- | --- |
| MF vs. voter-mean baseline | NDCG@5 | **+0.385** | 14.45 | 1.3e-26 |
| MF vs. voter-mean baseline | MSE | **−4.130** | −14.84 | 2.0e-27 |
| MF vs. bias-only | NDCG@5 | +0.033 | 2.40 | 0.018 |
| MF vs. bias-only | MSE | −0.850 | −4.87 | 4.0e-06 |
| neighborhood CF vs. bias-only | NDCG@5 | +0.046 | 3.31 | 0.0013 |
| implicit ALS (HKV) vs. bias-only | NDCG@5 | +0.040 | 4.36 | 3.0e-05 |

Notes on the table:

* The two naive baselines are **identical to three decimals** — a direct consequence of
  the fixed 58-point budget: a country's average points-per-candidate carries no
  information about that country beyond the size of the field.
* The HKV model is the best *ranker* on precision@5 but the worst of the CF models on
  MSE — exactly as expected from a model that optimizes a binary preference with
  confidence weights and only reaches the points scale through a post-hoc calibration.
* The two views of the plot (`outputs/voting_blocs_recsys.png`): predictions increase
  monotonically with the actual award (left), but sit far below the identity line — the
  model correctly orders a ballot while heavily shrinking magnitudes, which is what a
  regularized recommender does with a 12-point outlier it can only partially explain.
  Per-year RMSE (right) beats the baseline in every year except 2010.

---

## 5. Interpretation

**Does the model beat the baseline meaningfully? Yes, overwhelmingly — but almost all
of the gain is "the song", not "the bloc".**

Against the naive baseline the recommender is not close: RMSE 2.99 vs 3.61 (−17%),
precision@5 0.53 vs 0.19, NDCG@5 0.65 vs 0.27 (p ≈ 1e-26). Knowing nothing but the
history of the voting graph, the model puts on average 2.6 of a country's 5 favourite
entries into its own predicted top 5.

The decomposition is the real result. Splitting the total gain over the naive baseline
between the entry-quality term and the voter-specific latent term:

| Component | Share of NDCG@5 gain | Share of MSE gain |
| --- | --- | --- |
| Entry appeal (`b_i` — everyone agrees this song is good) | **91%** | **79%** |
| Voter-specific affinity (`p_u · q_i` — the bloc term) | **9%** | **21%** |

So the dominant predictable signal in a Eurovision ballot is *consensus*: most of what
one country will reward is what every other country is also rewarding that year. The
bloc term is real — it is statistically significant on every metric and every one of
the three CF models beats the bias-only model — but it is a second-order correction in
aggregate, worth ~0.03 NDCG and ~0.85 MSE.

**However, "small on average" is not "small everywhere".** The demo table
(`outputs/voting_blocs_recsys_examples.csv`) ranks each voter's partners twice: by mean
predicted points, and by **bloc lift** (points above the average the *rest of the field*
gave that same entry — the part of the vote that is not the song). By lift, the
learned structure is unmistakable and pair-specific:

* **Russia**: predicted top-4 = Azerbaijan, Armenia, Belarus, Georgia — the exact actual
  top-4, in the exact right order, with real lifts of +5.9, +5.7, +5.4, +3.3 points.
* **Moldova**: all 5 predicted partners (Ukraine, Azerbaijan, Russia, Belarus, Romania)
  are in the true top 5; Moldova gives Romania **+8.0 points** more than the field does.
* **Cyprus ↔ Greece**: mutual lifts of +8.5 and +10.0 points — i.e. each hands the other
  something close to a guaranteed 12 regardless of the song.
* **Sweden** recovers a purely geographic-cultural Nordic cluster (Denmark, Finland,
  Norway, Iceland) with lifts of ~+3, an order of magnitude weaker than the Russia or
  Moldova blocs.
* **United Kingdom** is the negative control: its strongest predicted lift is +1.3
  (Iceland/Denmark/Ireland) and only 2 of 5 predictions land — the UK's ballot is close
  to pure consensus voting with no bloc to exploit.
* An instructive **miss**: Romania's single largest true lift is Moldova (+8.3), but the
  model ranks Moldova only 11th (+0.2) for Romania — even though it nails the same pair
  from Moldova's side. A rank-4 factorization has to express affinity as a *taste
  direction*, and Romania's ballot otherwise looks like a generic Balkan/Mediterranean
  voter, so a single reciprocal relationship with no other structure behind it cannot be
  encoded. This is exactly the case an explicit pair dummy catches (see the sibling
  `pairwise_vote_model.py`) and low-rank CF does not — a real cost of the latent-factor
  formulation, and a reason the aggregate bloc contribution above should be read as a
  *lower* bound.

Across the eight demo countries, 70% of the lift-ranked predictions land in the voter's
true top 5, versus 65% for the raw-points ranking.

**Answer to the project question.** Eurovision voting is neither purely musical nor
purely political, and the two are separable in size and in shape. Shared taste — a
consensus ranking of the songs, which is what a strong entry-quality term measures —
accounts for roughly 80–90% of everything that is predictable about a ballot. Political
and geographic affinity accounts for the remaining 10–20%, but it is *concentrated*: it
is near-zero for Western European voters with no neighbours in the contest, and worth 5
to 10 points per ballot inside the post-Soviet, Balkan/Greek-Cypriot and
Romanian-Moldovan clusters, where it is effectively deterministic. And a large residual
remains: RMSE ≈ 3.0 points against a target standard deviation of ≈ 3.6 means the
history of the voting graph explains only about a third of the variance in an
individual ballot. The rest is the song of the year — which is the strongest evidence
in this piece that Eurovision is mostly a music contest with a small, extremely
predictable political layer bolted on.

---

## 6. Limitations

* Post-2016 the jury and televote ballots of a voter-year are averaged for training and
  receive identical predictions, so genuinely channel-specific taste is unmodelled
  (visible as slightly higher RMSE in 2016+ on the per-year chart).
* The source file contains only finals and only 21–27 of the ~40 voting countries per
  year; an entry that received zero points from *every* voter in the file never appears
  as a candidate at all.
* The demo table is computed with a model fitted on all years (an in-sample sanity
  check, not a held-out result); the held-out numbers are the ones in the metrics table.
* Hyper-parameters are tuned once on the primary training split and reused for the
  5-seed robustness run, which is mildly optimistic for those seeds.
* This is a purely observational, correlational model: it identifies affinity, not its
  cause (diaspora, shared language, geography and genuine musical similarity are not
  separated here — that is the causal piece's job).

## 7. Reproduce

```bash
python3 voting_blocs_recsys.py
```

Writes `outputs/voting_blocs_recsys_metrics.csv`,
`outputs/voting_blocs_recsys_examples.csv`,
`outputs/voting_blocs_recsys_significance.csv`,
`outputs/voting_blocs_recsys_tuning.csv` and `outputs/voting_blocs_recsys.png`
(~6 s, no dependencies beyond the repo's existing pandas/numpy/scipy/sklearn/seaborn).

### Sources consulted

* Hu, Koren & Volinsky (2008), *Collaborative Filtering for Implicit Feedback Datasets*
  — [Semantic Scholar](https://www.semanticscholar.org/paper/Collaborative-Filtering-for-Implicit-Feedback-Hu-Koren/184b7281a87ee16228b24716ca02b29519d52eb5)
* [Collaborative Filtering based Recommender Systems for Implicit Feedback Data](https://blog.reachsumit.com/posts/2022/09/explicit-implicit-cf/)
  — explicit vs. implicit CF, and why zero-filled SVD is the wrong tool for implicit data
* [ALS Implicit Collaborative Filtering](https://medium.com/radon-dev/als-implicit-collaborative-filtering-5ed653ba39fe)
  — confidence weighting `c = 1 + α·r` and the ALS solve
* [Matrix Factorization: The Bedrock of Collaborative Filtering](https://www.shaped.ai/blog/matrix-factorization-the-bedrock-of-collaborative-filtering-recommendations)
  — bias terms + latent factors, ALS vs. SVD trade-offs
