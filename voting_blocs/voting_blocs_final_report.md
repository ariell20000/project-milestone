# Eurovision Voting Blocs: Does Voting Reflect Musical Taste or Politics?

**Course project deliverable.** This report synthesizes four independent analyses of the same question, each covering a different course-syllabus method, run against the same underlying data (`eurovision_1957-2021.csv`, the country-to-country point-giving records, 2004-2021). Each piece has its own detailed report with full methodology, sensitivity checks, and sourcing; this document is the connective narrative between them and the answer they jointly support.

| # | Piece | Syllabus topic(s) | Script | Full report |
|---|---|---|---|---|
| 1 | Similarity & hierarchical clustering | Distance measures / similar items; clustering, non-Euclidean spaces | `voting_blocs_similarity.py`, `voting_blocs_clustering.py` | `outputs/voting_blocs_similarity_clustering_report.md` |
| 2 | Graph & community detection | Graph analysis, social networks, community detection | `voting_blocs_graph.py` | `outputs/voting_blocs_graph_report.md` |
| 3 | Recommendation system | Recommendation systems | `voting_blocs_recsys.py` | `outputs/voting_blocs_recsys_report.md` |
| 4a | Statistical inference | Statistical inference | `voting_blocs_inference.py` | `outputs/voting_blocs_inference_report.md` |
| 4b | Causal / experimental design | Experimental design, causality | `voting_blocs_causal.py` | `outputs/voting_blocs_causal_report.md` |

All four were run as independent investigations against the shared, cleaned votes data (`main.py`'s `load_data`/`clean_data`/`normalize_country`), so their agreement (or disagreement) is not an artifact of one shared pipeline choice.

---

## 1. Research question and design

Eurovision's public reputation is that voting is "political" - neighbors and diaspora communities reward each other regardless of the song. The professional jury system, introduced in its current form in 2009 and split from the televote in 2016, exists partly to counteract this. The question this project asks, methodically rather than anecdotally: **how much of Eurovision voting is explained by shared musical taste, and how much by geographic/political affinity - and are those even separable?**

Four independent methods were aimed at this question because no single one can answer it alone:

- **Clustering** finds *whether* vote-pattern blocs exist at all, from nothing but the numbers.
- **Graph/community detection** asks whether the *jury* and the *public* structure those blocs the same way.
- **A recommender system** quantifies, in prediction terms, how much of a ballot is "the song" versus "who's voting."
- **Causal analysis** asks the sharper question: does the bloc effect survive once you control for what the songs actually sound like, and is it even a defensible causal claim to begin with.

## 2. Part 1 - Similarity and clustering

**Method.** Each country's outgoing vote pattern (points given to every other country) becomes a feature vector, first divided by the number of editions where both countries could interact (opportunity normalization - raw totals mostly measure how many finals a country reached) and then centered by recipient (so two countries aren't "similar" merely because they both reward whatever the whole continent rewards). Cosine similarity on these residuals was chosen over Jaccard and Pearson after directly comparing all three on the same pipeline; cosine-on-centered-rates was the only one that produced balanced, non-degenerate clusters (silhouette 0.195 vs 0.045 raw-cosine vs 0.068 Jaccard). Clustering itself is hierarchical clustering on the precomputed cosine-distance matrix - deliberately not k-means, since there are no coordinates here, only pairwise distances (the course's "non-Euclidean spaces" point made concrete). Complete and average linkage were both checked at the chosen k rather than picking one by convention; average linkage currently wins (silhouette 0.195 vs 0.183) and is what the pipeline uses.

**Result.** k=9 clusters, chosen at the silhouette peak (0.195), recover real-world geopolitical blocs *without ever being shown geography*: a perfect ex-Yugoslav cluster (Croatia, Montenegro, Serbia, Slovenia), a Western-European core, a Nordic-plus group (the five Nordics joined by Estonia, Australia and the UK), a Caucasus/Black-Sea group (Belarus, Georgia, Russia, Turkey), and a Balkan/Mediterranean group. For the map, and for comparing against the graph-community method below, these 9 fine clusters are further coarsened into 3-4 families by cutting the same dendrogram at a higher level (3 families for the graph comparison, matching Louvain's 3 communities; 4 for the map's colour legend). The most informative results are the *exceptions*: Armenia and Azerbaijan - politically hostile, near-zero direct points to each other - are *not* merged at the fine k=9 level (their profiles land in neighbouring but distinct clusters), yet their pairwise similarity is still well above average (Azerbaijan is Armenia's 7th-closest match of 42) and the coarser 3-family cut reunites them; run on the televote ballots alone, they land in the same fine cluster outright, while the jury-only view keeps them apart - consistent with Part 2's finding that the public vote encodes bloc structure more strongly than juries do. Estonia sits with the Nordics rather than the Baltics (Latvia and Lithuania instead land with Ireland, Poland and Ukraine); the weakest, least geographically-coherent clusters are Hungary/Romania/Spain and Azerbaijan/Czech Republic/Israel/Moldova, plausibly diaspora-driven rather than bloc-driven.

**Output used downstream:** `outputs/voting_blocs_clusters.csv` (`country, cluster_id`) is the bloc-membership definition consumed by both the inference and causal pieces below.

## 3. Part 2 - Graph analysis and community detection

**Method.** A directed, weighted graph (nodes = countries, edge weight = total points A→B) was built for the full 2004-2021 window and, separately, for the 2016-2021 jury-only and televote-only ballots. Louvain community detection was chosen after research into directed-graph modularity (Leicht & Newman; Dugué & Perez's Directed Louvain) confirmed both a directed and a symmetrized (reciprocal-weight) version were appropriate; both were run and agreed almost perfectly (ARI = 1.000), so the symmetrized version is reported for comparability. Girvan-Newman was also tried and explicitly reported as a *negative result*: at 88% edge density there are no bridges to cut, so it just peels off singletons (modularity 0.0014 vs. Louvain's 0.13) - a genuine research finding about which algorithm suits a near-complete graph, not a coding failure.

**Result - the project's single strongest piece of evidence.** Modularity alone can't distinguish jury (Q=0.140) from televote (Q=0.148) communities - they have equally *much* structure. But a permutation test on geographic compactness (mean distance between same-community countries vs. 2,000 random relabellings) shows the *content* of that structure differs sharply: televote communities are 15% more geographically compact than chance (p<0.0005); jury communities are only 11% tighter (p<0.0005) and its most dispersed community spans the entire map (Australia to Latvia, mean 1,837 km apart). Since jury and televote score the *identical* performances on the *identical* night, this difference cannot be about the songs - it is a property of who is voting. Countries whose community becomes most geographically scattered moving from televote to jury: North Macedonia, Serbia, San Marino, Belgium, Croatia, Hungary.

## 4. Part 3 - Recommendation system

**Method.** Reframed as a genuine recommender problem at the (voter, specific song-entry, year) grain rather than a re-skinned country similarity matrix - "given a voter's history, which of this year's 25 songs will it reward?" Absent ballots were carefully distinguished from real zeros (only countries that actually voted that round contribute to the grid). Following Hu, Koren & Volinsky's implicit-feedback framework, a biased matrix-factorization model (`points ≈ overall mean + voter offset + entry-quality + latent voter·entry affinity`) was fit by masked ALS and compared against three alternatives plus two naive baselines, with hold-out by whole voter-year (not by cell, to avoid leaking the other 9 awards on the same ballot).

**Result.** The full model beats the naive baseline decisively (RMSE 2.99 vs. 3.61, p≈1e-26) - but the *decomposition* is the finding: splitting the improvement over baseline between "entry quality" (everyone agrees this song is good) and "voter-specific affinity" (the bloc term) gives **79-91% to entry quality and only 9-21% to bloc affinity**, on average. Critically, "small on average" hides that it is *concentrated*, not diffuse: Russia's predicted top-4 point recipients (Azerbaijan, Armenia, Belarus, Georgia) exactly match its actual top-4, with lifts of +3 to +6 points above what the field gives those songs; the UK, by contrast, shows almost no bloc lift (max +1.3) and votes close to pure consensus. Answer in the recommender's own terms: Eurovision voting is mostly a music contest (roughly 80-90% predictable from consensus quality) with a small, highly concentrated political layer bolted on for specific country clusters.

## 5. Part 4a - Statistical inference

**Method.** Uses 2016-2021, the only years with a jury/televote split, at the (voter, recipient, year) dyad level (N=2,883). The outcome is `gap = jury points - televote points` for each dyad - a within-dyad control that differences out everything about the song itself, since both ballots score the same performance the same night. Because dyads sharing a voter or a recipient are *not* independent observations (a documented failure mode where naive t-tests on network/dyadic data can have >50% true type-I error), inference uses **QAP permutation** (Mantel/Krackhardt quadratic assignment procedure - relabel which countries are in which bloc, 20,000 times, keeping the whole network structure intact) rather than a t-test, plus a node-level bootstrap for confidence intervals.

**Result - a genuine surprise that reframes the whole project.** Testing "is the average within-bloc dyad more televote-favored than the average between-bloc dyad" comes back **not significant** (T=-0.37 points, QAP p=0.194). That could look like a null result for bloc effects generally - except a *third* test, on whether blocs differ *from each other* in how their gap splits, is emphatically significant (weighted SD of bloc-level gaps = 2.00 vs. 0.80 expected by chance, p<0.00005). The resolution: there is no single "bloc effect" with one sign. Four blocs (led by the ex-Yugoslav trio, jury 0.33 pts vs. public 9.33 pts to each other) are overwhelmingly televote-driven; four others (Western-European core, eastern-Mediterranean) lean the *opposite* way, with juries favoring their partners *more* than the public does. Averaging across all eight blocs cancels these opposite signs out - which is exactly why the naive "bloc effect" test found nothing.

## 6. Part 4b - Causal analysis and experimental design

**Method.** Same 2,883-dyad panel. The question: does bloc membership predict points given, and does that survive controlling for how similar the two countries' songs actually are lyrically (cosine similarity on `eurovision_enriched2.csv`'s lyric-theme vectors, standardized within year)? Ten regression specifications were fit, adding controls progressively: lyrics similarity, shared performance language, recipient×year fixed effects (comparing only voters of *the same song, same night* - absorbing song quality, staging, and running order entirely), a second independent lyrics-text similarity measure (TF-IDF), jury-only vs. televote-only splits, and - critically - bloc labels **re-derived from 2004-2015 votes only**, so treatment cannot be a function of the very 2016-2021 outcome being predicted.

**Result.** Sharing a bloc is worth **+2.67 points per dyad** (mean 7.14 vs. 4.50), and that number barely moves across every specification: 2.66 with lyrics similarity controlled, 2.67 with shared language, 2.70 with full song fixed effects. The out-of-sample holdout (blocs from 2004-2015, tested on 2016-2021) still shows a clear, significant effect (+1.13, p=0.026) in *both* the jury ballot (+1.21) and the televote (+1.49) separately. **Honest complication the piece surfaced itself**: the lyric-theme control turned out to be measurement-degenerate (248 songs collapse into only 136 distinct theme vectors, and the measure correlates 0.73 with an independent text-similarity check overall but only 0.38 among same-language songs - i.e., it partly just encodes *language*, not content), so "the bloc effect survives controlling for lyrics" is weaker evidence than it looks on its own; the fixed-effects and out-of-sample specifications are what actually carry the conclusion, since they don't depend on that flawed feature.

A genuinely new finding beyond "does the effect survive": interacting bloc membership with song similarity shows the bloc premium **grows** with how similar the songs are (+1.53 pts/SD, replicated on the independent TF-IDF measure at +1.05 pts/SD) - bloc partners and shared taste are *complements*, not substitutes. The nearly obligatory-but-correct causal-design paragraph: this is not a randomized trial, `same_bloc` is an outcome of centuries of migration/broadcasting/geography rather than an assigned treatment, and three concrete unmeasured pair-level confounds remain even after all controls - diaspora population size, shared broadcast/streaming market exposure, and true mutual intelligibility (vs. the crude same-performance-language proxy, which is mostly an English/non-English indicator since 70% of entries are in English). The contest also structurally violates SUTVA (every ballot is a fixed 58-point budget, so one country's gain is mechanically another's loss) - the coefficient is best read as a *relative allocation* effect, not an absolute causal effect in the textbook RCT sense. The piece names the quasi-experiments a stronger design would exploit: the semi-final draw (partially randomized within pots), the 2016 rule change (a genuine policy shock, enabling difference-in-differences), and running order.

## 7. Synthesis: how the four pieces fit together

At first glance, two numbers look like they disagree: the recommender says only 9-21% of predictable ballot variation is "bloc," while the causal analysis finds a robust, hard-to-explain-away +2.67-point bloc premium. They are not actually in tension - they're answering different questions on different scales. The recommender decomposes *variance explained in a ranking/error sense across the whole dataset*; the causal estimate is *the average point gap for the specific pairs that are bloc partners*, most of whom are a small, concentrated subset of all possible country pairs (only 344 of 2,883 dyads are within-bloc). A small share of *total* explained variance is fully consistent with a large, real effect concentrated in a *few* relationships - which is exactly what the recommender's own lift analysis shows directly (near-zero for the UK, +5-10 points for Russia/Moldova/Cyprus-Greece pairs).

The full picture, triangulated across four independent methods:

1. **Bloc structure is real and recoverable from nothing but voting numbers** (clustering), and it substantially - not perfectly - overlaps real-world geography and politics.
2. **The bloc signal is not uniform across who's voting**: the public (televote) encodes it geographically; professional juries encode *some* structure too, but not a geographic one (graph/community detection) - and statistically, different blocs favor *different channels* for the same underlying loyalty, jury for some, televote for others (statistical inference) - so "does the jury fix bloc voting" has no single yes/no answer; it depends which bloc.
3. **In aggregate, taste dominates**: 80-90% of what's predictable about any single ballot is consensus quality, not politics (recommender).
4. **But where the bloc effect exists, it is large, causally well-controlled-for, and not explained by song content** (causal analysis) - and it *amplifies* rather than competes with shared taste, growing stronger precisely when the songs are also similar.

**Answer to the research question.** Eurovision voting is not "taste XOR politics." It is overwhelmingly taste on average, with a real, robust, geographically- *and* politically-concentrated bloc layer riding on top of it for a specific, identifiable subset of country pairs - a layer that operates through different channels (jury vs. televote) for different blocs, that persists under the strongest available observational controls, and that this project's data cannot fully separate from "shared cultural exposure" (as opposed to political favoritism), because no dataset used here measures diaspora size, cross-border media exposure, or genuine mutual intelligibility directly.

## 8. Limitations (project-wide)

- All four pieces are **observational**; only Part 4b makes a causal claim, and it is explicit about not being an RCT.
- The jury/televote split covers only **five contests (2016-2021, excluding cancelled 2020)** - the statistical-inference piece's own power analysis shows its primary test could only detect an effect 2.2× larger than what was observed, so "not significant" there is properly read as *inconclusive*, not *no effect*.
- Bloc labels come from **clustering the same vote data whose patterns are later tested** - partially addressed by the causal piece's out-of-sample holdout (blocs from 2004-2015), which still confirms the effect but doesn't fully eliminate the concern, since underlying relationships between countries are persistent across periods.
- The lyric-theme features in `eurovision_enriched2.csv` are **measurement-degenerate** (documented in Part 4b) - a data-quality finding worth fixing before this control is trusted further, e.g. by re-deriving theme scores or leaning more on the TF-IDF alternative.
- 6 countries could not be clustered (too few voting editions) and are excluded from every downstream analysis: Bosnia and Herzegovina, North Macedonia, Montenegro, San Marino, Czech Republic, Serbia and Montenegro.

## 9. Reproducing

All scripts and outputs live in this `voting_blocs/` directory; run the commands below with this directory as the working directory. They read three files from the parent (repo-root) directory - `eurovision_1957-2021.csv`, `eurovision_enriched2.csv`, and the shared helper modules `main.py`, `jury_televote.py`, `escxtra_country_mapping.py` - which each script locates automatically.

```bash
python3 voting_blocs_similarity.py      # Part 1a
python3 voting_blocs_clustering.py      # Part 1b - writes outputs/voting_blocs_clusters.csv
python3 voting_blocs_graph.py           # Part 2
python3 voting_blocs_recsys.py          # Part 3
python3 voting_blocs_inference.py       # Part 4a - reads voting_blocs_clusters.csv
python3 voting_blocs_causal.py          # Part 4b - reads voting_blocs_clusters.csv
```

Parts 1a/1b must run before 4a/4b (both depend on `outputs/voting_blocs_clusters.csv`); 2 and 3 are independent of everything else. All outputs are written under `outputs/voting_blocs_*` inside this directory.

## 10. Key figures

- `outputs/voting_blocs_dendrogram.png`, `outputs/voting_blocs_similarity_heatmap.png` - Part 1
- `outputs/voting_blocs_network_comparison.png` - Part 2 (full / jury / televote side by side)
- `outputs/voting_blocs_recsys.png` - Part 3
- `outputs/voting_blocs_inference_permutation.png`, `outputs/voting_blocs_inference_by_bloc.png`, `outputs/voting_blocs_inference_gap_distribution.png` - Part 4a
- `outputs/voting_blocs_causal_coefficients.png`, `outputs/voting_blocs_causal_similarity.png` - Part 4b
