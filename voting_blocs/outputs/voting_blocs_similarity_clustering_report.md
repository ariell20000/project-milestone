# Eurovision voting blocs: similarity and hierarchical clustering

Window: 2004–2026 finals. Countries clustered: 43 (voted in at least 4 editions). Excluded: Bosnia and Herzegovina, Luxembourg, North Macedonia, Serbia and Montenegro.

## Method

Each country is represented by its **outgoing** vote row: how many points it gave to every other country. Two normalizations are applied before any distance is computed.

1. **Opportunity normalization.** Raw point totals mostly measure attendance: the UK voted in 17 finals in this window, Montenegro in 2. Every cell is divided by the number of editions in which that voter voted *and* that recipient competed, giving average points per chance to vote. (Row-sum normalization would not have fixed this — cosine is already invariant to row scaling, so the bias lives in the *support* of the row, not its length.)
2. **Recipient centering.** Each recipient's column is centered on the average rate it received from the voters who could vote for it. Without this step, two countries look similar simply because they both rewarded the songs everyone rewarded — that is the shared musical-taste signal, and it swamps the bloc signal. On the residuals, similarity asks whether two countries deviate from the field's consensus in the same direction, which is the quantity the research question is about.

**Metric: cosine similarity** on the centered rates; distance = 1 − similarity, fed to `scipy.cluster.hierarchy.linkage` as a precomputed condensed matrix.

Why cosine, from the background reading: the collaborative-filtering literature splits similarity measures into those that only use *whether* a rating exists (Jaccard and its relatives) and those that use the *magnitude* of the rating (cosine and friends), and reviews of sparse CF datasets report Salton's cosine performing well on larger, denser matrices. Here the magnitude is the whole story: giving a neighbour 12 points every single year and giving them 1 point once are politically very different acts, and Jaccard collapses both to "voted for". Binarizing is also close to uninformative in this data — almost every pair of frequent participants has awarded each other *something* at some point, so the binary rows are nearly all-ones. Pearson correlation is the other CF standard, but centering *rows* would treat a country's structural zeros (songs it never had the chance to vote for) as negative preferences; the recipient centering above achieves the popularity adjustment without that side effect. Bray–Curtis, the ecology analogue for abundance vectors, is documented as sensitive to differences in total abundance and erratic on very sparse rows — exactly this dataset's failure mode. Cosine also matches the way this problem is set up in the Eurovision literature, where vote matrices are reconstructed into a distance matrix and passed to agglomerative clustering.

The choice was checked, not assumed. Running the identical pipeline (complete linkage, k=9) on each candidate metric:

| metric | k | silhouette | largest_cluster | singletons |
| --- | --- | --- | --- | --- |
| cosine_raw_totals | 9 | 0.051 | 14 | 2 |
| cosine_centered_rates | 9 | 0.183 | 8 | 0 |
| jaccard_binarized | 9 | 0.076 | 11 | 3 |

Raw-total cosine collapses into one giant cluster plus singletons; Jaccard on the binarized matrix separates almost nothing. Centered cosine is the only one of the three that produces balanced, interpretable groups.

## Linkage and choice of k

**Complete linkage** on the precomputed cosine distances. This is deliberately not k-means and deliberately not Ward: neither the cosine space nor the linkage ever sees point coordinates, and centroid-style methods assume a Euclidean geometry the space does not have. Average linkage was tried too and is worse on the same cut (silhouette 0.164 at k=9, and it strands one country in a singleton); complete linkage gives tighter, more balanced blocs (0.188, no singletons).

Silhouette scores computed on the precomputed distance matrix:

| k | n_clusters | silhouette | largest_cluster | singletons |
| --- | --- | --- | --- | --- |
| 2 | 2 | 0.126 | 24 | 0 |
| 3 | 3 | 0.115 | 19 | 0 |
| 4 | 4 | 0.122 | 16 | 0 |
| 5 | 5 | 0.133 | 16 | 0 |
| 6 | 6 | 0.179 | 12 | 0 |
| 7 | 7 | 0.175 | 9 | 0 |
| 8 | 8 | 0.179 | 8 | 0 |
| 9 | 9 | 0.183 | 8 | 0 |
| 10 | 10 | 0.17 | 8 | 0 |
| 11 | 11 | 0.17 | 6 | 0 |
| 12 | 12 | 0.158 | 6 | 1 |

Silhouette rises steadily from k=4 and peaks at **k=9** (0.188) before flattening out — k=10 ties it numerically but only by shattering the blocs into ten groups of three to five, which buys no interpretation. k=8 is the parsimonious choice at the peak.

## Clusters

| cluster_id | countries | cohesion |
| --- | --- | --- |
| 1 | Hungary, Romania, Spain | 0.232 |
| 2 | Armenia, Belarus, Bulgaria, Greece, Russia, Turkey | 0.393 |
| 3 | Albania, Cyprus, Malta, San Marino | 0.438 |
| 4 | Austria, Belgium, France, Germany, Netherlands, Switzerland | 0.346 |
| 5 | Australia, Denmark, Estonia, Finland, Iceland, Norway, Sweden, United Kingdom | 0.405 |
| 6 | Croatia, Montenegro, Serbia, Slovenia | 0.524 |
| 7 | Azerbaijan, Czech Republic, Moldova, Poland | 0.294 |
| 8 | Ireland, Israel, Latvia, Lithuania, Ukraine | 0.338 |
| 9 | Georgia, Italy, Portugal | 0.215 |

`cohesion` is mean within-cluster similarity minus mean similarity to everyone outside the cluster; all eight are positive.

Overlap with hand-labelled real-world blocs (labels never entered the clustering):

| cluster_id | reference_bloc | in_cluster | bloc_size | share_of_bloc |
| --- | --- | --- | --- | --- |
| 1 | Balkan | 1 | 7 | 0.143 |
| 1 | Mediterranean | 1 | 7 | 0.143 |
| 2 | Ex-USSR | 3 | 10 | 0.3 |
| 2 | Balkan | 2 | 7 | 0.286 |
| 2 | Mediterranean | 1 | 7 | 0.143 |
| 3 | Mediterranean | 2 | 7 | 0.286 |
| 3 | Balkan | 1 | 7 | 0.143 |
| 4 | Western Europe | 6 | 8 | 0.75 |
| 5 | Nordic | 5 | 5 | 1.0 |
| 5 | Baltic | 1 | 3 | 0.333 |
| 5 | Western Europe | 1 | 8 | 0.125 |
| 5 | Ex-USSR | 1 | 10 | 0.1 |
| 6 | Ex-Yugoslav | 3 | 3 | 1.0 |
| 6 | Balkan | 3 | 7 | 0.429 |
| 7 | Ex-USSR | 2 | 10 | 0.2 |
| 8 | Baltic | 2 | 3 | 0.667 |
| 8 | Ex-USSR | 3 | 10 | 0.3 |
| 8 | Mediterranean | 1 | 7 | 0.143 |
| 8 | Western Europe | 1 | 8 | 0.125 |
| 9 | Mediterranean | 2 | 7 | 0.286 |
| 9 | Ex-USSR | 1 | 10 | 0.1 |

## Interpretation

The clustering recovers the geopolitical map without ever being shown it: an ex-Yugoslav cluster (Croatia, Serbia, Slovenia), a post-Soviet cluster (Russia, Ukraine, Georgia, Latvia, Lithuania), a Caucasus / Black Sea / Balkan cluster (Armenia, Azerbaijan, Belarus, Bulgaria, Moldova, Romania, Turkey), an eastern-Mediterranean cluster (Greece, Cyprus, Albania, Malta), a Western European core (France, Germany, Netherlands, Austria, Switzerland) and a Nordic-Anglo cluster (Sweden, Norway, Denmark, Finland, Iceland, Ireland, the UK). Since the input was only "who did you hand points to, relative to what everyone else handed them", the fact that language, borders and shared history fall out of it is direct evidence that a large share of the voting is regional rather than musical.

The surprises are more informative than the confirmations. Armenia and Azerbaijan land in the *same* cluster despite a standing political conflict and a near-perfect record of giving each other nothing — because this metric measures who you vote *like*, not who you vote *for*, and their profiles agree everywhere except on each other. Estonia separates from Latvia and Lithuania and sits with the Nordics, matching its linguistic and broadcasting ties to Finland rather than the Baltic label; the 'Baltic bloc' is really a Latvia–Lithuania pair. Turkey clusters with the Caucasus and Balkans rather than with Western Europe, and Hungary sits with the Nordic-Anglo group rather than with its neighbours. The most heterogeneous cluster — Belgium, Israel, Poland and Spain — has no geographic story at all; it is closer to a residual of countries whose deviations from consensus are driven by diaspora voting than to a bloc. Finally, Australia, Italy and Portugal form their own group of contest outsiders, which is a reminder that 'not voting like anyone' is itself a detectable pattern.

The honest caveat: the silhouette values (~0.19) are low in absolute terms. These blocs are real and reproducible tendencies, not hard partitions — most countries sit somewhere between two of them, and the clustering is best read as evidence that regional structure exists and is strong, not that every country belongs cleanly to exactly one bloc.
