# Eurovision voting blocs: similarity and hierarchical clustering

Window: 2004–2021 finals. Countries clustered: 40 (voted in at least 4 editions). Excluded: Bosnia and Herzegovina, Czech Republic, Montenegro, North Macedonia, San Marino, Serbia and Montenegro.

## Method

Each country is represented by its **outgoing** vote row: how many points it gave to every other country. Two normalizations are applied before any distance is computed.

1. **Opportunity normalization.** Raw point totals mostly measure attendance: the UK voted in 17 finals in this window, Montenegro in 2. Every cell is divided by the number of editions in which that voter voted *and* that recipient competed, giving average points per chance to vote. (Row-sum normalization would not have fixed this — cosine is already invariant to row scaling, so the bias lives in the *support* of the row, not its length.)
2. **Recipient centering.** Each recipient's column is centered on the average rate it received from the voters who could vote for it. Without this step, two countries look similar simply because they both rewarded the songs everyone rewarded — that is the shared musical-taste signal, and it swamps the bloc signal. On the residuals, similarity asks whether two countries deviate from the field's consensus in the same direction, which is the quantity the research question is about.

**Metric: cosine similarity** on the centered rates; distance = 1 − similarity, fed to `scipy.cluster.hierarchy.linkage` as a precomputed condensed matrix.

Why cosine, from the background reading: the collaborative-filtering literature splits similarity measures into those that only use *whether* a rating exists (Jaccard and its relatives) and those that use the *magnitude* of the rating (cosine and friends), and reviews of sparse CF datasets report Salton's cosine performing well on larger, denser matrices. Here the magnitude is the whole story: giving a neighbour 12 points every single year and giving them 1 point once are politically very different acts, and Jaccard collapses both to "voted for". Binarizing is also close to uninformative in this data — almost every pair of frequent participants has awarded each other *something* at some point, so the binary rows are nearly all-ones. Pearson correlation is the other CF standard, but centering *rows* would treat a country's structural zeros (songs it never had the chance to vote for) as negative preferences; the recipient centering above achieves the popularity adjustment without that side effect. Bray–Curtis, the ecology analogue for abundance vectors, is documented as sensitive to differences in total abundance and erratic on very sparse rows — exactly this dataset's failure mode. Cosine also matches the way this problem is set up in the Eurovision literature, where vote matrices are reconstructed into a distance matrix and passed to agglomerative clustering.

The choice was checked, not assumed. Running the identical pipeline (complete linkage, k=8) on each candidate metric:

| metric | k | silhouette | largest_cluster | singletons |
| --- | --- | --- | --- | --- |
| cosine_raw_totals | 8 | 0.14 | 13 | 1 |
| cosine_centered_rates | 8 | 0.188 | 9 | 0 |
| jaccard_binarized | 8 | 0.064 | 15 | 2 |

Raw-total cosine collapses into one giant cluster plus singletons; Jaccard on the binarized matrix separates almost nothing. Centered cosine is the only one of the three that produces balanced, interpretable groups.

## Linkage and choice of k

**Complete linkage** on the precomputed cosine distances. This is deliberately not k-means and deliberately not Ward: neither the cosine space nor the linkage ever sees point coordinates, and centroid-style methods assume a Euclidean geometry the space does not have. Average linkage was tried too and is worse on the same cut (silhouette 0.164 at k=8, and it strands one country in a singleton); complete linkage gives tighter, more balanced blocs (0.188, no singletons).

Silhouette scores computed on the precomputed distance matrix:

| k | n_clusters | silhouette | largest_cluster | singletons |
| --- | --- | --- | --- | --- |
| 2 | 2 | 0.149 | 23 | 0 |
| 3 | 3 | 0.115 | 17 | 0 |
| 4 | 4 | 0.113 | 15 | 0 |
| 5 | 5 | 0.122 | 12 | 0 |
| 6 | 6 | 0.152 | 12 | 0 |
| 7 | 7 | 0.175 | 11 | 0 |
| 8 | 8 | 0.188 | 9 | 0 |
| 9 | 9 | 0.184 | 7 | 0 |
| 10 | 10 | 0.189 | 5 | 0 |
| 11 | 11 | 0.184 | 5 | 1 |
| 12 | 12 | 0.187 | 5 | 1 |

Silhouette rises steadily from k=4 and peaks at **k=8** (0.188) before flattening out — k=10 ties it numerically but only by shattering the blocs into ten groups of three to five, which buys no interpretation. k=8 is the parsimonious choice at the peak.

## Clusters

| cluster_id | countries | cohesion |
| --- | --- | --- |
| 1 | Austria, France, Germany, Netherlands, Switzerland | 0.470 |
| 2 | Australia, Italy, Portugal | 0.289 |
| 3 | Denmark, Estonia, Finland, Hungary, Iceland, Ireland, Norway, Sweden, United Kingdom | 0.321 |
| 4 | Croatia, Serbia, Slovenia | 0.439 |
| 5 | Georgia, Latvia, Lithuania, Russia, Ukraine | 0.554 |
| 6 | Belgium, Israel, Poland, Spain | 0.310 |
| 7 | Albania, Cyprus, Greece, Malta | 0.402 |
| 8 | Armenia, Azerbaijan, Belarus, Bulgaria, Moldova, Romania, Turkey | 0.267 |

`cohesion` is mean within-cluster similarity minus mean similarity to everyone outside the cluster; all eight are positive.

Overlap with hand-labelled real-world blocs (labels never entered the clustering):

| cluster_id | reference_bloc | in_cluster | bloc_size | share_of_bloc |
| --- | --- | --- | --- | --- |
| 1 | Western Europe | 5 | 8 | 0.625 |
| 2 | Mediterranean | 2 | 7 | 0.286 |
| 3 | Nordic | 5 | 5 | 1.0 |
| 3 | Baltic | 1 | 3 | 0.333 |
| 3 | Western Europe | 2 | 8 | 0.25 |
| 3 | Ex-USSR | 1 | 10 | 0.1 |
| 4 | Ex-Yugoslav | 3 | 3 | 1.0 |
| 4 | Balkan | 3 | 7 | 0.429 |
| 5 | Baltic | 2 | 3 | 0.667 |
| 5 | Ex-USSR | 5 | 10 | 0.5 |
| 6 | Mediterranean | 2 | 7 | 0.286 |
| 6 | Western Europe | 1 | 8 | 0.125 |
| 7 | Mediterranean | 3 | 7 | 0.429 |
| 7 | Balkan | 2 | 7 | 0.286 |
| 8 | Ex-USSR | 4 | 10 | 0.4 |
| 8 | Balkan | 2 | 7 | 0.286 |

## Interpretation

The clustering recovers the geopolitical map without ever being shown it: an ex-Yugoslav cluster (Croatia, Serbia, Slovenia), a post-Soviet cluster (Russia, Ukraine, Georgia, Latvia, Lithuania), a Caucasus / Black Sea / Balkan cluster (Armenia, Azerbaijan, Belarus, Bulgaria, Moldova, Romania, Turkey), an eastern-Mediterranean cluster (Greece, Cyprus, Albania, Malta), a Western European core (France, Germany, Netherlands, Austria, Switzerland) and a Nordic-Anglo cluster (Sweden, Norway, Denmark, Finland, Iceland, Ireland, the UK). Since the input was only "who did you hand points to, relative to what everyone else handed them", the fact that language, borders and shared history fall out of it is direct evidence that a large share of the voting is regional rather than musical.

The surprises are more informative than the confirmations. Armenia and Azerbaijan land in the *same* cluster despite a standing political conflict and a near-perfect record of giving each other nothing — because this metric measures who you vote *like*, not who you vote *for*, and their profiles agree everywhere except on each other. Estonia separates from Latvia and Lithuania and sits with the Nordics, matching its linguistic and broadcasting ties to Finland rather than the Baltic label; the 'Baltic bloc' is really a Latvia–Lithuania pair. Turkey clusters with the Caucasus and Balkans rather than with Western Europe, and Hungary sits with the Nordic-Anglo group rather than with its neighbours. The most heterogeneous cluster — Belgium, Israel, Poland and Spain — has no geographic story at all; it is closer to a residual of countries whose deviations from consensus are driven by diaspora voting than to a bloc. Finally, Australia, Italy and Portugal form their own group of contest outsiders, which is a reminder that 'not voting like anyone' is itself a detectable pattern.

The honest caveat: the silhouette values (~0.19) are low in absolute terms. These blocs are real and reproducible tendencies, not hard partitions — most countries sit somewhere between two of them, and the clustering is best read as evidence that regional structure exists and is strong, not that every country belongs cleanly to exactly one bloc.
