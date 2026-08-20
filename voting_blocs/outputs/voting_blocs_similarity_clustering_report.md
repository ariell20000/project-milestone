# Eurovision voting blocs: similarity and hierarchical clustering

Window: 2004–2026 finals. Countries clustered: 43 (voted in at least 4 editions). Excluded: Bosnia and Herzegovina, Luxembourg, North Macedonia, Serbia and Montenegro.

## Method

Each country is represented by its **outgoing** vote row: how many points it gave to every other country. Two normalizations are applied before any distance is computed.

1. **Opportunity normalization.** Raw point totals mostly measure attendance: the UK voted in 17 finals in this window, Montenegro in 2. Every cell is divided by the number of editions in which that voter voted *and* that recipient competed, giving average points per chance to vote. (Row-sum normalization would not have fixed this — cosine is already invariant to row scaling, so the bias lives in the *support* of the row, not its length.)
2. **Recipient centering.** Each recipient's column is centered on the average rate it received from the voters who could vote for it. Without this step, two countries look similar simply because they both rewarded the songs everyone rewarded — that is the shared musical-taste signal, and it swamps the bloc signal. On the residuals, similarity asks whether two countries deviate from the field's consensus in the same direction, which is the quantity the research question is about.

**Metric: cosine similarity** on the centered rates; distance = 1 − similarity, fed to `scipy.cluster.hierarchy.linkage` as a precomputed condensed matrix.

Why cosine, from the background reading: the collaborative-filtering literature splits similarity measures into those that only use *whether* a rating exists (Jaccard and its relatives) and those that use the *magnitude* of the rating (cosine and friends), and reviews of sparse CF datasets report Salton's cosine performing well on larger, denser matrices. Here the magnitude is the whole story: giving a neighbour 12 points every single year and giving them 1 point once are politically very different acts, and Jaccard collapses both to "voted for". Binarizing is also close to uninformative in this data — almost every pair of frequent participants has awarded each other *something* at some point, so the binary rows are nearly all-ones. Pearson correlation is the other CF standard, but centering *rows* would treat a country's structural zeros (songs it never had the chance to vote for) as negative preferences; the recipient centering above achieves the popularity adjustment without that side effect. Bray–Curtis, the ecology analogue for abundance vectors, is documented as sensitive to differences in total abundance and erratic on very sparse rows — exactly this dataset's failure mode. Cosine also matches the way this problem is set up in the Eurovision literature, where vote matrices are reconstructed into a distance matrix and passed to agglomerative clustering.

The choice was checked, not assumed. Running the identical pipeline (average linkage, k=9) on each candidate metric:

| metric | k | silhouette | largest_cluster | singletons |
| --- | --- | --- | --- | --- |
| cosine_raw_totals | 9 | 0.045 | 31 | 5 |
| cosine_centered_rates | 9 | 0.195 | 8 | 0 |
| jaccard_binarized | 9 | 0.068 | 30 | 4 |

Raw-total cosine collapses into one giant cluster plus singletons; Jaccard on the binarized matrix separates almost nothing. Centered cosine is the only one of the three that produces balanced, interpretable groups.

## Linkage and choice of k

**Average linkage** on the precomputed cosine distances. This is deliberately not k-means and deliberately not Ward: neither the cosine space nor the linkage ever sees point coordinates, and centroid-style methods assume a Euclidean geometry the space does not have. The choice between the two linkage methods that do work on precomputed distances was checked, not assumed, at k=9:

| method | k | silhouette | largest_cluster | singletons |
| --- | --- | --- | --- | --- |
| complete | 9 | 0.183 | 8 | 0 |
| average | 9 | 0.195 | 8 | 0 |

Average linkage has the higher silhouette at k=9, so it is what the pipeline uses. (This is a re-check, not a fixed assumption: on an earlier, shorter window complete linkage used to win; which method is better can shift as more years are added, so both are always compared on the current data rather than one being hardcoded.)

Silhouette scores computed on the precomputed distance matrix:

| k | n_clusters | silhouette | largest_cluster | singletons |
| --- | --- | --- | --- | --- |
| 2 | 2 | 0.141 | 23 | 0 |
| 3 | 3 | 0.14 | 20 | 0 |
| 4 | 4 | 0.164 | 20 | 0 |
| 5 | 5 | 0.159 | 14 | 0 |
| 6 | 6 | 0.179 | 13 | 0 |
| 7 | 7 | 0.186 | 13 | 0 |
| 8 | 8 | 0.184 | 13 | 0 |
| 9 | 9 | 0.195 | 8 | 0 |
| 10 | 10 | 0.189 | 8 | 1 |
| 11 | 11 | 0.175 | 8 | 2 |
| 12 | 12 | 0.163 | 8 | 3 |

Silhouette peaks at **k=9** (0.195) and falls off on both sides, so k is not a knife-edge choice sensitive to one extra or one fewer group.

## Clusters

| cluster_id | countries | cohesion |
| --- | --- | --- |
| 1 | Italy, Portugal | 0.410 |
| 2 | Azerbaijan, Czech Republic, Israel, Moldova | 0.277 |
| 3 | Albania, Armenia, Bulgaria, Cyprus, Greece, Malta, San Marino | 0.387 |
| 4 | Belarus, Georgia, Russia, Turkey | 0.434 |
| 5 | Hungary, Romania, Spain | 0.232 |
| 6 | Croatia, Montenegro, Serbia, Slovenia | 0.524 |
| 7 | Austria, Belgium, France, Germany, Netherlands, Switzerland | 0.346 |
| 8 | Australia, Denmark, Estonia, Finland, Iceland, Norway, Sweden, United Kingdom | 0.405 |
| 9 | Ireland, Latvia, Lithuania, Poland, Ukraine | 0.356 |

`cohesion` is mean within-cluster similarity minus mean similarity to everyone outside the cluster; all eight are positive.

Overlap with hand-labelled real-world blocs (labels never entered the clustering):

| cluster_id | reference_bloc | in_cluster | bloc_size | share_of_bloc |
| --- | --- | --- | --- | --- |
| 1 | Mediterranean | 2 | 7 | 0.286 |
| 2 | Ex-USSR | 2 | 10 | 0.2 |
| 2 | Mediterranean | 1 | 7 | 0.143 |
| 3 | Balkan | 3 | 7 | 0.429 |
| 3 | Mediterranean | 3 | 7 | 0.429 |
| 3 | Ex-USSR | 1 | 10 | 0.1 |
| 4 | Ex-USSR | 3 | 10 | 0.3 |
| 5 | Balkan | 1 | 7 | 0.143 |
| 5 | Mediterranean | 1 | 7 | 0.143 |
| 6 | Ex-Yugoslav | 3 | 3 | 1.0 |
| 6 | Balkan | 3 | 7 | 0.429 |
| 7 | Western Europe | 6 | 8 | 0.75 |
| 8 | Nordic | 5 | 5 | 1.0 |
| 8 | Baltic | 1 | 3 | 0.333 |
| 8 | Western Europe | 1 | 8 | 0.125 |
| 8 | Ex-USSR | 1 | 10 | 0.1 |
| 9 | Baltic | 2 | 3 | 0.667 |
| 9 | Ex-USSR | 3 | 10 | 0.3 |
| 9 | Western Europe | 1 | 8 | 0.125 |

## Interpretation

The clustering recovers the geopolitical map without ever being shown it: a perfect ex-Yugoslav cluster (Croatia, Montenegro, Serbia, Slovenia — all four of the reference bloc, cohesion 0.524, its own tightest cluster), a Western European core (Austria, Belgium, France, Germany, Netherlands, Switzerland — 6 of 8 of that reference bloc), and a Nordic-plus cluster (all five Nordic countries, joined by Estonia, Australia and the UK). Since the input was only "who did you hand points to, relative to what everyone else handed them", the fact that language, borders and shared history fall out of it is direct evidence that a large share of the voting is regional rather than musical.

The surprises are more informative than the confirmations. Armenia and Azerbaijan — politically hostile, and giving each other almost no direct points — are *not* merged at this fine resolution: Armenia sits with Albania, Bulgaria, Cyprus, Greece, Malta and San Marino, Azerbaijan with Czech Republic, Israel and Moldova. But their profiles still agree well above the field average (cosine similarity 0.167 — Azerbaijan is Armenia's 7th-closest match out of 42), and coarsening the same tree to 3 families (see the method-comparison piece) puts them back in the same family. Run the identical pipeline on the televote ballots only, and the two *do* land in the same fine cluster (with Bulgaria, Cyprus, Greece and San Marino) — the jury-only and combined views keep them apart. That is consistent with this project's other finding that the public vote encodes geographic/bloc structure more strongly than professional juries do: whether Armenia and Azerbaijan 'cluster together' depends on which ballot and which resolution you're asking about, not on a single fixed answer.

Latvia and Lithuania — the actual Baltic pair — land with Ireland, Poland and Ukraine rather than with Estonia, which instead joins the Nordics, matching its closer linguistic and broadcasting ties to Finland. Belarus, Georgia, Russia and Turkey form their own Caucasus/Black-Sea cluster (cohesion 0.434, the second-tightest), distinct from the Balkan/Mediterranean cluster Armenia sits in. The two weakest, least cohesive clusters — Hungary/Romania/Spain (cohesion 0.232) and Azerbaijan/Czech Republic/Israel/Moldova (0.277) — have no clean geographic story; they read as residuals of countries whose deviations from consensus don't line up with anyone else's as strongly as the tighter blocs do.

The honest caveat: the silhouette values (~0.18-0.20) are low in absolute terms. These blocs are real and reproducible tendencies, not hard partitions — most countries sit somewhere between two of them, and the clustering is best read as evidence that regional structure exists and is strong, not that every country belongs cleanly to exactly one bloc.
