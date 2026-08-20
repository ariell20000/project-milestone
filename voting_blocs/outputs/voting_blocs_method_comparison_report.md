# Do the two voting-bloc methods agree? Profile-similarity clusters vs. graph communities

Two independent partitions of Eurovision countries exist in this project:

1. **Profile-similarity clustering** (`voting_blocs_clustering.py`): hierarchical clustering on cosine similarity of *outgoing vote profiles* - opportunity-normalized, recipient-centered so it measures who votes *like* whom, independent of whether they ever voted for each other. Fine-grained (9-12 clusters depending on window).
2. **Graph communities** (`voting_blocs_graph.py`): Louvain community detection on the *direct* country-to-country points graph - who votes *for* whom. Coarse (3 communities in every window, by construction of that method's modularity optimum).

Both are run on the same three windows (all points / jury-only / televote-only) and reused here exactly as saved by those scripts - nothing is re-clustered here.

## Why a flat agreement score isn't the whole story

The two methods don't just disagree about *which* countries belong together - they operate at different resolutions by construction (9-12 groups vs. 3). A low raw agreement score could mean the methods see genuinely different structure, or it could mean they see the *same* regional structure but one method zooms in further than the other. This report checks both: ARI/NMI (do the raw labels agree) and a nesting/purity check (does each fine cluster sit mostly inside one coarse community, the way you'd expect if the fine method were just a closer look at the coarse one).

## Summary across windows

| window | n_countries | fine_clusters | coarse_communities | ARI | NMI | nesting_purity |
| --- | --- | --- | --- | --- | --- | --- |
| All points | 43 | 9 | 3 | 0.238 | 0.487 | 0.907 |
| Jury points only | 38 | 7 | 3 | 0.241 | 0.406 | 0.789 |
| Televote points only | 38 | 9 | 3 | 0.192 | 0.399 | 0.816 |

- **ARI / NMI**: agreement between the raw fine-cluster and coarse-community labels (0 = no better than chance, 1 = identical partitions).
- **nesting_purity**: of all shared countries, the fraction whose graph community matches the *majority* graph community of their own profile cluster. 1.0 would mean every profile cluster sits perfectly inside one graph community (methods agree on structure, differ only in resolution); values near what 3 random coarse labels would give a 9-12-way partition purely by chance would suggest little real overlap.

## The fair comparison: matched resolution

The table above compares 9-12 profile clusters against 3 graph communities directly, which conflates disagreement with the resolution gap itself. A fairer test: re-cluster the fine profile clusters into the *same number* of groups the graph method found for that window (3 everywhere here), via `voting_blocs_clustering.cluster_families()`. That function specifically does *not* just cut the existing complete-linkage tree higher - complete linkage measures cluster-to-cluster distance as the worst pair, which strands a tight cluster alone and globs loose ones together once continued to a coarse cut (checked on the `full` window: a same-tree cut of 9 clusters into 3 scored ARI 0.134 with lopsided sizes 19/8/16). Instead it treats each fine cluster as a unit, measures inter-cluster distance as the *average* over all member pairs, and re-clusters with average linkage - on `full` that scored ARI 0.311 with sizes 18/12/13.

| window | n_countries | families | coarse_communities | ARI | NMI | purity |
| --- | --- | --- | --- | --- | --- | --- |
| All points | 43 | 3 | 3 | 0.704 | 0.669 | 0.907 |
| Jury points only | 38 | 3 | 3 | 0.165 | 0.245 | 0.579 |
| Televote points only | 38 | 3 | 3 | 0.188 | 0.22 | 0.658 |

## All points (43 countries in both methods)

Only in profile clustering (too few graph edges / excluded there): —. Only in graph communities (excluded from profile clustering, e.g. too few voting editions): Bosnia and Herzegovina, Luxembourg, North Macedonia, Serbia and Montenegro.

ARI = 0.238, NMI = 0.487, nesting purity = 0.907.

Crosstab (rows = profile cluster, columns = graph community; a clean block-diagonal pattern - each row concentrated in one column - is what nesting looks like):

| profile \ graph | 0 | 1 | 2 |
| --- | --- | --- | --- |
| 1 | 2 | 0 | 0 |
| 2 | 3 | 1 | 0 |
| 3 | 7 | 0 | 0 |
| 4 | 4 | 0 | 0 |
| 5 | 2 | 0 | 1 |
| 6 | 0 | 0 | 4 |
| 7 | 1 | 5 | 0 |
| 8 | 0 | 8 | 0 |
| 9 | 1 | 4 | 0 |

Misfits in this window (4 of 43): Czech Republic, France, Hungary, Ukraine.

**Matched resolution** (3 profile families vs. 3 graph communities): ARI = 0.704, NMI = 0.669, purity = 0.907.

| family \ graph | 0 | 1 | 2 |
| --- | --- | --- | --- |
| 1 | 18 | 1 | 1 |
| 2 | 0 | 0 | 4 |
| 3 | 2 | 17 | 0 |

Exceptions at matched resolution (4 of 43): Czech Republic, France, Hungary, Ukraine.

## Jury points only (38 countries in both methods)

Only in profile clustering (too few graph edges / excluded there): —. Only in graph communities (excluded from profile clustering, e.g. too few voting editions): Belarus, Hungary, Luxembourg, Montenegro, North Macedonia, Russia.

ARI = 0.241, NMI = 0.406, nesting purity = 0.789.

Crosstab (rows = profile cluster, columns = graph community; a clean block-diagonal pattern - each row concentrated in one column - is what nesting looks like):

| profile \ graph | 0 | 1 | 2 |
| --- | --- | --- | --- |
| 1 | 1 | 4 | 1 |
| 2 | 1 | 1 | 5 |
| 3 | 6 | 0 | 0 |
| 4 | 2 | 0 | 2 |
| 5 | 1 | 7 | 0 |
| 6 | 0 | 0 | 4 |
| 7 | 0 | 1 | 2 |

Misfits in this window (8 of 38): Armenia, Croatia, Czech Republic, France, Italy, Poland, Romania, Spain.

**Matched resolution** (3 profile families vs. 3 graph communities): ARI = 0.165, NMI = 0.245, purity = 0.579.

| family \ graph | 0 | 1 | 2 |
| --- | --- | --- | --- |
| 1 | 2 | 5 | 6 |
| 2 | 8 | 0 | 2 |
| 3 | 1 | 8 | 6 |

Exceptions at matched resolution (16 of 38): Armenia, Australia, Austria, France, Germany, Italy, Latvia, Lithuania, Netherlands, Poland, Portugal, Romania, Serbia, Spain, Ukraine, United Kingdom.

## Televote points only (38 countries in both methods)

Only in profile clustering (too few graph edges / excluded there): —. Only in graph communities (excluded from profile clustering, e.g. too few voting editions): Belarus, Hungary, Luxembourg, Montenegro, North Macedonia, Russia.

ARI = 0.192, NMI = 0.399, nesting purity = 0.816.

Crosstab (rows = profile cluster, columns = graph community; a clean block-diagonal pattern - each row concentrated in one column - is what nesting looks like):

| profile \ graph | 0 | 1 | 2 |
| --- | --- | --- | --- |
| 1 | 0 | 2 | 1 |
| 2 | 1 | 2 | 0 |
| 3 | 0 | 6 | 0 |
| 4 | 1 | 0 | 2 |
| 5 | 2 | 1 | 1 |
| 6 | 6 | 0 | 0 |
| 7 | 0 | 5 | 0 |
| 8 | 1 | 0 | 1 |
| 9 | 5 | 1 | 0 |

Misfits in this window (7 of 38): Austria, Czech Republic, Germany, Malta, Romania, Slovenia, Spain.

**Matched resolution** (3 profile families vs. 3 graph communities): ARI = 0.188, NMI = 0.220, purity = 0.658.

| family \ graph | 0 | 1 | 2 |
| --- | --- | --- | --- |
| 1 | 1 | 10 | 1 |
| 2 | 9 | 1 | 3 |
| 3 | 6 | 6 | 1 |

Exceptions at matched resolution (13 of 38): Austria, Croatia, Czech Republic, Estonia, Germany, Latvia, Lithuania, Malta, Poland, Serbia, Slovenia, Spain, Ukraine.

## Are the same countries misfits across windows?

Counting how many of the three windows each country is a misfit in (0-3; the 29 countries that are never a misfit are omitted below):

| country | windows_as_misfit |
| --- | --- |
| Czech Republic | 3 |
| France | 2 |
| Romania | 2 |
| Spain | 2 |
| Armenia | 1 |
| Austria | 1 |
| Croatia | 1 |
| Germany | 1 |
| Hungary | 1 |
| Italy | 1 |
| Malta | 1 |
| Poland | 1 |
| Slovenia | 1 |
| Ukraine | 1 |

**4 countries are misfits in 2 or more of the 3 windows**: Czech Republic, France, Romania, Spain. Recurring across independently-fit windows (full/jury/televote each get their own clustering and their own Louvain run) is what distinguishes a structural pattern from noise - a country that only breaks nesting once could be a single window's clustering wobble, but one that does it in 2-3 windows is consistently voting-like-its-neighbors while exchanging-points-like-a-different-region, or vice versa.

## Are the same countries exceptions across windows? (matched resolution)

Same question as above, but counted on the matched-resolution (family-level) comparison instead of the raw fine clusters (0-3; the 21 countries that are never an exception are omitted below):

| country | windows_as_exception |
| --- | --- |
| Ukraine | 3 |
| Austria | 2 |
| Czech Republic | 2 |
| France | 2 |
| Germany | 2 |
| Latvia | 2 |
| Lithuania | 2 |
| Poland | 2 |
| Serbia | 2 |
| Spain | 2 |
| Armenia | 1 |
| Australia | 1 |
| Croatia | 1 |
| Estonia | 1 |
| Hungary | 1 |
| Italy | 1 |
| Malta | 1 |
| Netherlands | 1 |
| Portugal | 1 |
| Romania | 1 |
| Slovenia | 1 |
| United Kingdom | 1 |

**10 countries are exceptions in 2 or more of the 3 windows, at matched resolution**: Austria, Czech Republic, France, Germany, Latvia, Lithuania, Poland, Serbia, Spain, Ukraine.

## Reading the two questions together

**Are the groupings similar?** Broadly yes at the regional level (high nesting purity, each fine cluster mostly falls inside one graph community) but the raw ARI/NMI is modest, which is expected given the 3-vs-9to12 granularity mismatch rather than evidence the methods disagree. The matched-resolution numbers (family-level, same k as the graph method) are the fairer read on this question, since they aren't conflating disagreement with resolution at all: and the picture is mixed *across windows*, not just across metrics: matched-resolution ARI moves in different directions depending on the window (All points 0.238 → 0.704; Jury points only 0.241 → 0.165; Televote points only 0.192 → 0.188). Where it rises sharply, the raw headline number was mostly a resolution-gap artifact; where it falls or holds flat, the disagreement between the two methods is real rather than a granularity effect. Read per-window below rather than as one verdict.

**If they differ, is it noise or a few specific countries?** The repeat-offender list above (4 countries misfitting in 2+ windows) points to specific countries, not diffuse noise: a handful of countries vote *like* one region's profile but *exchange points* more with another, consistently enough to show up across independently-built windows. At matched resolution, 10 countries repeat as exceptions across 2+ windows (Austria, Czech Republic, France, Germany, Latvia, Lithuania, Poland, Serbia, Spain, Ukraine) - the same conclusion holds even when resolution is no longer a confound.
