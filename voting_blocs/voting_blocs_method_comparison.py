"""Compares the two independent country-grouping methods elsewhere in this
project, at three matched windows (full / jury-only / televote-only):

  - voting_blocs_clustering.py: hierarchical clustering of *outgoing vote
    profiles* (opportunity-normalized, recipient-centered cosine similarity
    - who votes LIKE whom). Fine-grained: k=9/11/12 clusters.
  - voting_blocs_graph.py: Louvain communities on the *direct*
    country-to-country points graph (who votes FOR whom). Coarse: 3
    communities in every window.

Different input representation, different algorithm, and - crucially - very
different granularity, so a flat ARI/NMI between the raw labels conflates
"disagree about structure" with "describe the same structure at different
resolutions." Two ways of controlling for that are checked side by side:

1. Nesting/purity at the fine resolution: does each fine (profile) cluster
   sit mostly inside one coarse (graph) community?
2. A matched-resolution comparison: re-cluster the fine profile clusters
   into the *same number* of groups the graph method found (3, in every
   window here) via voting_blocs_clustering.cluster_families() - average
   inter-cluster similarity, not a coarser cut of the same complete-linkage
   tree, which was checked separately and matters (see that function's
   docstring) - and compare that directly to the graph communities. This is
   the fairer comparison when one method's natural k is much larger than
   the other's: 9-vs-3 conflates disagreement with resolution, 3-vs-3 does
   not.

And whichever check flags disagreement, is it the same handful of countries
doing it across windows (a structural finding) or a different set each time
(closer to noise)?

Reads the saved cluster-assignment CSVs for the fine partitions and country
lists, but re-derives the similarity/distance/linkage for the
matched-resolution re-clustering in (2) - that intermediate structure isn't
saved anywhere, only the final k-cluster assignment is.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from voting_blocs_clustering import build_linkage, cluster_families, to_distance
from voting_blocs_similarity import (
    JURY_COL,
    MAX_YEAR,
    MIN_VOTING_EDITIONS,
    MIN_YEAR,
    SPLIT_MIN_YEAR,
    TELEVOTE_COL,
    build_similarity,
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

SIMILARITY_CLUSTER_PATHS = {
    "full": OUTPUT_DIR / "voting_blocs_clusters.csv",
    "jury": OUTPUT_DIR / "voting_blocs_clusters_jury.csv",
    "televote": OUTPUT_DIR / "voting_blocs_clusters_televote.csv",
}
GRAPH_COMMUNITIES_PATH = OUTPUT_DIR / "voting_blocs_communities.csv"

SUMMARY_PATH = OUTPUT_DIR / "voting_blocs_method_comparison_summary.csv"
MATCHED_SUMMARY_PATH = OUTPUT_DIR / "voting_blocs_method_comparison_matched_summary.csv"
CROSSTAB_PATH = OUTPUT_DIR / "voting_blocs_method_comparison_crosstabs.csv"
MISFITS_PATH = OUTPUT_DIR / "voting_blocs_method_comparison_misfits.csv"
EXCEPTIONS_PATH = OUTPUT_DIR / "voting_blocs_method_comparison_matched_exceptions.csv"
REPORT_PATH = OUTPUT_DIR / "voting_blocs_method_comparison_report.md"

WINDOWS = ("full", "jury", "televote")
WINDOW_LABELS = {
    "full": "All points",
    "jury": "Jury points only",
    "televote": "Televote points only",
}
# Mirrors exactly how voting_blocs_clustering.py builds each window's
# similarity matrix (main() for "full", cluster_subset() for jury/televote),
# so the re-derived distance/linkage here matches the saved fine clusters.
WINDOW_PARAMS = {
    "full": dict(min_year=MIN_YEAR, max_year=MAX_YEAR, points_types=None, min_editions=MIN_VOTING_EDITIONS),
    "jury": dict(
        min_year=SPLIT_MIN_YEAR, max_year=MAX_YEAR, points_types={JURY_COL}, min_editions=MIN_VOTING_EDITIONS
    ),
    "televote": dict(
        min_year=SPLIT_MIN_YEAR, max_year=MAX_YEAR, points_types={TELEVOTE_COL}, min_editions=MIN_VOTING_EDITIONS
    ),
}


def load_similarity_clusters() -> dict[str, pd.Series]:
    return {
        window: pd.read_csv(path).set_index("country")["cluster_id"]
        for window, path in SIMILARITY_CLUSTER_PATHS.items()
    }


def load_graph_communities() -> dict[str, pd.Series]:
    # voting_blocs_graph.py keys the Netherlands as "The Netherlands" (from its
    # own NODE_ALIASES); voting_blocs_similarity.py normalizes it to
    # "Netherlands" - the same mismatch already handled in
    # voting_blocs_clustering.py's MAP_COORD_ALIASES, fixed here so the two
    # methods' country lists actually line up instead of each showing it as
    # "missing" from the other.
    df = pd.read_csv(GRAPH_COMMUNITIES_PATH)
    df["country"] = df["country"].replace({"The Netherlands": "Netherlands"})
    return {
        window: df[df["graph"] == window].set_index("country")["community_id"]
        for window in WINDOWS
    }


def nesting_purity(fine: pd.Series, coarse: pd.Series) -> tuple[float, pd.DataFrame, list[str]]:
    """For each fine (profile-similarity) cluster, its majority coarse
    (graph) community and what fraction of its members agree with that
    majority. A country is a "misfit" if its own coarse label isn't its
    fine cluster's majority - i.e. it votes like the rest of its profile
    cluster, but exchanges points more like a different region."""
    rows = []
    misfits: list[str] = []
    agree = 0
    for fine_id, members in fine.groupby(fine).groups.items():
        members = list(members)
        majority = coarse.loc[members].mode().iloc[0]
        matching = (coarse.loc[members] == majority).sum()
        agree += matching
        rows.append(
            {
                "fine_cluster": fine_id,
                "size": len(members),
                "majority_coarse_community": majority,
                "purity": matching / len(members),
                "members": ", ".join(sorted(members)),
            }
        )
        misfits.extend(m for m in members if coarse.loc[m] != majority)
    purity_df = pd.DataFrame(rows).sort_values("fine_cluster")
    overall_purity = agree / len(fine)
    return overall_purity, purity_df, sorted(misfits)


def build_family_series(window: str, k: int, n_families: int) -> pd.Series:
    """Re-derives this window's similarity/distance/linkage (not saved
    anywhere - only the final k-cluster assignment is) and re-clusters the k
    fine profile clusters into n_families coarse groups via
    cluster_families(), returning a country -> family_id Series at the
    *same* resolution as the graph communities, for a matched comparison.

    Uses raw scipy fcluster ids internally (not assign_clusters()'s
    dendrogram-order renumbering that the saved CSVs use), so this does not
    depend on - and does not need to match - the cluster_id values in the
    saved fine-cluster CSVs; only their count (k) is reused."""
    similarity = build_similarity(**WINDOW_PARAMS[window])
    distance = to_distance(similarity)
    linkage_matrix = build_linkage(distance)
    families = cluster_families(distance, linkage_matrix, k, n_families)
    family_of_fine = {fine_id: family_idx + 1 for family_idx, ids in enumerate(families) for fine_id in ids}
    fine_raw = pd.Series(fcluster(linkage_matrix, k, criterion="maxclust"), index=distance.index)
    return fine_raw.map(family_of_fine)


def compare_window(window: str, sim: pd.Series, graph: pd.Series) -> dict:
    shared = sorted(set(sim.index) & set(graph.index))
    only_sim = sorted(set(sim.index) - set(graph.index))
    only_graph = sorted(set(graph.index) - set(sim.index))

    sim_shared, graph_shared = sim.loc[shared], graph.loc[shared]
    ari = adjusted_rand_score(sim_shared, graph_shared)
    nmi = normalized_mutual_info_score(sim_shared, graph_shared)
    overall_purity, purity_df, misfits = nesting_purity(sim_shared, graph_shared)

    crosstab = pd.crosstab(sim_shared, graph_shared)
    crosstab.index.name, crosstab.columns.name = "profile_cluster", "graph_community"

    n_families = graph_shared.nunique()
    family = build_family_series(window, sim.nunique(), n_families)
    family_shared_idx = sorted(set(family.index) & set(graph.index))
    fam, fam_graph = family.loc[family_shared_idx], graph.loc[family_shared_idx]
    family_ari = adjusted_rand_score(fam, fam_graph)
    family_nmi = normalized_mutual_info_score(fam, fam_graph)
    family_purity, _family_purity_df, family_exceptions = nesting_purity(fam, fam_graph)
    family_crosstab = pd.crosstab(fam, fam_graph)
    family_crosstab.index.name, family_crosstab.columns.name = "profile_family", "graph_community"

    return {
        "window": window,
        "n_shared": len(shared),
        "only_similarity": only_sim,
        "only_graph": only_graph,
        "n_fine_clusters": sim_shared.nunique(),
        "n_coarse_communities": graph_shared.nunique(),
        "ari": ari,
        "nmi": nmi,
        "purity": overall_purity,
        "purity_by_cluster": purity_df,
        "crosstab": crosstab,
        "misfits": misfits,
        "n_families": n_families,
        "n_family_shared": len(family_shared_idx),
        "family_ari": family_ari,
        "family_nmi": family_nmi,
        "family_purity": family_purity,
        "family_crosstab": family_crosstab,
        "family_exceptions": family_exceptions,
    }


def markdown_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    divider = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = [
        "| " + " | ".join(str(v) for v in record) + " |"
        for record in df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def crosstab_markdown(crosstab: pd.DataFrame, row_label: str = "profile") -> str:
    header = f"| {row_label} \\ graph | " + " | ".join(str(c) for c in crosstab.columns) + " |"
    divider = "| --- | " + " | ".join("---" for _ in crosstab.columns) + " |"
    rows = [
        f"| {index} | " + " | ".join(str(v) for v in row) + " |"
        for index, row in crosstab.iterrows()
    ]
    return "\n".join([header, divider, *rows])


def build_report(
    results: dict[str, dict], misfit_counts: pd.Series, exception_counts: pd.Series
) -> str:
    summary_rows = pd.DataFrame(
        [
            {
                "window": WINDOW_LABELS[r["window"]],
                "n_countries": r["n_shared"],
                "fine_clusters": r["n_fine_clusters"],
                "coarse_communities": r["n_coarse_communities"],
                "ARI": round(r["ari"], 3),
                "NMI": round(r["nmi"], 3),
                "nesting_purity": round(r["purity"], 3),
            }
            for r in results.values()
        ]
    )
    matched_rows = pd.DataFrame(
        [
            {
                "window": WINDOW_LABELS[r["window"]],
                "n_countries": r["n_family_shared"],
                "families": r["n_families"],
                "coarse_communities": r["n_coarse_communities"],
                "ARI": round(r["family_ari"], 3),
                "NMI": round(r["family_nmi"], 3),
                "purity": round(r["family_purity"], 3),
            }
            for r in results.values()
        ]
    )

    repeat_offenders = misfit_counts[misfit_counts >= 2]
    repeat_exceptions = exception_counts[exception_counts >= 2]

    ari_improved = all(r["family_ari"] >= r["ari"] for r in results.values())
    nmi_improved = all(r["family_nmi"] >= r["nmi"] for r in results.values())
    if ari_improved and nmi_improved:
        matched_resolution_verdict = (
            "and they agree - ARI and NMI at matched resolution are both at or above the raw "
            "fine-vs-coarse numbers in every window, so the apparent weak agreement in the headline "
            "table was mostly the resolution gap, not real disagreement."
        )
    elif ari_improved and not nmi_improved:
        matched_resolution_verdict = (
            "and the picture is mixed: ARI improves at matched resolution in every window (fewer "
            "accidental pairwise disagreements once both sides use the same number of groups), but "
            "NMI is actually *lower* everywhere - collapsing 9-12 profile clusters down to 3 throws "
            "away fine structure that genuinely did align with the graph communities, which a pairwise "
            "measure like ARI doesn't penalize but an information-theoretic one like NMI does. Read "
            "together: the two methods aren't purely disagreeing, but the fine profile clusters carry "
            "real information about graph community structure that a matched, coarser comparison loses."
        )
    elif nmi_improved and not ari_improved:
        matched_resolution_verdict = (
            "and NMI improves at matched resolution in every window but ARI does not - the opposite "
            "asymmetry, worth a closer look at the per-window crosstabs below rather than trusting "
            "either single number."
        )
    else:
        per_window = "; ".join(
            f"{WINDOW_LABELS[r['window']]} {r['ari']:.3f} → {r['family_ari']:.3f}"
            for r in results.values()
        )
        matched_resolution_verdict = (
            "and the picture is mixed *across windows*, not just across metrics: matched-resolution "
            f"ARI moves in different directions depending on the window ({per_window}). Where it rises "
            "sharply, the raw headline number was mostly a resolution-gap artifact; where it falls or "
            "holds flat, the disagreement between the two methods is real rather than a granularity "
            "effect. Read per-window below rather than as one verdict."
        )

    lines: list[str] = [
        "# Do the two voting-bloc methods agree? Profile-similarity clusters vs. graph communities",
        "",
        "Two independent partitions of Eurovision countries exist in this project:",
        "",
        "1. **Profile-similarity clustering** (`voting_blocs_clustering.py`): hierarchical clustering "
        "on cosine similarity of *outgoing vote profiles* - opportunity-normalized, recipient-centered "
        "so it measures who votes *like* whom, independent of whether they ever voted for each other. "
        "Fine-grained (9-12 clusters depending on window).",
        "2. **Graph communities** (`voting_blocs_graph.py`): Louvain community detection on the "
        "*direct* country-to-country points graph - who votes *for* whom. Coarse (3 communities in "
        "every window, by construction of that method's modularity optimum).",
        "",
        "Both are run on the same three windows (all points / jury-only / televote-only) and reused "
        "here exactly as saved by those scripts - nothing is re-clustered here.",
        "",
        "## Why a flat agreement score isn't the whole story",
        "",
        "The two methods don't just disagree about *which* countries belong together - they operate "
        "at different resolutions by construction (9-12 groups vs. 3). A low raw agreement score could "
        "mean the methods see genuinely different structure, or it could mean they see the *same* "
        "regional structure but one method zooms in further than the other. This report checks both: "
        "ARI/NMI (do the raw labels agree) and a nesting/purity check (does each fine cluster sit "
        "mostly inside one coarse community, the way you'd expect if the fine method were just a "
        "closer look at the coarse one).",
        "",
        "## Summary across windows",
        "",
        markdown_table(summary_rows),
        "",
        "- **ARI / NMI**: agreement between the raw fine-cluster and coarse-community labels "
        "(0 = no better than chance, 1 = identical partitions).",
        "- **nesting_purity**: of all shared countries, the fraction whose graph community matches "
        "the *majority* graph community of their own profile cluster. 1.0 would mean every profile "
        "cluster sits perfectly inside one graph community (methods agree on structure, differ only "
        "in resolution); values near what 3 random coarse labels would give a 9-12-way partition "
        "purely by chance would suggest little real overlap.",
        "",
        "## The fair comparison: matched resolution",
        "",
        "The table above compares 9-12 profile clusters against 3 graph communities directly, which "
        "conflates disagreement with the resolution gap itself. A fairer test: re-cluster the fine "
        "profile clusters into the *same number* of groups the graph method found for that window (3 "
        "everywhere here), via `voting_blocs_clustering.cluster_families()`. That function specifically "
        "does *not* just cut the existing complete-linkage tree higher - complete linkage measures "
        "cluster-to-cluster distance as the worst pair, which strands a tight cluster alone and globs "
        "loose ones together once continued to a coarse cut (checked on the `full` window: a same-tree "
        "cut of 9 clusters into 3 scored ARI 0.134 with lopsided sizes 19/8/16). Instead it treats each "
        "fine cluster as a unit, measures inter-cluster distance as the *average* over all member pairs, "
        "and re-clusters with average linkage - on `full` that scored ARI 0.311 with sizes 18/12/13.",
        "",
        markdown_table(matched_rows),
        "",
    ]

    for window in WINDOWS:
        r = results[window]
        lines += [
            f"## {WINDOW_LABELS[window]} ({r['n_shared']} countries in both methods)",
            "",
        ]
        if r["only_similarity"] or r["only_graph"]:
            lines.append(
                f"Only in profile clustering (too few graph edges / excluded there): "
                f"{', '.join(r['only_similarity']) or '—'}. "
                f"Only in graph communities (excluded from profile clustering, e.g. too few voting "
                f"editions): {', '.join(r['only_graph']) or '—'}."
            )
            lines.append("")
        lines += [
            f"ARI = {r['ari']:.3f}, NMI = {r['nmi']:.3f}, nesting purity = {r['purity']:.3f}.",
            "",
            "Crosstab (rows = profile cluster, columns = graph community; a clean block-diagonal "
            "pattern - each row concentrated in one column - is what nesting looks like):",
            "",
            crosstab_markdown(r["crosstab"]),
            "",
            f"Misfits in this window ({len(r['misfits'])} of {r['n_shared']}): "
            f"{', '.join(r['misfits']) if r['misfits'] else '—'}.",
            "",
            f"**Matched resolution** ({r['n_families']} profile families vs. {r['n_coarse_communities']} "
            f"graph communities): ARI = {r['family_ari']:.3f}, NMI = {r['family_nmi']:.3f}, "
            f"purity = {r['family_purity']:.3f}.",
            "",
            crosstab_markdown(r["family_crosstab"], row_label="family"),
            "",
            f"Exceptions at matched resolution ({len(r['family_exceptions'])} of {r['n_family_shared']}): "
            f"{', '.join(r['family_exceptions']) if r['family_exceptions'] else '—'}.",
            "",
        ]

    nonzero = misfit_counts[misfit_counts > 0]
    lines += [
        "## Are the same countries misfits across windows?",
        "",
        f"Counting how many of the three windows each country is a misfit in (0-3; the "
        f"{(misfit_counts == 0).sum()} countries that are never a misfit are omitted below):",
        "",
        markdown_table(
            nonzero.rename("windows_as_misfit")
            .reset_index()
            .rename(columns={"index": "country"})
            .sort_values(["windows_as_misfit", "country"], ascending=[False, True])
        ),
        "",
    ]
    if len(repeat_offenders):
        lines += [
            f"**{len(repeat_offenders)} countries are misfits in 2 or more of the 3 windows**: "
            f"{', '.join(sorted(repeat_offenders.index))}. Recurring across independently-fit windows "
            "(full/jury/televote each get their own clustering and their own Louvain run) is what "
            "distinguishes a structural pattern from noise - a country that only breaks nesting once "
            "could be a single window's clustering wobble, but one that does it in 2-3 windows is "
            "consistently voting-like-its-neighbors while exchanging-points-like-a-different-region, "
            "or vice versa.",
            "",
        ]
    else:
        lines += [
            "No country is a misfit in more than one window - whatever nesting failures exist look "
            "window-specific rather than a consistent property of particular countries.",
            "",
        ]

    nonzero_exceptions = exception_counts[exception_counts > 0]
    lines += [
        "## Are the same countries exceptions across windows? (matched resolution)",
        "",
        "Same question as above, but counted on the matched-resolution (family-level) comparison "
        f"instead of the raw fine clusters (0-3; the {(exception_counts == 0).sum()} countries that are "
        "never an exception are omitted below):",
        "",
        markdown_table(
            nonzero_exceptions.rename("windows_as_exception")
            .reset_index()
            .rename(columns={"index": "country"})
            .sort_values(["windows_as_exception", "country"], ascending=[False, True])
        ),
        "",
    ]
    if len(repeat_exceptions):
        lines += [
            f"**{len(repeat_exceptions)} countries are exceptions in 2 or more of the 3 windows, at "
            f"matched resolution**: {', '.join(sorted(repeat_exceptions.index))}.",
            "",
        ]
    else:
        lines += [
            "No country is an exception in more than one window at matched resolution.",
            "",
        ]

    lines += [
        "## Reading the two questions together",
        "",
        "**Are the groupings similar?** " + (
            "Broadly yes at the regional level (high nesting purity, each fine cluster mostly falls "
            "inside one graph community) but the raw ARI/NMI is modest, which is expected given the "
            "3-vs-9to12 granularity mismatch rather than evidence the methods disagree."
            if all(r["purity"] >= 0.7 for r in results.values())
            else "Only partially - nesting purity is well below 1 in at least one window, meaning "
            "profile clusters routinely span more than one graph community, not just resolution."
        )
        + " The matched-resolution numbers (family-level, same k as the graph method) are the fairer "
        "read on this question, since they aren't conflating disagreement with resolution at all: "
        + matched_resolution_verdict,
        "",
        "**If they differ, is it noise or a few specific countries?** "
        + (
            f"The repeat-offender list above ({len(repeat_offenders)} countries misfitting in "
            "2+ windows) points to specific countries, not diffuse noise: a handful of countries "
            "vote *like* one region's profile but *exchange points* more with another, consistently "
            "enough to show up across independently-built windows."
            if len(repeat_offenders)
            else "The misfit set reshuffles between windows with no repeat offenders, which is more "
            "consistent with each window's clustering/Louvain run having its own noise than with a "
            "stable subset of countries driving a real conflict between the two notions of 'bloc'."
        )
        + " At matched resolution, "
        + (
            f"{len(repeat_exceptions)} countries repeat as exceptions across 2+ windows "
            f"({', '.join(sorted(repeat_exceptions.index))}) - the same conclusion holds even when "
            "resolution is no longer a confound."
            if len(repeat_exceptions)
            else "no country repeats as an exception across windows, which weakens the fine-resolution "
            "repeat-offender finding above: some of that may have been an artifact of forcing 9-12 "
            "small clusters to nest inside only 3 coarse ones."
        ),
        "",
    ]

    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    similarity = load_similarity_clusters()
    graph = load_graph_communities()

    results = {window: compare_window(window, similarity[window], graph[window]) for window in WINDOWS}

    summary = pd.DataFrame(
        [
            {
                "window": r["window"],
                "n_shared": r["n_shared"],
                "n_fine_clusters": r["n_fine_clusters"],
                "n_coarse_communities": r["n_coarse_communities"],
                "ari": r["ari"],
                "nmi": r["nmi"],
                "nesting_purity": r["purity"],
            }
            for r in results.values()
        ]
    )
    summary.to_csv(SUMMARY_PATH, index=False)

    matched_summary = pd.DataFrame(
        [
            {
                "window": r["window"],
                "n_shared": r["n_family_shared"],
                "n_families": r["n_families"],
                "n_coarse_communities": r["n_coarse_communities"],
                "ari": r["family_ari"],
                "nmi": r["family_nmi"],
                "purity": r["family_purity"],
            }
            for r in results.values()
        ]
    )
    matched_summary.to_csv(MATCHED_SUMMARY_PATH, index=False)

    crosstab_rows = []
    for window, r in results.items():
        ct = r["crosstab"].reset_index().melt(
            id_vars="profile_cluster", var_name="graph_community", value_name="n_countries"
        )
        ct.insert(0, "window", window)
        crosstab_rows.append(ct[ct["n_countries"] > 0])
    pd.concat(crosstab_rows, ignore_index=True).to_csv(CROSSTAB_PATH, index=False)

    all_shared: set[str] = set()
    for window in WINDOWS:
        all_shared |= set(similarity[window].index) & set(graph[window].index)
    misfit_counts = pd.Series(0, index=sorted(all_shared))
    for r in results.values():
        misfit_counts.loc[r["misfits"]] += 1
    misfit_counts.to_frame("windows_as_misfit").to_csv(MISFITS_PATH, index_label="country")

    all_family_shared: set[str] = set()
    for r in results.values():
        all_family_shared |= set(similarity[r["window"]].index) & set(graph[r["window"]].index)
    exception_counts = pd.Series(0, index=sorted(all_family_shared))
    for r in results.values():
        exception_counts.loc[r["family_exceptions"]] += 1
    exception_counts.to_frame("windows_as_exception").to_csv(EXCEPTIONS_PATH, index_label="country")

    report = build_report(results, misfit_counts, exception_counts)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print("Window summary (raw fine-vs-coarse):")
    print(summary.round(3).to_string(index=False))
    print("\nWindow summary (matched resolution):")
    print(matched_summary.round(3).to_string(index=False))
    print(f"\nRepeat-offender misfits (2+ windows): {sorted(misfit_counts[misfit_counts >= 2].index)}")
    print(
        f"Repeat-offender exceptions at matched resolution (2+ windows): "
        f"{sorted(exception_counts[exception_counts >= 2].index)}"
    )
    print("\nSaved:")
    for path in (
        SUMMARY_PATH,
        MATCHED_SUMMARY_PATH,
        CROSSTAB_PATH,
        MISFITS_PATH,
        EXCEPTIONS_PATH,
        REPORT_PATH,
    ):
        print(f"  - {path}")


if __name__ == "__main__":
    main()
