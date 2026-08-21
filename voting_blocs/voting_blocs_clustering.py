from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import (
    dendrogram,
    fcluster,
    leaves_list,
    linkage,
    set_link_color_palette,
)
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

from voting_blocs_graph import COUNTRY_COORDS, LAYOUT_OVERRIDES, SHORT_NAMES
from voting_blocs_similarity import (
    HEATMAP_PATH,
    JURY_COL,
    MAX_YEAR,
    MIN_VOTING_EDITIONS,
    MIN_YEAR,
    OUTPUT_DIR,
    SIMILARITY_PATH,
    SPLIT_MIN_YEAR,
    TELEVOTE_COL,
    all_countries,
    build_profiles,
    build_similarity,
    cosine_similarity_matrix,
    eligible_voters,
    jaccard_similarity_matrix,
    load_votes,
    plot_similarity_heatmap,
)

CLUSTERS_PATH = OUTPUT_DIR / "voting_blocs_clusters.csv"
CLUSTERS_JURY_PATH = OUTPUT_DIR / "voting_blocs_clusters_jury.csv"
CLUSTERS_TELEVOTE_PATH = OUTPUT_DIR / "voting_blocs_clusters_televote.csv"
DENDROGRAM_PATH = OUTPUT_DIR / "voting_blocs_dendrogram.png"
SILHOUETTE_PATH = OUTPUT_DIR / "voting_blocs_silhouette_by_k.csv"
SILHOUETTE_JURY_PATH = OUTPUT_DIR / "voting_blocs_silhouette_by_k_jury.csv"
SILHOUETTE_TELEVOTE_PATH = OUTPUT_DIR / "voting_blocs_silhouette_by_k_televote.csv"
METRIC_COMPARISON_PATH = OUTPUT_DIR / "voting_blocs_metric_comparison.csv"
CLUSTER_MAP_PATH = OUTPUT_DIR / "voting_blocs_clusters_map.png"
REPORT_PATH = OUTPUT_DIR / "voting_blocs_similarity_clustering_report.md"

# voting_blocs_similarity.py's harmonize_country normalizes "The Netherlands" /
# "The-Netherlands" down to "Netherlands", but voting_blocs_graph.py's
# COUNTRY_COORDS (built independently, from the network-analysis side) keys it
# as "The Netherlands" - this is the only name that differs between the two
# modules' harmonization, so it is aliased here rather than reconciling both
# modules' naming schemes.
MAP_COORD_ALIASES = {"Netherlands": "The Netherlands"}

LINKAGE_METHOD = "average"
LINK_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
]
CANDIDATE_K = range(2, 13)
# Silhouette peak on the 2004-2026 window, under LINKAGE_METHOD: k=9 (0.195
# for average linkage, 0.183 for complete - see compare_linkage_methods() and
# outputs/voting_blocs_silhouette_by_k.csv for the current curve).
CHOSEN_K = 9

# How many broad colour families the map splits the k clusters into. Fine
# clusters grouped into the same family (see cluster_families) get shades of
# one hue instead of unrelated colours, so visual similarity on the map
# tracks how close the clusters actually are, not an arbitrary palette cycle.
N_COLOR_FAMILIES = 4
FAMILY_BASE_COLORS = LINK_COLORS[:N_COLOR_FAMILIES]

# Named purely as a sanity check against real-world geography/history - these
# labels are never fed into the clustering, they only score how much of each
# discovered cluster a human would have predicted.
REFERENCE_BLOCS = {
    "Nordic": ["Denmark", "Finland", "Iceland", "Norway", "Sweden"],
    "Baltic": ["Estonia", "Latvia", "Lithuania"],
    "Ex-USSR": [
        "Armenia",
        "Azerbaijan",
        "Belarus",
        "Estonia",
        "Georgia",
        "Latvia",
        "Lithuania",
        "Moldova",
        "Russia",
        "Ukraine",
    ],
    "Ex-Yugoslav": ["Croatia", "Serbia", "Slovenia"],
    "Balkan": [
        "Albania",
        "Bulgaria",
        "Croatia",
        "Greece",
        "Romania",
        "Serbia",
        "Slovenia",
    ],
    "Western Europe": [
        "Austria",
        "Belgium",
        "France",
        "Germany",
        "Ireland",
        "Netherlands",
        "Switzerland",
        "United Kingdom",
    ],
    "Mediterranean": [
        "Cyprus",
        "Greece",
        "Israel",
        "Italy",
        "Malta",
        "Portugal",
        "Spain",
    ],
}


def to_distance(similarity: pd.DataFrame) -> pd.DataFrame:
    distance = 1.0 - similarity.to_numpy()
    distance = np.clip((distance + distance.T) / 2.0, 0.0, None)
    np.fill_diagonal(distance, 0.0)
    return pd.DataFrame(distance, index=similarity.index, columns=similarity.columns)


def build_linkage(distance: pd.DataFrame, method: str = LINKAGE_METHOD) -> np.ndarray:
    # Condensed precomputed distances: the space is cosine, not Euclidean, so
    # the linkage never sees coordinates and centroid-style methods (ward,
    # centroid, median) would be reading a geometry that does not exist here.
    return linkage(squareform(distance.to_numpy(), checks=False), method=method)


def silhouette_by_k(
    distance: pd.DataFrame, linkage_matrix: np.ndarray, ks: range = CANDIDATE_K
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for k in ks:
        labels = fcluster(linkage_matrix, k, criterion="maxclust")
        if len(set(labels)) < 2:
            continue
        rows.append(
            {
                "k": int(k),
                "n_clusters": int(len(set(labels))),
                "silhouette": float(
                    silhouette_score(distance.to_numpy(), labels, metric="precomputed")
                ),
                "largest_cluster": int(np.bincount(labels)[1:].max()),
                "singletons": int((np.bincount(labels)[1:] == 1).sum()),
            }
        )
    return pd.DataFrame(rows)


def assign_clusters(
    distance: pd.DataFrame, linkage_matrix: np.ndarray, k: int = CHOSEN_K
) -> pd.DataFrame:
    labels = fcluster(linkage_matrix, k, criterion="maxclust")
    order = leaves_list(linkage_matrix)
    # Renumber so cluster_id follows dendrogram left-to-right order, which keeps
    # the ids stable and readable against the plotted tree.
    seen: dict[int, int] = {}
    for index in order:
        seen.setdefault(labels[index], len(seen) + 1)
    return pd.DataFrame(
        {
            "country": distance.index,
            "cluster_id": [seen[label] for label in labels],
        }
    ).sort_values(["cluster_id", "country"], ignore_index=True)


def compare_metrics(k: int = CHOSEN_K) -> pd.DataFrame:
    """Silhouette of the same clustering pipeline under the candidate metrics,
    so the choice of similarity is an evidenced one rather than an assumption."""
    rates, centered = build_profiles()
    candidates = {
        "cosine_raw_totals": cosine_similarity_matrix(rates),
        "cosine_centered_rates": cosine_similarity_matrix(centered),
        "jaccard_binarized": jaccard_similarity_matrix(rates),
    }
    rows: list[dict[str, float | str | int]] = []
    for name, similarity in candidates.items():
        distance = to_distance(similarity)
        linkage_matrix = build_linkage(distance)
        labels = fcluster(linkage_matrix, k, criterion="maxclust")
        rows.append(
            {
                "metric": name,
                "k": int(k),
                "silhouette": float(
                    silhouette_score(distance.to_numpy(), labels, metric="precomputed")
                ),
                "largest_cluster": int(np.bincount(labels)[1:].max()),
                "singletons": int((np.bincount(labels)[1:] == 1).sum()),
            }
        )
    return pd.DataFrame(rows)


def compare_linkage_methods(
    distance: pd.DataFrame, k: int = CHOSEN_K, methods: tuple[str, ...] = ("complete", "average")
) -> pd.DataFrame:
    """Silhouette of the same cosine-distance matrix under candidate linkage
    methods, so LINKAGE_METHOD is an evidenced choice rather than an
    assumption - and stays evidenced if the underlying data changes (unlike
    a hardcoded number in the report text)."""
    rows: list[dict[str, float | str | int]] = []
    for method in methods:
        linkage_matrix = build_linkage(distance, method=method)
        labels = fcluster(linkage_matrix, k, criterion="maxclust")
        rows.append(
            {
                "method": method,
                "k": int(k),
                "silhouette": float(
                    silhouette_score(distance.to_numpy(), labels, metric="precomputed")
                ),
                "largest_cluster": int(np.bincount(labels)[1:].max()),
                "singletons": int((np.bincount(labels)[1:] == 1).sum()),
            }
        )
    return pd.DataFrame(rows)


def plot_dendrogram(
    distance: pd.DataFrame,
    linkage_matrix: np.ndarray,
    output_path: Path,
    k: int = CHOSEN_K,
) -> None:
    # Halfway between the last within-cluster merge and the first cross-cluster
    # merge, so the k coloured subtrees match fcluster's k groups exactly.
    threshold = float(linkage_matrix[-k, 2] + linkage_matrix[-(k - 1), 2]) / 2.0
    # scipy's default link palette contains a grey that is indistinguishable
    # from above_threshold_color, which made one bloc look like it was cut off.
    set_link_color_palette(LINK_COLORS)
    # Figures get scaled down to \textwidth in the paper (from 18in to ~6.5in,
    # a ~0.36x factor), so every font/line size here is picked to still read
    # comfortably after that shrink, not at this file's native resolution.
    plt.figure(figsize=(18, 9.5))
    dendrogram(
        linkage_matrix,
        labels=list(distance.index),
        color_threshold=threshold,
        above_threshold_color="#9c9c9c",
        leaf_rotation=90,
        leaf_font_size=21,
    )
    ax = plt.gca()
    for line in ax.get_lines():
        line.set_linewidth(2.0)
    ax.axhline(threshold, color="#b30000", linestyle="--", linewidth=2.2)
    ax.set_ylabel("Cosine distance (1 − similarity), complete linkage", fontsize=24)
    ax.set_xlabel("Country", fontsize=24, labelpad=12)
    ax.set_title(
        f"Eurovision voting blocs: hierarchical clustering of outgoing vote "
        f"profiles ({MIN_YEAR}–{MAX_YEAR}, cut at k={k})",
        fontsize=28,
        pad=16,
    )
    ax.tick_params(axis="y", labelsize=22)
    ax.tick_params(axis="x", pad=6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def best_k_by_silhouette(silhouettes: pd.DataFrame) -> int:
    """Unlike CHOSEN_K (picked once, by inspecting the full-window curve),
    the jury/televote subsets get their k picked automatically from their own
    silhouette curve - there's no manual inspection backing a fixed choice
    for those smaller, differently-distributed samples."""
    return int(silhouettes.loc[silhouettes["silhouette"].idxmax(), "k"])


def country_coord(country: str) -> tuple[float, float] | None:
    key = MAP_COORD_ALIASES.get(country, country)
    if key in LAYOUT_OVERRIDES:
        return LAYOUT_OVERRIDES[key]
    return COUNTRY_COORDS.get(key)


def family_shades(base_hex: str, n: int) -> list[str]:
    """n shades of one hue, so clusters sharing a colour family read as
    visibly related. Both the value and saturation bands are deliberately
    narrow (not the full 0-1 range) so even a 4-member family stays close
    enough to read as "the same crayon" at a glance - telling the family
    apart from other families matters more than telling its own members
    apart from each other."""
    h, s, _v = mcolors.rgb_to_hsv(mcolors.to_rgb(base_hex))
    if n == 1:
        return [mcolors.to_hex(mcolors.hsv_to_rgb((h, s, 0.75)))]
    values = np.linspace(0.68, 0.80, n)
    saturations = np.linspace(min(s, 0.85), max(s * 0.55, 0.45), n)
    return [mcolors.to_hex(mcolors.hsv_to_rgb((h, sv, v))) for v, sv in zip(values, saturations)]


def cluster_families(
    distance: pd.DataFrame,
    linkage_matrix: np.ndarray,
    k: int,
    n_families: int = N_COLOR_FAMILIES,
) -> list[list[int]]:
    """Return all fine clusters as a single family so the legend has no blank lines."""
    fine = pd.Series(fcluster(linkage_matrix, k, criterion="maxclust"), index=distance.index)
    leaf_order = distance.index[leaves_list(linkage_matrix)]
    fine_order = list(dict.fromkeys(fine.loc[leaf_order]))
    return [fine_order]


def cluster_id_colors(families: list[list[int]]) -> dict[int, str]:
    """One distinct colour per fine cluster id for maximum contrast."""
    color_by_fine: dict[int, str] = {}
    distinct = [
        "#17becf", "#e377c2", "#ff7f0e", "#9467bd", "#d62728",
        "#8c564b", "#2ca02c", "#1f77b4", "#bcbd22", "#d73027"
    ]
    idx = 0
    for family_ids in families:
        for fine_id in family_ids:
            color_by_fine[fine_id] = distinct[idx % len(distinct)]
            idx += 1
    return color_by_fine


def assign_country_colors(
    distance: pd.DataFrame,
    linkage_matrix: np.ndarray,
    k: int,
    id_colors: dict[int, str],
) -> dict[str, str]:
    fine = pd.Series(fcluster(linkage_matrix, k, criterion="maxclust"), index=distance.index)
    return {country: id_colors[fine[country]] for country in distance.index}


def plot_clusters_on_map(
    ax: plt.Axes,
    clusters: pd.DataFrame,
    colors_by_country: dict[str, str],
    title: str,
    subtitle: str,
    font_scale: float = 1.0,
) -> list[str]:
    """Each country plotted at its real-world coordinates, coloured by
    dendrogram-family shade (see assign_country_colors) - a geographic view
    of a partition that clustering.py itself builds from voting *profiles*,
    not geography, so the map is a check on the clustering rather than an
    input to it."""
    positions = {country: coord for country in clusters["country"] if (coord := country_coord(country)) is not None}
    plotted = clusters[clusters["country"].isin(positions)]
    skipped = sorted(set(clusters["country"]) - set(positions))

    colors = [colors_by_country[country] for country in plotted["country"]]
    ax.scatter(
        [positions[c][0] for c in plotted["country"]],
        [positions[c][1] for c in plotted["country"]],
        s=140 * (font_scale ** 1.5),
        c=colors,
        edgecolors="white",
        linewidths=0.7 * font_scale,
        zorder=2,
    )
    for country in plotted["country"]:
        ax.annotate(
            SHORT_NAMES.get(country, country),
            positions[country],
            xytext=(0, 7 * font_scale),
            textcoords="offset points",
            ha="center",
            fontsize=5.5 * font_scale,
            zorder=3,
        )
    ax.set_title(title, fontsize=13 * font_scale, pad=10 * font_scale)
    ax.text(
        0.5, -0.04, subtitle, transform=ax.transAxes,
        ha="center", va="top", fontsize=9 * font_scale, color="#444444",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_aspect(1.4)
    return skipped


def draw_cluster_legend(
    ax: plt.Axes,
    clusters: pd.DataFrame,
    families: list[list[int]],
    id_colors: dict[int, str],
    wrap_width: int = 44,
    font_scale: float = 1.0,
) -> None:
    """Lists every cluster's members below its map, in the same colour as
    its dots, with a blank line between colour families - so which
    countries share a cluster, and which clusters are close relatives
    (same family = shares a nearby common ancestor in the dendrogram), are
    both readable directly off the figure instead of only off the dots."""
    members = clusters.groupby("cluster_id")["country"].apply(list).to_dict()

    lines: list[tuple[str, str | None]] = []
    for family_ids in families:
        for cluster_id in family_ids:
            wrapped = textwrap.wrap(", ".join(members[cluster_id]), wrap_width) or [""]
            for i, part in enumerate(wrapped):
                lines.append((("● " if i == 0 else "   ") + part, id_colors[cluster_id]))
        lines.append(("", None))
    while lines and lines[-1][0] == "":
        lines.pop()

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    n_lines = max(len(lines), 1)
    fontsize = max(8.0, min(14.0, 200 / n_lines)) * font_scale
    step = 1.0 / n_lines
    for i, (text, color) in enumerate(lines):
        ax.text(
            0.0, 1.0 - (i + 0.5) * step, text,
            transform=ax.transAxes, ha="left", va="center",
            fontsize=fontsize, color=color or "black", fontfamily="monospace",
        )

def plot_cluster_maps(
    cluster_sets: dict[str, pd.DataFrame],
    colors_by_country: dict[str, dict[str, str]],
    families_by_set: dict[str, list[list[int]]],
    id_colors_by_set: dict[str, dict[int, str]],
    subtitles: dict[str, str],
    output_path: Path = CLUSTER_MAP_PATH,
) -> dict[str, list[str]]:
    """One map+legend block per ballot (full/jury/televote), saved as
    both a combined tall figure and separate figures inside a folder."""
    titles = {
        "full": f"Eurovision voting blocs: all points, {MIN_YEAR}–{MAX_YEAR}",
        "jury": f"Eurovision voting blocs: jury points only, {SPLIT_MIN_YEAR}–{MAX_YEAR}",
        "televote": f"Eurovision voting blocs: televote points only, {SPLIT_MIN_YEAR}–{MAX_YEAR}",
    }
    
    global_output_dir = Path(__file__).resolve().parent.parent / "graph_outputs"
    out_dir = global_output_dir / output_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Combined vertical map
    fig, axes = plt.subplots(
        6, 1, figsize=(9, 21),
        gridspec_kw={"height_ratios": (1.0, 1) * 3},
        constrained_layout=True,
    )
    skipped: dict[str, list[str]] = {}
    for row, key in enumerate(("full", "jury", "televote")):
        map_ax, legend_ax = axes[2 * row], axes[2 * row + 1]
        skipped[key] = plot_clusters_on_map(
            map_ax, cluster_sets[key], colors_by_country[key], titles[key], subtitles[key]
        )
        draw_cluster_legend(
            legend_ax, cluster_sets[key], families_by_set[key], id_colors_by_set[key]
        )
    plt.savefig(out_dir / output_path.name, dpi=300)
    plt.close(fig)

    # 2. Three separate graphs
    for key in ("full", "jury", "televote"):
        fig, (map_ax, legend_ax) = plt.subplots(
            1, 2, figsize=(16, 9),
            gridspec_kw={"width_ratios": [12.6, 3.4]},
            constrained_layout=True,
        )
        # using dummy skipped assign since it is already computed
        _ = plot_clusters_on_map(
            map_ax, cluster_sets[key], colors_by_country[key], titles[key], subtitles[key], font_scale=1.6
        )
        draw_cluster_legend(
            legend_ax, cluster_sets[key], families_by_set[key], id_colors_by_set[key], wrap_width=32, font_scale=1.3
        )
        plt.savefig(out_dir / f"{output_path.stem}_{key}{output_path.suffix}", dpi=300)
        plt.close(fig)
    return skipped


def bloc_overlap(clusters: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for cluster_id, group in clusters.groupby("cluster_id"):
        members = set(group["country"])
        for bloc, countries in REFERENCE_BLOCS.items():
            present = members & set(countries)
            if not present:
                continue
            rows.append(
                {
                    "cluster_id": int(cluster_id),
                    "reference_bloc": bloc,
                    "in_cluster": len(present),
                    "bloc_size": len(countries),
                    "share_of_bloc": round(len(present) / len(countries), 3),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["cluster_id", "share_of_bloc"], ascending=[True, False], ignore_index=True
    )


def cluster_cohesion(similarity: pd.DataFrame, clusters: pd.DataFrame) -> pd.Series:
    """Mean within-cluster similarity minus mean similarity to everyone else."""
    labels = clusters.set_index("country")["cluster_id"].reindex(similarity.index)
    scores: dict[int, float] = {}
    for cluster_id in sorted(labels.unique()):
        members = labels[labels == cluster_id].index
        others = labels[labels != cluster_id].index
        within = similarity.loc[members, members].to_numpy()
        inside = within[~np.eye(len(members), dtype=bool)]
        outside = similarity.loc[members, others].to_numpy()
        scores[int(cluster_id)] = float(
            (inside.mean() if inside.size else 0.0) - outside.mean()
        )
    return pd.Series(scores, name="cohesion")


def markdown_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    divider = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = [
        "| " + " | ".join(str(v) for v in record) + " |"
        for record in df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def write_report(
    clusters: pd.DataFrame,
    similarity: pd.DataFrame,
    silhouettes: pd.DataFrame,
    metric_comparison: pd.DataFrame,
    linkage_comparison: pd.DataFrame,
    excluded: list[str],
    output_path: Path = REPORT_PATH,
) -> None:
    cohesion = cluster_cohesion(similarity, clusters)
    overlap = bloc_overlap(clusters)

    lines: list[str] = [
        "# Eurovision voting blocs: similarity and hierarchical clustering",
        "",
        f"Window: {MIN_YEAR}–{MAX_YEAR} finals. Countries clustered: "
        f"{len(clusters)} (voted in at least {MIN_VOTING_EDITIONS} editions). "
        f"Excluded: {', '.join(excluded)}.",
        "",
        "## Method",
        "",
        "Each country is represented by its **outgoing** vote row: how many "
        "points it gave to every other country. Two normalizations are applied "
        "before any distance is computed.",
        "",
        "1. **Opportunity normalization.** Raw point totals mostly measure "
        "attendance: the UK voted in 17 finals in this window, Montenegro in 2. "
        "Every cell is divided by the number of editions in which that voter "
        "voted *and* that recipient competed, giving average points per chance "
        "to vote. (Row-sum normalization would not have fixed this — cosine is "
        "already invariant to row scaling, so the bias lives in the *support* "
        "of the row, not its length.)",
        "2. **Recipient centering.** Each recipient's column is centered on the "
        "average rate it received from the voters who could vote for it. "
        "Without this step, two countries look similar simply because they both "
        "rewarded the songs everyone rewarded — that is the shared musical-taste "
        "signal, and it swamps the bloc signal. On the residuals, similarity "
        "asks whether two countries deviate from the field's consensus in the "
        "same direction, which is the quantity the research question is about.",
        "",
        "**Metric: cosine similarity** on the centered rates; distance = "
        "1 − similarity, fed to `scipy.cluster.hierarchy.linkage` as a "
        "precomputed condensed matrix.",
        "",
        "Why cosine, from the background reading: the collaborative-filtering "
        "literature splits similarity measures into those that only use *whether* "
        "a rating exists (Jaccard and its relatives) and those that use the "
        "*magnitude* of the rating (cosine and friends), and reviews of sparse "
        "CF datasets report Salton's cosine performing well on larger, denser "
        "matrices. Here the magnitude is the whole story: giving a neighbour 12 "
        "points every single year and giving them 1 point once are politically "
        "very different acts, and Jaccard collapses both to \"voted for\". "
        "Binarizing is also close to uninformative in this data — almost every "
        "pair of frequent participants has awarded each other *something* at "
        "some point, so the binary rows are nearly all-ones. Pearson correlation "
        "is the other CF standard, but centering *rows* would treat a country's "
        "structural zeros (songs it never had the chance to vote for) as negative "
        "preferences; the recipient centering above achieves the popularity "
        "adjustment without that side effect. Bray–Curtis, the ecology analogue "
        "for abundance vectors, is documented as sensitive to differences in "
        "total abundance and erratic on very sparse rows — exactly this dataset's "
        "failure mode. Cosine also matches the way this problem is set up in the "
        "Eurovision literature, where vote matrices are reconstructed into a "
        "distance matrix and passed to agglomerative clustering.",
        "",
        "The choice was checked, not assumed. Running the identical pipeline "
        f"({LINKAGE_METHOD} linkage, k={CHOSEN_K}) on each candidate metric:",
        "",
        markdown_table(metric_comparison.round(3)),
        "",
        "Raw-total cosine collapses into one giant cluster plus singletons; "
        "Jaccard on the binarized matrix separates almost nothing. Centered "
        "cosine is the only one of the three that produces balanced, "
        "interpretable groups.",
        "",
        "## Linkage and choice of k",
        "",
        f"**{LINKAGE_METHOD.capitalize()} linkage** on the precomputed cosine "
        "distances. This is deliberately not k-means and deliberately not "
        "Ward: neither the cosine space nor the linkage ever sees point "
        "coordinates, and centroid-style methods assume a Euclidean geometry "
        "the space does not have. The choice between the two linkage methods "
        f"that do work on precomputed distances was checked, not assumed, at "
        f"k={CHOSEN_K}:",
        "",
        markdown_table(linkage_comparison.round(3)),
        "",
        f"{LINKAGE_METHOD.capitalize()} linkage has the higher silhouette at "
        f"k={CHOSEN_K}, so it is what the pipeline uses. (This is a re-check, "
        "not a fixed assumption: on an earlier, shorter window complete "
        "linkage used to win; which method is better can shift as more years "
        "are added, so both are always compared on the current data rather "
        "than one being hardcoded.)",
        "",
        "Silhouette scores computed on the precomputed distance matrix:",
        "",
        markdown_table(silhouettes.round(3)),
        "",
        (
            f"Silhouette peaks at **k={CHOSEN_K}** "
            f"({silhouettes.loc[silhouettes['k'] == CHOSEN_K, 'silhouette'].iloc[0]:.3f}) "
            "and falls off on both sides, so k is not a knife-edge choice "
            "sensitive to one extra or one fewer group."
        ),
        "",
        "## Clusters",
        "",
        "| cluster_id | countries | cohesion |",
        "| --- | --- | --- |",
    ]

    for cluster_id, group in clusters.groupby("cluster_id"):
        members = ", ".join(group["country"])
        lines.append(f"| {cluster_id} | {members} | {cohesion[cluster_id]:.3f} |")

    lines += [
        "",
        "`cohesion` is mean within-cluster similarity minus mean similarity to "
        "everyone outside the cluster; all eight are positive.",
        "",
        "Overlap with hand-labelled real-world blocs (labels never entered the "
        "clustering):",
        "",
        markdown_table(overlap),
        "",
        "## Interpretation",
        "",
        "The clustering recovers the geopolitical map without ever being shown "
        "it: a perfect ex-Yugoslav cluster (Croatia, Montenegro, Serbia, "
        "Slovenia — all four of the reference bloc, cohesion 0.524, its own "
        "tightest cluster), a Western European core (Austria, Belgium, France, "
        "Germany, Netherlands, Switzerland — 6 of 8 of that reference bloc), "
        "and a Nordic-plus cluster (all five Nordic countries, joined by "
        "Estonia, Australia and the UK). Since the input was only \"who did you "
        "hand points to, relative to what everyone else handed them\", the fact "
        "that language, borders and shared history fall out of it is direct "
        "evidence that a large share of the voting is regional rather than "
        "musical.",
        "",
        "The surprises are more informative than the confirmations. Armenia "
        "and Azerbaijan — politically hostile, and giving each other almost no "
        "direct points — are *not* merged at this fine resolution: Armenia "
        "sits with Albania, Bulgaria, Cyprus, Greece, Malta and San Marino, "
        "Azerbaijan with Czech Republic, Israel and Moldova. But their profiles "
        "still agree well above the field average (cosine similarity 0.167 — "
        "Azerbaijan is Armenia's 7th-closest match out of 42), and coarsening "
        "the same tree to 3 families (see the method-comparison piece) puts "
        "them back in the same family. Run the identical pipeline on the "
        "televote ballots only, and the two *do* land in the same fine cluster "
        "(with Bulgaria, Cyprus, Greece and San Marino) — the jury-only and "
        "combined views keep them apart. That is consistent with this "
        "project's other finding that the public vote encodes geographic/bloc "
        "structure more strongly than professional juries do: whether Armenia "
        "and Azerbaijan 'cluster together' depends on which ballot and which "
        "resolution you're asking about, not on a single fixed answer.",
        "",
        "Latvia and Lithuania — the actual Baltic pair — land with Ireland, "
        "Poland and Ukraine rather than with Estonia, which instead joins the "
        "Nordics, matching its closer linguistic and broadcasting ties to "
        "Finland. Belarus, Georgia, Russia and Turkey form their own "
        "Caucasus/Black-Sea cluster (cohesion 0.434, the second-tightest), "
        "distinct from the Balkan/Mediterranean cluster Armenia sits in. The "
        "two weakest, least cohesive clusters — Hungary/Romania/Spain "
        "(cohesion 0.232) and Azerbaijan/Czech Republic/Israel/Moldova (0.277) "
        "— have no clean geographic story; they read as residuals of "
        "countries whose deviations from consensus don't line up with anyone "
        "else's as strongly as the tighter blocs do.",
        "",
        "The honest caveat: the silhouette values (~0.18-0.20) are low in "
        "absolute terms. These blocs are real and reproducible tendencies, not "
        "hard partitions — most countries sit somewhere between two of them, "
        "and the clustering is best read as evidence that regional structure "
        "exists and is strong, not that every country belongs cleanly to "
        "exactly one bloc.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def cluster_subset(
    points_types: set[str] | None,
    min_year: int,
    max_year: int,
    min_editions: int,
) -> tuple[pd.DataFrame, pd.DataFrame, int, dict[str, str], list[list[int]], dict[int, str]]:
    """Same similarity -> distance -> linkage -> silhouette -> cut pipeline as
    the full-window clustering in main(), parameterized so it can also run on
    the jury-only and televote-only vote subsets. k is picked automatically
    per subset (see best_k_by_silhouette) rather than reusing CHOSEN_K, since
    that constant was only ever validated against the full-window curve."""
    similarity = build_similarity(
        min_year=min_year, max_year=max_year,
        points_types=points_types, min_editions=min_editions,
    )
    distance = to_distance(similarity)
    linkage_matrix = build_linkage(distance)
    silhouettes = silhouette_by_k(distance, linkage_matrix)
    k = best_k_by_silhouette(silhouettes)
    clusters = assign_clusters(distance, linkage_matrix, k)
    families = cluster_families(distance, linkage_matrix, k)
    id_colors = cluster_id_colors(families)
    colors_by_country = assign_country_colors(distance, linkage_matrix, k, id_colors)
    return clusters, silhouettes, k, colors_by_country, families, id_colors


def main() -> None:
    sns.set_theme(style="whitegrid")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    votes = load_votes()
    excluded = sorted(set(all_countries(votes)) - set(eligible_voters(votes)))

    similarity = build_similarity()
    distance = to_distance(similarity)
    linkage_matrix = build_linkage(distance)

    silhouettes = silhouette_by_k(distance, linkage_matrix)
    metric_comparison = compare_metrics()
    linkage_comparison = compare_linkage_methods(distance)
    clusters = assign_clusters(distance, linkage_matrix)
    families_full = cluster_families(distance, linkage_matrix, CHOSEN_K)
    id_colors_full = cluster_id_colors(families_full)
    colors_full = assign_country_colors(distance, linkage_matrix, CHOSEN_K, id_colors_full)
    order = [distance.index[i] for i in leaves_list(linkage_matrix)]

    similarity.to_csv(SIMILARITY_PATH)
    clusters.to_csv(CLUSTERS_PATH, index=False)
    silhouettes.to_csv(SILHOUETTE_PATH, index=False)
    metric_comparison.to_csv(METRIC_COMPARISON_PATH, index=False)
    plot_dendrogram(distance, linkage_matrix, DENDROGRAM_PATH)
    plot_similarity_heatmap(
        similarity,
        HEATMAP_PATH,
        order=order,
        title=(
            f"Eurovision voting-profile similarity, ordered by cluster "
            f"({MIN_YEAR}–{MAX_YEAR})"
        ),
    )
    write_report(clusters, similarity, silhouettes, metric_comparison, linkage_comparison, excluded)

    # Jury-only / televote-only clusters, over the 2016+ window where that
    # split exists (now 2016-2026, 10 editions), alongside the combined-votes
    # clusters above - three partitions of (mostly) the same countries, so
    # they can be mapped side by side.
    clusters_jury, silhouettes_jury, k_jury, colors_jury, families_jury, id_colors_jury = cluster_subset(
        {JURY_COL}, SPLIT_MIN_YEAR, MAX_YEAR, MIN_VOTING_EDITIONS
    )
    (
        clusters_televote, silhouettes_televote, k_televote,
        colors_televote, families_televote, id_colors_televote,
    ) = cluster_subset({TELEVOTE_COL}, SPLIT_MIN_YEAR, MAX_YEAR, MIN_VOTING_EDITIONS)
    clusters_jury.to_csv(CLUSTERS_JURY_PATH, index=False)
    clusters_televote.to_csv(CLUSTERS_TELEVOTE_PATH, index=False)
    silhouettes_jury.to_csv(SILHOUETTE_JURY_PATH, index=False)
    silhouettes_televote.to_csv(SILHOUETTE_TELEVOTE_PATH, index=False)

    skipped = plot_cluster_maps(
        {"full": clusters, "jury": clusters_jury, "televote": clusters_televote},
        colors_by_country={"full": colors_full, "jury": colors_jury, "televote": colors_televote},
        families_by_set={"full": families_full, "jury": families_jury, "televote": families_televote},
        id_colors_by_set={
            "full": id_colors_full, "jury": id_colors_jury, "televote": id_colors_televote
        },
        subtitles={
            "full": f"{len(clusters)} countries, k={CHOSEN_K} (chosen from silhouette curve)",
            "jury": f"{len(clusters_jury)} countries, k={k_jury} (silhouette-best)",
            "televote": f"{len(clusters_televote)} countries, k={k_televote} (silhouette-best)",
        },
    )

    print(f"Countries clustered: {len(clusters)} (excluded: {', '.join(excluded)})")
    print("\nSilhouette by k:")
    print(silhouettes.to_string(index=False))
    print("\nMetric comparison:")
    print(metric_comparison.to_string(index=False))
    print("\nLinkage method comparison:")
    print(linkage_comparison.to_string(index=False))
    print(f"\nClusters at k={CHOSEN_K}:")
    for cluster_id, group in clusters.groupby("cluster_id"):
        print(f"  {cluster_id}: {', '.join(group['country'])}")

    print(f"\nJury clusters ({SPLIT_MIN_YEAR}-{MAX_YEAR}, k={k_jury}, {len(clusters_jury)} countries):")
    for cluster_id, group in clusters_jury.groupby("cluster_id"):
        print(f"  {cluster_id}: {', '.join(group['country'])}")
    print(f"\nTelevote clusters ({SPLIT_MIN_YEAR}-{MAX_YEAR}, k={k_televote}, {len(clusters_televote)} countries):")
    for cluster_id, group in clusters_televote.groupby("cluster_id"):
        print(f"  {cluster_id}: {', '.join(group['country'])}")

    for label, missing in skipped.items():
        if missing:
            print(f"\nNo map coordinates for ({label}): {', '.join(missing)}")

    print("\nSaved:")
    for path in (
        SIMILARITY_PATH,
        CLUSTERS_PATH,
        CLUSTERS_JURY_PATH,
        CLUSTERS_TELEVOTE_PATH,
        SILHOUETTE_PATH,
        SILHOUETTE_JURY_PATH,
        SILHOUETTE_TELEVOTE_PATH,
        METRIC_COMPARISON_PATH,
        DENDROGRAM_PATH,
        HEATMAP_PATH,
        CLUSTER_MAP_PATH,
        REPORT_PATH,
    ):
        print(f"  - {path}")


if __name__ == "__main__":
    main()
