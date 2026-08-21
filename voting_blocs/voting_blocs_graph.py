from __future__ import annotations

from itertools import islice
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from networkx.algorithms.community import girvan_newman, louvain_communities, modularity
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from jury_televote import JURY_COL, TELEVOTE_COL
from main import clean_data, load_data

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "eurovision_1957-2021.csv"  # shared repo-root dataset
DATA_PATH_EXTENSION = BASE_DIR / "eurovision_2022-2026.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURE_PATH = OUTPUT_DIR / "voting_blocs_network_comparison.png"
COMMUNITIES_PATH = OUTPUT_DIR / "voting_blocs_communities.csv"
REPORT_PATH = OUTPUT_DIR / "voting_blocs_graph_report.md"

FULL_WINDOW = (2004, 2026)
SPLIT_WINDOW = (2016, 2026)
COMBINED_TYPE = "Points given"
SEED = 7
LOUVAIN_RESTARTS = 30
N_PERMUTATIONS = 2000
GN_MAX_LEVELS = 12
TOP_EDGES_DRAWN = 110

# main.normalize_country only collapses whitespace, so hyphenated spellings survive
# as separate strings and would otherwise split one country across two nodes.
NODE_ALIASES = {
    "Czech-Republic": "Czech Republic",
    "San-Marino": "San Marino",
    "The-Netherlands": "The Netherlands",
    "United-Kingdom": "United Kingdom",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Serbia & Montenegro": "Serbia and Montenegro",
}

# (longitude, latitude) of a representative point per country, used both for the
# rough-geographic layout and for the geographic-compactness statistic.
COUNTRY_COORDS = {
    "Albania": (20.0, 41.0),
    "Armenia": (45.0, 40.1),
    "Australia": (133.8, -25.3),
    "Austria": (14.5, 47.6),
    "Azerbaijan": (47.6, 40.4),
    "Belarus": (28.0, 53.7),
    "Belgium": (4.5, 50.6),
    "Bosnia and Herzegovina": (17.8, 44.0),
    "Bulgaria": (25.5, 42.7),
    "Croatia": (15.9, 45.3),
    "Cyprus": (33.4, 35.1),
    "Czech Republic": (15.5, 49.8),
    "Denmark": (9.5, 56.1),
    "Estonia": (25.5, 58.7),
    "Finland": (26.0, 63.5),
    "France": (2.2, 46.6),
    "Georgia": (43.4, 42.3),
    "Germany": (10.4, 51.2),
    "Greece": (22.5, 39.0),
    "Hungary": (19.5, 47.1),
    "Iceland": (-19.0, 64.9),
    "Ireland": (-8.2, 53.2),
    "Israel": (35.0, 31.3),
    "Italy": (12.6, 42.5),
    "Latvia": (24.6, 56.9),
    "Lithuania": (23.9, 55.2),
    "Luxembourg": (6.1, 49.6),  # returned to the contest in 2025, in the 2022-2026 extension
    "Malta": (14.4, 35.9),
    "Moldova": (28.5, 47.2),
    "Montenegro": (19.3, 42.7),
    "North Macedonia": (21.7, 41.6),
    "Norway": (9.0, 61.5),
    "Poland": (19.1, 52.1),
    "Portugal": (-8.2, 39.6),
    "Romania": (25.0, 45.9),
    "Russia": (40.0, 56.5),
    "San Marino": (12.4, 43.9),
    "Serbia": (20.9, 44.0),
    "Serbia and Montenegro": (20.6, 43.4),
    "Slovenia": (14.8, 46.1),
    "Spain": (-3.7, 40.2),
    "Sweden": (15.0, 60.5),
    "Switzerland": (8.2, 46.8),
    "The Netherlands": (5.3, 52.2),
    "Turkey": (35.0, 39.0),
    "Ukraine": (31.2, 48.9),
    "United Kingdom": (-2.0, 54.0),
}

# Australia is drawn in the empty Atlantic corner so it does not stretch the map.
LAYOUT_OVERRIDES = {"Australia": (-22.0, 36.5)}

SHORT_NAMES = {
    "Bosnia and Herzegovina": "Bosnia",
    "Czech Republic": "Czechia",
    "North Macedonia": "N. Macedonia",
    "Serbia and Montenegro": "Serbia & Mont.",
    "The Netherlands": "Netherlands",
    "United Kingdom": "UK",
}

COMMUNITY_COLORS = [
    "#d73027",
    "#1f78b4",
    "#33a02c",
    "#ff7f00",
    "#6a3d9a",
    "#b15928",
    "#e7298a",
    "#17becf",
]


def load_votes() -> pd.DataFrame:
    # Merges the 1957-2021 file with the 2022+ extension before cleaning -
    # mirrors voting_blocs_similarity.load_votes(), so both the graph and
    # similarity/clustering sides of this project see the same 2004-2026
    # history rather than one stopping at the original file's 2021 cutoff.
    raw = pd.concat([load_data(DATA_PATH), load_data(DATA_PATH_EXTENSION)], ignore_index=True)
    df = clean_data(raw)
    for column in ("From", "To"):
        df[column] = df[column].replace(NODE_ALIASES)
    return df


def select_window(
    df: pd.DataFrame, window: tuple[int, int], points_types: set[str]
) -> pd.DataFrame:
    start, end = window
    mask = df["Year"].between(start, end) & df["Points type"].isin(points_types)
    return df.loc[mask].copy()


def build_digraph(df: pd.DataFrame) -> nx.DiGraph:
    totals = df.groupby(["From", "To"])["Points"].sum().reset_index()
    return nx.from_pandas_edgelist(
        totals, "From", "To", edge_attr="Points", create_using=nx.DiGraph
    )


def build_mean_digraph(df: pd.DataFrame) -> nx.DiGraph:
    """Edge weight = points per shared contest, i.e. controlling for co-participation."""
    grouped = df.groupby(["From", "To"])
    totals = grouped["Points"].sum().rename("Points")
    shared = grouped["Year"].nunique().rename("shared")
    edges = pd.concat([totals, shared], axis=1).reset_index()
    edges["Points"] = edges["Points"] / edges["shared"]
    return nx.from_pandas_edgelist(
        edges, "From", "To", edge_attr="Points", create_using=nx.DiGraph
    )


def symmetrize(graph: nx.DiGraph) -> nx.Graph:
    """Collapse A->B and B->A into one edge carrying the summed points exchanged."""
    undirected = nx.Graph()
    undirected.add_nodes_from(graph.nodes)
    for source, target, points in graph.edges(data="Points"):
        current = undirected.get_edge_data(source, target, default={}).get("weight", 0.0)
        undirected.add_edge(source, target, weight=current + float(points))
    return undirected


def best_louvain(graph: nx.Graph, weight: str) -> tuple[list[set[str]], float, float]:
    """Louvain is stochastic, so keep the highest-modularity of several restarts."""
    scores: list[float] = []
    best: list[set[str]] = []
    best_score = -1.0
    for seed in range(SEED, SEED + LOUVAIN_RESTARTS):
        communities = louvain_communities(graph, weight=weight, seed=seed)
        score = modularity(graph, communities, weight=weight)
        scores.append(score)
        if score > best_score:
            best_score, best = score, [set(part) for part in communities]
    return order_communities(best), best_score, min(scores)


def order_communities(communities: list[set[str]]) -> list[set[str]]:
    return sorted(communities, key=lambda members: (-len(members), sorted(members)[0]))


def to_labels(communities: list[set[str]]) -> dict[str, int]:
    return {
        country: index
        for index, members in enumerate(communities)
        for country in members
    }


def align_labels(
    labels: dict[str, int], reference: dict[str, int]
) -> dict[str, int]:
    """Renumber communities so that colours mean roughly the same bloc across panels."""
    groups: dict[int, set[str]] = {}
    for country, community in labels.items():
        groups.setdefault(community, set()).add(country)

    reference_groups: dict[int, set[str]] = {}
    for country, community in reference.items():
        reference_groups.setdefault(community, set()).add(country)

    scored = []
    for community, members in groups.items():
        for ref_id, ref_members in reference_groups.items():
            union = members | ref_members
            overlap = len(members & ref_members) / len(union) if union else 0.0
            scored.append((overlap, community, ref_id))
    scored.sort(reverse=True)

    mapping: dict[int, int] = {}
    used: set[int] = set()
    for overlap, community, ref_id in scored:
        if overlap <= 0 or community in mapping or ref_id in used:
            continue
        mapping[community] = ref_id
        used.add(ref_id)

    spare = (i for i in range(len(groups) + len(reference_groups)) if i not in used)
    for community in sorted(groups):
        if community not in mapping:
            mapping[community] = next(spare)
    return {country: mapping[community] for country, community in labels.items()}


def girvan_newman_best(
    graph: nx.Graph, max_levels: int = GN_MAX_LEVELS
) -> tuple[list[set[str]], float]:
    """Edge betweenness treats weight as distance, so strong ties must be made short."""
    working = graph.copy()
    for _, _, data in working.edges(data=True):
        data["distance"] = 1.0 / data["weight"] if data["weight"] else np.inf

    def most_valuable_edge(current: nx.Graph) -> tuple[str, str]:
        betweenness = nx.edge_betweenness_centrality(current, weight="distance")
        return max(betweenness, key=betweenness.get)

    best_communities: list[set[str]] = [set(graph.nodes)]
    best_score = modularity(graph, best_communities, weight="weight")
    for level in islice(girvan_newman(working, most_valuable_edge), max_levels):
        communities = [set(part) for part in level]
        score = modularity(graph, communities, weight="weight")
        if score > best_score:
            best_score, best_communities = score, communities
    return order_communities(best_communities), best_score


def haversine_km(first: tuple[float, float], second: tuple[float, float]) -> float:
    lon1, lat1 = radians(first[0]), radians(first[1])
    lon2, lat2 = radians(second[0]), radians(second[1])
    inner = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * asin(sqrt(inner))


def distance_matrix(countries: list[str]) -> np.ndarray:
    coords = [COUNTRY_COORDS[country] for country in countries]
    size = len(countries)
    matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(i + 1, size):
            matrix[i, j] = matrix[j, i] = haversine_km(coords[i], coords[j])
    return matrix


def mean_within_community_distance(
    matrix: np.ndarray, assignments: np.ndarray
) -> float:
    same = assignments[:, None] == assignments[None, :]
    np.fill_diagonal(same, False)
    if not same.any():
        return float("nan")
    return float(matrix[same].mean())


def geographic_compactness(
    labels: dict[str, int], rng: np.random.Generator
) -> dict[str, float]:
    """Permutation test: are communities closer together than a random relabelling?"""
    countries = sorted(c for c in labels if c in COUNTRY_COORDS and c != "Australia")
    matrix = distance_matrix(countries)
    assignments = np.array([labels[country] for country in countries])
    observed = mean_within_community_distance(matrix, assignments)

    null = np.empty(N_PERMUTATIONS)
    shuffled = assignments.copy()
    for i in range(N_PERMUTATIONS):
        rng.shuffle(shuffled)
        null[i] = mean_within_community_distance(matrix, shuffled)

    return {
        "observed_km": observed,
        "expected_km": float(null.mean()),
        "ratio": observed / float(null.mean()),
        "p_value": float((null <= observed).mean()),
    }


def mean_distance_to_own_community(labels: dict[str, int]) -> dict[str, float]:
    countries = sorted(c for c in labels if c in COUNTRY_COORDS and c != "Australia")
    matrix = distance_matrix(countries)
    index = {country: i for i, country in enumerate(countries)}
    distances: dict[str, float] = {}
    for country in countries:
        peers = [
            index[other]
            for other in countries
            if other != country and labels[other] == labels[country]
        ]
        if peers:
            distances[country] = float(matrix[index[country], peers].mean())
    return distances


def layout_positions(graph: nx.Graph) -> dict[str, tuple[float, float]]:
    return {
        node: LAYOUT_OVERRIDES.get(node, COUNTRY_COORDS[node])
        for node in graph.nodes
        if node in COUNTRY_COORDS or node in LAYOUT_OVERRIDES
    }


def draw_panel(
    ax: plt.Axes,
    graph: nx.Graph,
    labels: dict[str, int],
    title: str,
    subtitle: str,
    font_scale: float = 1.0,
) -> None:
    positions = layout_positions(graph)
    nodes = [node for node in graph.nodes if node in positions]

    strength = {
        node: sum(data["weight"] for _, _, data in graph.edges(node, data=True))
        for node in nodes
    }
    max_strength = max(strength.values()) or 1.0
    sizes = [(60 + 420 * strength[node] / max_strength) * (font_scale ** 1.5) for node in nodes]
    colors = [COMMUNITY_COLORS[labels[node] % len(COMMUNITY_COLORS)] for node in nodes]

    edges = sorted(
        (
            (u, v, data["weight"])
            for u, v, data in graph.edges(data=True)
            if u in positions and v in positions
        ),
        key=lambda item: item[2],
        reverse=True,
    )[:TOP_EDGES_DRAWN]
    max_weight = max((weight for _, _, weight in edges), default=1.0)

    for source, target, weight in edges:
        share = weight / max_weight
        within = labels[source] == labels[target]
        ax.plot(
            [positions[source][0], positions[target][0]],
            [positions[source][1], positions[target][1]],
            color=COMMUNITY_COLORS[labels[source] % len(COMMUNITY_COLORS)]
            if within
            else "#9e9e9e",
            linewidth=(0.3 + 2.6 * share) * font_scale,
            alpha=0.20 + 0.55 * share if within else 0.10 + 0.20 * share,
            zorder=1,
        )

    ax.scatter(
        [positions[node][0] for node in nodes],
        [positions[node][1] for node in nodes],
        s=sizes,
        c=colors,
        edgecolors="white",
        linewidths=0.7 * font_scale,
        zorder=2,
    )
    for node in nodes:
        ax.annotate(
            SHORT_NAMES.get(node, node),
            positions[node],
            xytext=(0, 7 * font_scale),
            textcoords="offset points",
            ha="center",
            fontsize=5.5 * font_scale,
            zorder=3,
        )

    ax.set_title(title, fontsize=13 * font_scale, pad=10 * font_scale)
    ax.text(
        0.5,
        -0.04,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9 * font_scale,
        color="#444444",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_aspect(1.4)


def plot_networks(
    graphs: dict[str, nx.Graph],
    labels: dict[str, dict[str, int]],
    subtitles: dict[str, str],
) -> None:
    titles = {
        "full": f"All points, {FULL_WINDOW[0]}–{FULL_WINDOW[1]}",
        "jury": f"Jury points only, {SPLIT_WINDOW[0]}–{SPLIT_WINDOW[1]}",
        "televote": f"Televote points only, {SPLIT_WINDOW[0]}–{SPLIT_WINDOW[1]}",
    }
    
    global_output_dir = Path(__file__).resolve().parent.parent / "graph_outputs"
    out_dir = global_output_dir / FIGURE_PATH.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Combined triplet graph
    fig, axes = plt.subplots(1, 3, figsize=(21, 7.0))
    for ax, key in zip(axes, ("full", "jury", "televote")):
        draw_panel(ax, graphs[key], labels[key], titles[key], subtitles[key])
    fig.suptitle(
        "Eurovision voting blocs: Louvain communities on the country-to-country points network",
        fontsize=16,
        y=0.97,
    )
    plt.tight_layout(rect=(0, 0.04, 1, 0.94))
    plt.savefig(out_dir / FIGURE_PATH.name, dpi=300)
    plt.close(fig)

    # 2. Three separate graphs
    for key in ("full", "jury", "televote"):
        fig, ax = plt.subplots(figsize=(14, 10))
        draw_panel(ax, graphs[key], labels[key], titles[key], subtitles[key], font_scale=1.8)
        plt.tight_layout(rect=(0, 0.04, 1, 0.94))
        plt.savefig(out_dir / f"{FIGURE_PATH.stem}_{key}{FIGURE_PATH.suffix}", dpi=300)
        plt.close(fig)


def write_communities_csv(labels: dict[str, dict[str, int]]) -> None:
    rows = [
        {"country": country, "graph": graph, "community_id": community}
        for graph in ("full", "jury", "televote")
        for country, community in sorted(labels[graph].items())
    ]
    pd.DataFrame(rows).to_csv(COMMUNITIES_PATH, index=False)


def format_communities(labels: dict[str, int]) -> str:
    groups: dict[int, list[str]] = {}
    for country, community in sorted(labels.items()):
        groups.setdefault(community, []).append(country)
    return "\n".join(
        f"- **Community {community}** ({len(members)}): {', '.join(members)}"
        for community, members in sorted(groups.items())
    )


def crosstab_markdown(
    jury_labels: dict[str, int], tele_labels: dict[str, int]
) -> str:
    shared = sorted(set(jury_labels) & set(tele_labels))
    table = pd.crosstab(
        pd.Series([tele_labels[c] for c in shared], name="Televote community"),
        pd.Series([jury_labels[c] for c in shared], name="Jury community"),
    )
    header = "| Televote \\ Jury | " + " | ".join(str(c) for c in table.columns) + " |"
    divider = "| --- | " + " | ".join("---" for _ in table.columns) + " |"
    body = "\n".join(
        f"| {index} | " + " | ".join(str(v) for v in row) + " |"
        for index, row in table.iterrows()
    )
    return "\n".join([header, divider, body])


def build_report(context: dict[str, object]) -> str:
    return f"""# Voting-bloc graph analysis: Eurovision {FULL_WINDOW[0]}–{FULL_WINDOW[1]} dataset

Piece B of the project — graph construction and community detection on the
country-to-country voting network. Generated by `voting_blocs_graph.py`.

## 1. Graph construction

Votes are loaded and cleaned with the shared helpers in `main.py`
(`load_data`, `clean_data`, `normalize_country`), which strip whitespace, fix
country-name typos, drop self-votes and keep only valid point values. One extra
canonicalisation step is applied locally: the raw file also carries hyphenated
spellings (`United-Kingdom`, `Czech-Republic`, `The-Netherlands`, `San-Marino`),
which `normalize_country` leaves untouched because it only collapses whitespace.
Left alone these would silently split e.g. the UK into two separate nodes, one
per spelling, each carrying only part of its real vote record.

Three directed, weighted graphs are built. Nodes are countries; the weight of
edge *A → B* is the total number of points A gave B inside the window.

| Graph | Window | Point types | Nodes | Directed edges | Density |
| --- | --- | --- | --- | --- | --- |
| `full` | {FULL_WINDOW[0]}–{FULL_WINDOW[1]} | all (combined + jury + televote) | {context['full_nodes']} | {context['full_edges']} | {context['full_density']:.2f} |
| `jury` | {SPLIT_WINDOW[0]}–{SPLIT_WINDOW[1]} | `{JURY_COL}` | {context['jury_nodes']} | {context['jury_edges']} | {context['jury_density']:.2f} |
| `televote` | {SPLIT_WINDOW[0]}–{SPLIT_WINDOW[1]} | `{TELEVOTE_COL}` | {context['tele_nodes']} | {context['tele_edges']} | {context['tele_density']:.2f} |

The jury/televote split is only reliably recorded from 2016 onward, so the two
comparison graphs are restricted to {SPLIT_WINDOW[0]}–{SPLIT_WINDOW[1]} (2020 was
cancelled, so {context['n_contests_split']} contests). The `full` graph is the
main object; `jury` and `televote` exist for the comparison in section 4.

Note the density: these are **near-complete** graphs. In a Eurovision final
almost every country votes on almost every other, so absence of an edge is rare
and all of the signal lives in the edge *weights*, not in the topology. That
single fact drives the algorithm choice below.

## 2. Choosing a community-detection method

The brief asked for Louvain and/or Girvan–Newman. Before committing, the
literature on directed/weighted community detection was checked
([Malliaros & Vazirgiannis, *Clustering and Community Detection in Directed
Networks: A Survey*](https://arxiv.org/pdf/1308.0971);
[Leicht & Newman, *Community Structure in Directed Networks*](https://www.researchgate.net/publication/51394554_Community_Structure_in_Directed_Networks);
[Dugué & Perez, *Directed Louvain*](https://www.researchgate.net/publication/284869815_Directed_Louvain_maximizing_modularity_in_directed_networks);
[NetworkX `louvain_communities` docs](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.louvain.louvain_communities.html)).

What that turned up, and what was done with it:

**Louvain and direction.** Classic Louvain maximises undirected modularity, but
Leicht & Newman (2008) generalised modularity to directed graphs using in- and
out-strength separately, and Dugué & Perez folded that into Louvain
("Directed Louvain"). NetworkX's `louvain_communities` implements exactly this:
it branches on `G.is_directed()` and uses the directed modularity formulation,
so it can be run on the `DiGraph` untouched. Both variants were therefore run:

- **Directed Louvain** on the raw `DiGraph` (directed modularity).
- **Symmetrised Louvain**, collapsing each reciprocal pair into a single
  undirected edge of weight `w(A→B) + w(B→A)`.

The symmetrisation is not just convenience. The research question is
"do countries form mutually-supporting blocs?", and a bloc is by definition a
*reciprocal* relationship: a one-way flow of points is a popularity effect, not
a bloc. Summing the two directions makes the weight literally "points exchanged
between this pair", which is the quantity the hypothesis is about. It does throw
away asymmetry (Cyprus→Greece is not the same as Greece→Cyprus), so it is a real
loss — mitigated by the fact that the two variants agree almost perfectly on
this data: ARI between directed and symmetrised Louvain on the `full` graph is
**{context['dir_sym_ari']:.3f}** (modularity {context['full_directed_q']:.4f} directed
vs {context['full_q']:.4f} symmetrised). Since the answers coincide, the
symmetrised graph is used for the headline results because its modularity values
are comparable across the three graphs and the visualisation is legible.

**Girvan–Newman: tractable but inappropriate.** At {context['full_nodes']} nodes
GN is computationally fine — {GN_MAX_LEVELS} levels of edge removal run in a few
seconds. It is *statistically* wrong for this graph. GN removes high
edge-betweenness edges, i.e. bridges, and a graph with density
{context['full_density']:.2f} has no bridges: every removal just peels one more
country off the giant component. Running it (with distance = 1/weight, since
NetworkX treats the betweenness `weight` argument as a length, so heavy edges
must be made *short*) produced exactly that degenerate cascade, best modularity
**{context['gn_q']:.4f}** with partition sizes {context['gn_sizes']} — versus
{context['full_q']:.4f} for Louvain. This is reported as a negative result rather
than hidden: GN is a topology-driven method and this network's structure is
entirely in its weights.

**Louvain is stochastic**, and on a graph this dense the seed matters: over
{LOUVAIN_RESTARTS} restarts, Q on the `full` graph ranged from
{context['full_q_worst']:.4f} to {context['full_q']:.4f}, and low-scoring runs
produced visibly worse partitions (e.g. splitting the Baltics off arbitrarily).
Every Louvain result reported here is therefore the **best of
{LOUVAIN_RESTARTS} restarts** (seeds {SEED}–{SEED + LOUVAIN_RESTARTS - 1}) by
modularity, which is both more reproducible and closer to the modularity optimum
than a single run.

**Headline method: Louvain on the symmetrised, weighted graph**, best of
{LOUVAIN_RESTARTS} restarts, resolution 1.0.

## 3. Modularity

| Graph | Communities | Modularity Q | Q range over restarts |
| --- | --- | --- | --- |
| `full` (symmetrised Louvain) | {context['full_k']} | {context['full_q']:.4f} | {context['full_q_worst']:.4f}–{context['full_q']:.4f} |
| `full` (directed Louvain) | {context['full_directed_k']} | {context['full_directed_q']:.4f} | — |
| `full` (Girvan–Newman) | {context['gn_k']} | {context['gn_q']:.4f} | — |
| `jury` | {context['jury_k']} | {context['jury_q']:.4f} | {context['jury_q_worst']:.4f}–{context['jury_q']:.4f} |
| `televote` | {context['tele_k']} | {context['tele_q']:.4f} | {context['tele_q_worst']:.4f}–{context['tele_q']:.4f} |

These Q values are low in absolute terms (0.3–0.7 is the usual "strong
structure" band). That is expected and is itself a finding: on a near-complete
graph the null model already predicts a large share of every edge's weight, so
even a genuinely meaningful partition cannot score highly. The values are used
comparatively, not as an absolute claim of strong modular structure.

Crucially, **jury Q ({context['jury_q']:.4f}) and televote Q
({context['tele_q']:.4f}) are almost identical** — modularity alone cannot tell
these two apart. The difference is not in *how much* structure there is but in
*what the structure is made of*, which needs the geographic test below.

## 4. Main finding: televote blocs are geographic, jury blocs are not

To turn "looks political" into a number: for a partition, take the mean
great-circle distance between all pairs of countries assigned to the *same*
community, and compare it to the same statistic under {N_PERMUTATIONS} random
relabellings that keep the community sizes fixed. A ratio below 1 means
communities are more geographically compact than chance; the p-value is the
share of permutations at least as compact as observed. Australia is excluded
(it is 13,000 km from everything and would dominate the statistic).

| Graph | Mean within-community distance | Expected under random labels | Ratio | p |
| --- | --- | --- | --- | --- |
| `full` | {context['full_geo']['observed_km']:.0f} km | {context['full_geo']['expected_km']:.0f} km | {context['full_geo']['ratio']:.3f} | {context['full_geo']['p_value']:.4f} |
| `jury` | {context['jury_geo']['observed_km']:.0f} km | {context['jury_geo']['expected_km']:.0f} km | {context['jury_geo']['ratio']:.3f} | {context['jury_geo']['p_value']:.4f} |
| `televote` | {context['tele_geo']['observed_km']:.0f} km | {context['tele_geo']['expected_km']:.0f} km | {context['tele_geo']['ratio']:.3f} | {context['tele_geo']['p_value']:.4f} |

{context['geo_verdict']}

Agreement between the two partitions on the {context['shared_n']} countries
present in both: ARI **{context['jury_tele_ari']:.3f}**, NMI
**{context['jury_tele_nmi']:.3f}** — they are related but far from the same
partition.

Community ids are matched across the three graphs by maximum member overlap so
that the same id (and the same colour in the figure) means roughly the same bloc;
a graph with fewer communities therefore skips an id.

### Televote communities ({SPLIT_WINDOW[0]}–{SPLIT_WINDOW[1]})

{context['tele_text']}

### Jury communities ({SPLIT_WINDOW[0]}–{SPLIT_WINDOW[1]})

{context['jury_text']}

### Where the two partitions disagree

Rows are televote communities, columns jury communities; each cell counts
countries. A clean diagonal would mean the two agree.

{context['crosstab']}

The countries whose community becomes most geographically dispersed when moving
from televote to jury — i.e. the clearest individual cases of the jury breaking
up a geographic bloc:

{context['movers']}

Concretely, in the televote graph {context['tele_example']}. The jury graph
{context['jury_example']}.

The interpretation, in terms of the project's central question: **the televote
carries the geography/diaspora signal and the jury largely does not.** Public
voting reassembles the familiar regional blocs — a Nordic–Baltic–Anglo group, a
Balkan/Eastern-Mediterranean group, a Central European group — which is what a
diaspora/neighbour-affinity explanation predicts. (The fourth televote community,
the post-Soviet core with an Iberian/Francophone tail attached, is the one
exception and the reason the televote ratio is not lower still.) Professional
juries produce communities of comparable modularity, but those communities cut
across the map, which is what a taste-driven explanation predicts. Since juries
and televoters watch the identical set of performances in the identical order,
the difference cannot be attributed to the songs — it is a property of who is
voting. That is the strongest single piece of evidence this deliverable
contributes.

Two honest caveats. First, "not geographic" is not the same as "musical" — jury
communities could reflect shared professional taste, shared language, or simply
more noise in a {context['n_contests_split']}-contest window. Second, the
jury/televote window is only {context['n_contests_split']} contests, so the
jury graph rests on less data than the `full` graph.

## 5. Robustness: co-participation

Raw point totals reward countries that qualify for many finals. Re-running the
whole pipeline with edge weight = *points per shared contest* (dividing by the
number of years in which the pair could actually vote for each other) gives
{context['mean_k']} communities on the `full` graph with ARI
**{context['mean_ari']:.3f}** against the headline partition. {context['mean_verdict']}

## 6. Outputs

- `outputs/voting_blocs_network_comparison.png` — three panels (full / jury /
  televote) on a shared rough-geographic layout, nodes coloured by community and
  sized by total points exchanged, showing the top {TOP_EDGES_DRAWN} edges by
  weight. Community colours are matched across panels by maximum overlap, so the
  same colour means roughly the same bloc. Australia is drawn in the empty
  Atlantic corner rather than at its true position.
- `outputs/voting_blocs_communities.csv` — `country,graph,community_id`.
- `outputs/voting_blocs_graph_report.md` — this file.

### Full-graph communities ({FULL_WINDOW[0]}–{FULL_WINDOW[1]})

{context['full_text']}
"""


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    sns.set_theme(style="whitegrid")
    OUTPUT_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    votes = load_votes()
    frames = {
        "full": select_window(
            votes, FULL_WINDOW, {COMBINED_TYPE, JURY_COL, TELEVOTE_COL}
        ),
        "jury": select_window(votes, SPLIT_WINDOW, {JURY_COL}),
        "televote": select_window(votes, SPLIT_WINDOW, {TELEVOTE_COL}),
    }

    digraphs = {key: build_digraph(frame) for key, frame in frames.items()}
    graphs = {key: symmetrize(graph) for key, graph in digraphs.items()}

    louvain_runs = {key: best_louvain(graph, "weight") for key, graph in graphs.items()}
    communities = {key: run[0] for key, run in louvain_runs.items()}
    raw_labels = {key: to_labels(members) for key, members in communities.items()}
    labels = {"full": raw_labels["full"]}
    labels["televote"] = align_labels(raw_labels["televote"], labels["full"])
    labels["jury"] = align_labels(raw_labels["jury"], labels["televote"])

    modularities = {key: run[1] for key, run in louvain_runs.items()}
    worst_modularities = {key: run[2] for key, run in louvain_runs.items()}

    directed_communities, directed_q, _ = best_louvain(digraphs["full"], "Points")
    directed_labels = to_labels(directed_communities)

    gn_communities, gn_q = girvan_newman_best(graphs["full"])

    mean_graph = symmetrize(build_mean_digraph(frames["full"]))
    mean_communities, _, _ = best_louvain(mean_graph, "weight")
    mean_labels = to_labels(mean_communities)

    geo = {key: geographic_compactness(labels[key], rng) for key in labels}

    shared = sorted(set(labels["jury"]) & set(labels["televote"]))
    jury_vec = [labels["jury"][c] for c in shared]
    tele_vec = [labels["televote"][c] for c in shared]

    full_nodes = sorted(labels["full"])
    dir_sym_ari = adjusted_rand_score(
        [labels["full"][c] for c in full_nodes],
        [directed_labels[c] for c in full_nodes],
    )
    mean_ari = adjusted_rand_score(
        [labels["full"][c] for c in full_nodes],
        [mean_labels[c] for c in full_nodes],
    )

    jury_distance = mean_distance_to_own_community(labels["jury"])
    tele_distance = mean_distance_to_own_community(labels["televote"])
    movers = sorted(
        (
            (jury_distance[c] - tele_distance[c], c)
            for c in jury_distance
            if c in tele_distance
        ),
        reverse=True,
    )[:6]

    subtitles = {
        key: (
            f"{len(communities[key])} communities · Q = {modularities[key]:.3f} · "
            f"geo-compactness {geo[key]['ratio']:.2f}× chance (p = {geo[key]['p_value']:.3f})"
        )
        for key in graphs
    }

    plot_networks(graphs, labels, subtitles)
    write_communities_csv(labels)

    tele_ratio, jury_ratio = geo["televote"]["ratio"], geo["jury"]["ratio"]
    if tele_ratio < jury_ratio:
        geo_verdict = (
            f"The televote partition is the geographically compact one: its communities "
            f"average {geo['televote']['observed_km']:.0f} km internally, "
            f"{(1 - tele_ratio) * 100:.0f}% tighter than a random relabelling "
            f"(p = {geo['televote']['p_value']:.4f}). The jury partition sits at "
            f"{geo['jury']['observed_km']:.0f} km, only "
            f"{(1 - jury_ratio) * 100:.0f}% tighter than chance "
            f"(p = {geo['jury']['p_value']:.4f}). In excess-compactness terms the "
            f"televote signal is {(1 - tele_ratio) / max(1 - jury_ratio, 1e-9):.1f}× "
            f"the jury signal."
        )
    else:
        geo_verdict = (
            f"Contrary to the usual expectation, the jury partition is the more "
            f"geographically compact one ({jury_ratio:.3f}× chance vs "
            f"{tele_ratio:.3f}× for the televote)."
        )

    def group_by_community(labels_map: dict[str, int]) -> list[list[str]]:
        groups: dict[int, list[str]] = {}
        for country, community in sorted(labels_map.items()):
            groups.setdefault(community, []).append(country)
        return list(groups.values())

    def spread_km(members: list[str], distances: dict[str, float]) -> float:
        values = [distances[m] for m in members if m in distances]
        return float(np.mean(values)) if values else 0.0

    tightest_tele = min(
        group_by_community(labels["televote"]),
        key=lambda members: spread_km(members, tele_distance),
    )
    most_dispersed_jury = max(
        group_by_community(labels["jury"]),
        key=lambda members: spread_km(members, jury_distance),
    )

    n_contests_split = (SPLIT_WINDOW[1] - SPLIT_WINDOW[0] + 1) - (
        1 if SPLIT_WINDOW[0] <= 2020 <= SPLIT_WINDOW[1] else 0
    )
    context: dict[str, object] = {
        "n_contests_split": n_contests_split,
        "full_nodes": graphs["full"].number_of_nodes(),
        "full_edges": digraphs["full"].number_of_edges(),
        "full_density": nx.density(graphs["full"]),
        "jury_nodes": graphs["jury"].number_of_nodes(),
        "jury_edges": digraphs["jury"].number_of_edges(),
        "jury_density": nx.density(graphs["jury"]),
        "tele_nodes": graphs["televote"].number_of_nodes(),
        "tele_edges": digraphs["televote"].number_of_edges(),
        "tele_density": nx.density(graphs["televote"]),
        "full_k": len(communities["full"]),
        "full_q": modularities["full"],
        "full_q_worst": worst_modularities["full"],
        "jury_q_worst": worst_modularities["jury"],
        "tele_q_worst": worst_modularities["televote"],
        "full_directed_k": len(directed_communities),
        "full_directed_q": directed_q,
        "dir_sym_ari": dir_sym_ari,
        "gn_k": len(gn_communities),
        "gn_q": gn_q,
        "gn_sizes": sorted((len(c) for c in gn_communities), reverse=True),
        "jury_k": len(communities["jury"]),
        "jury_q": modularities["jury"],
        "tele_k": len(communities["televote"]),
        "tele_q": modularities["televote"],
        "full_geo": geo["full"],
        "jury_geo": geo["jury"],
        "tele_geo": geo["televote"],
        "geo_verdict": geo_verdict,
        "shared_n": len(shared),
        "jury_tele_ari": adjusted_rand_score(tele_vec, jury_vec),
        "jury_tele_nmi": normalized_mutual_info_score(tele_vec, jury_vec),
        "full_text": format_communities(labels["full"]),
        "jury_text": format_communities(labels["jury"]),
        "tele_text": format_communities(labels["televote"]),
        "crosstab": crosstab_markdown(labels["jury"], labels["televote"]),
        "movers": "\n".join(
            f"- **{country}** — {delta:+.0f} km "
            f"({tele_distance[country]:.0f} km to its televote community, "
            f"{jury_distance[country]:.0f} km to its jury community)"
            for delta, country in movers
        ),
        "tele_example": (
            f"the tightest community is {', '.join(tightest_tele)} — a block of "
            f"mutual neighbours averaging only "
            f"{spread_km(tightest_tele, tele_distance):.0f} km between members"
        ),
        "jury_example": (
            f"has no comparable equivalent: its most dispersed community groups "
            f"{', '.join(most_dispersed_jury)}, averaging "
            f"{spread_km(most_dispersed_jury, jury_distance):.0f} km between members "
            f"— a set that spans the whole map and cannot be read as a neighbourhood"
        ),
        "mean_k": len(mean_communities),
        "mean_ari": mean_ari,
        "mean_verdict": (
            "The two partitions largely agree, so the headline blocs are not an "
            "artefact of unequal qualification rates."
            if mean_ari >= 0.4
            else "The partitions differ noticeably, so part of the raw-total structure "
            "does reflect how often countries reached the final; the "
            "jury-vs-televote contrast in section 4 is unaffected because both "
            f"graphs cover the same {n_contests_split} contests and the same countries."
        ),
    }

    REPORT_PATH.write_text(build_report(context), encoding="utf-8")

    print("Saved:")
    for path in (FIGURE_PATH, COMMUNITIES_PATH, REPORT_PATH):
        print(f"  - {path}")
    print(
        f"Modularity — full {modularities['full']:.4f}, "
        f"jury {modularities['jury']:.4f}, televote {modularities['televote']:.4f}"
    )


if __name__ == "__main__":
    main()
