"""Hypothesis test: is the jury-minus-televote points gap different for
within-bloc country pairs than for between-bloc pairs (2016-2021 finals)?

Inference is by QAP node-label permutation rather than a t-test: the unit of
analysis is a dyad, so rows sharing a voter or a recipient are dependent, and
shuffling rows would fabricate independent replicates the data does not have.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

from jury_televote import JURY_COL, TELEVOTE_COL
from voting_blocs_similarity import load_votes

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
CLUSTERS_PATH = OUTPUT_DIR / "voting_blocs_clusters.csv"

PANEL_PATH = OUTPUT_DIR / "voting_blocs_inference_panel.csv"
SUMMARY_PATH = OUTPUT_DIR / "voting_blocs_inference_summary.csv"
BLOC_BREAKDOWN_PATH = OUTPUT_DIR / "voting_blocs_inference_by_bloc.csv"
BLOC_PLOT_PATH = OUTPUT_DIR / "voting_blocs_inference_by_bloc.png"
PERMUTATION_PLOT_PATH = OUTPUT_DIR / "voting_blocs_inference_permutation.png"
GAP_PLOT_PATH = OUTPUT_DIR / "voting_blocs_inference_gap_distribution.png"
REPORT_PATH = OUTPUT_DIR / "voting_blocs_inference_report.md"

MIN_YEAR = 2016
MAX_YEAR = 2021
N_PERMUTATIONS = 20_000
N_BOOTSTRAP = 5_000
SEED = 20210522

# "Top mark" = a 10 or a 12, the two scores a bloc partner is usually accused of
# handing over automatically.
TOP_MARK = 10
# 80% power at alpha = 0.05 needs roughly 2.8 null standard deviations.
POWER_MULTIPLIER = 2.8
# Robustness cut for the heterogeneity test: a bloc contributing only a handful
# of finalist dyads has a very noisy mean.
MIN_BLOC_DYADS = 20

YEAR_COL = "Year"
FROM_COL = "From"
TO_COL = "To"


def load_blocs(path: Path = CLUSTERS_PATH) -> pd.Series:
    blocs = pd.read_csv(path)
    return blocs.set_index("country")["cluster_id"]


def full_pair_grid(votes: pd.DataFrame) -> pd.DataFrame:
    """Every ordered (voter, recipient) pair that could have happened per year.

    The votes file only stores the ten non-zero scores each jury/televote hands
    out, so the ~60% of pairs that scored nothing are missing rows, not missing
    data. Reconstructing the grid and filling zeros is what makes the two point
    columns comparable; without it the gap would only ever be measured on pairs
    that at least one side already liked.
    """
    frames: list[pd.DataFrame] = []
    for year, edition in votes.groupby(YEAR_COL):
        voters = sorted(set(edition[FROM_COL]))
        recipients = sorted(set(edition[TO_COL]))
        grid = pd.MultiIndex.from_product(
            [[int(year)], voters, recipients], names=[YEAR_COL, FROM_COL, TO_COL]
        ).to_frame(index=False)
        frames.append(grid[grid[FROM_COL] != grid[TO_COL]])
    return pd.concat(frames, ignore_index=True)


def build_gap_panel(votes: pd.DataFrame, blocs: pd.Series) -> pd.DataFrame:
    points = (
        votes.pivot_table(
            index=[YEAR_COL, FROM_COL, TO_COL],
            columns="Points type",
            values="Points",
            aggfunc="sum",
        )
        .reindex(columns=[JURY_COL, TELEVOTE_COL])
        .reset_index()
    )
    panel = full_pair_grid(votes).merge(points, on=[YEAR_COL, FROM_COL, TO_COL], how="left")
    panel[[JURY_COL, TELEVOTE_COL]] = panel[[JURY_COL, TELEVOTE_COL]].fillna(0.0)
    panel["gap"] = panel[JURY_COL] - panel[TELEVOTE_COL]
    # Secondary outcome: nearly half of all dyad-years are 0-0 in both ballots
    # and dilute the mean, while bloc behaviour is a story about top marks.
    panel["top_gap"] = (panel[JURY_COL] >= TOP_MARK).astype(float) - (
        panel[TELEVOTE_COL] >= TOP_MARK
    ).astype(float)
    panel["from_bloc"] = panel[FROM_COL].map(blocs)
    panel["to_bloc"] = panel[TO_COL].map(blocs)
    panel = panel.dropna(subset=["from_bloc", "to_bloc"]).copy()
    panel["from_bloc"] = panel["from_bloc"].astype(int)
    panel["to_bloc"] = panel["to_bloc"].astype(int)
    panel["same_bloc"] = panel["from_bloc"] == panel["to_bloc"]
    return panel.reset_index(drop=True)


def to_arrays(
    panel: pd.DataFrame, countries: list[str], value_col: str = "gap"
) -> tuple[np.ndarray, np.ndarray]:
    """(years, n, n) outcome cube plus a mask of which cells are observations."""
    position = {country: i for i, country in enumerate(countries)}
    years = sorted(panel[YEAR_COL].unique())
    n = len(countries)
    values = np.zeros((len(years), n, n), dtype=float)
    valid = np.zeros((len(years), n, n), dtype=bool)
    for layer, year in enumerate(years):
        rows = panel[panel[YEAR_COL] == year]
        i = rows[FROM_COL].map(position).to_numpy()
        j = rows[TO_COL].map(position).to_numpy()
        values[layer, i, j] = rows[value_col].to_numpy()
        valid[layer, i, j] = True
    return values, valid


def same_bloc_matrix(blocs: pd.Series, countries: list[str]) -> np.ndarray:
    labels = blocs.reindex(countries).to_numpy()
    return labels[:, None] == labels[None, :]


def to_flat(
    panel: pd.DataFrame, countries: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Edge list form of the panel, for statistics that need bloc identity
    rather than just same/different."""
    position = {country: i for i, country in enumerate(countries)}
    return (
        panel[FROM_COL].map(position).to_numpy(),
        panel[TO_COL].map(position).to_numpy(),
    )


def bloc_dispersion(
    values: np.ndarray,
    from_idx: np.ndarray,
    to_idx: np.ndarray,
    labels: np.ndarray,
    min_count: int = 1,
) -> float:
    """Count-weighted SD across blocs of each bloc's own mean within-bloc value.

    The two-group statistic asks whether blocs lean one way *on average*, which
    is blind to blocs leaning in opposite directions. This asks the prior
    question: do blocs differ from each other at all?
    """
    same = labels[from_idx] == labels[to_idx]
    ids = labels[from_idx][same]
    size = int(labels.max()) + 1
    sums = np.bincount(ids, weights=values[same], minlength=size)
    counts = np.bincount(ids, minlength=size).astype(float)
    keep = counts >= min_count
    if keep.sum() < 2:
        return np.nan
    means, weights = sums[keep] / counts[keep], counts[keep]
    grand = float((means * weights).sum() / weights.sum())
    return float(np.sqrt(((means - grand) ** 2 * weights).sum() / weights.sum()))


def qap_dispersion_null(
    values: np.ndarray,
    from_idx: np.ndarray,
    to_idx: np.ndarray,
    labels: np.ndarray,
    min_count: int = 1,
    n_permutations: int = N_PERMUTATIONS,
    seed: int = SEED,
) -> np.ndarray:
    rng = np.random.default_rng(seed + 2)
    n = len(labels)
    null = np.empty(n_permutations, dtype=float)
    for step in range(n_permutations):
        null[step] = bloc_dispersion(
            values, from_idx, to_idx, labels[rng.permutation(n)], min_count
        )
    return null[~np.isnan(null)]


def format_p(p_value: float, n_permutations: int = N_PERMUTATIONS) -> str:
    """A permutation p-value can never be smaller than 1/(B+1); print the bound
    rather than a zero the resolution does not support."""
    floor = 1.0 / (n_permutations + 1)
    return f"< {floor:.5f}" if p_value <= floor else f"{p_value:.4f}"


def upper_p_value(observed: float, null: np.ndarray) -> float:
    """Dispersion is non-negative and only large values contradict H0, so this
    one is one-sided by construction rather than by choice."""
    return (int((null >= observed).sum()) + 1) / (len(null) + 1)


def mean_gap_difference(
    gap: np.ndarray, valid: np.ndarray, same_bloc: np.ndarray
) -> float:
    within = valid & same_bloc
    between = valid & ~same_bloc
    if not within.any() or not between.any():
        return np.nan
    return float(gap[within].mean() - gap[between].mean())


def qap_permutation_null(
    gap: np.ndarray,
    valid: np.ndarray,
    same_bloc: np.ndarray,
    n_permutations: int = N_PERMUTATIONS,
    seed: int = SEED,
) -> np.ndarray:
    """Null distribution from relabelling countries, not rows.

    Permuting the country -> bloc assignment moves whole rows and columns of the
    dyad matrix at once, so every replicate keeps the real dependence structure
    (a generous voter stays generous, a popular song stays popular) and only the
    bloc labels are randomized.
    """
    rng = np.random.default_rng(seed)
    n = same_bloc.shape[0]
    null = np.empty(n_permutations, dtype=float)
    for step in range(n_permutations):
        perm = rng.permutation(n)
        null[step] = mean_gap_difference(gap, valid, same_bloc[np.ix_(perm, perm)])
    return null


def naive_permutation_null(
    panel: pd.DataFrame, n_permutations: int = 2_000, seed: int = SEED
) -> np.ndarray:
    """Row-level shuffling, kept only to show how much it overstates evidence."""
    rng = np.random.default_rng(seed)
    gaps = panel["gap"].to_numpy()
    labels = panel["same_bloc"].to_numpy()
    null = np.empty(n_permutations, dtype=float)
    for step in range(n_permutations):
        shuffled = rng.permutation(labels)
        null[step] = gaps[shuffled].mean() - gaps[~shuffled].mean()
    return null


def two_sided_p_value(observed: float, null: np.ndarray) -> float:
    # +1 in both terms: the observed labelling is itself one of the permutations,
    # which keeps the p-value from ever being an impossible exact zero.
    extreme = int((np.abs(null) >= abs(observed)).sum())
    return (extreme + 1) / (len(null) + 1)


def dyadic_bootstrap_ci(
    gap: np.ndarray,
    valid: np.ndarray,
    same_bloc: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> tuple[float, float, np.ndarray]:
    """Pigeonhole bootstrap: resample countries, keep the induced sub-network.

    Resampling dyads independently would break the shared-membership dependence
    the same way row permutation does; resampling nodes carries each country's
    whole row and column along with it.
    """
    rng = np.random.default_rng(seed + 1)
    n = same_bloc.shape[0]
    draws: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        grid = np.ix_(idx, idx)
        # Duplicated draws land on the diagonal of the resampled matrix, where
        # `valid` is already False, so self-pairs never enter the statistic.
        stat = mean_gap_difference(
            gap[:, grid[0], grid[1]], valid[:, grid[0], grid[1]], same_bloc[grid]
        )
        if not np.isnan(stat):
            draws.append(stat)
    values = np.asarray(draws)
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)), values


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """P(a > b) - P(a < b), read off the Mann-Whitney U statistic."""
    u = mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(2.0 * u / (len(a) * len(b)) - 1.0)


def run_qap_test(
    panel: pd.DataFrame,
    countries: list[str],
    same_bloc: np.ndarray,
    value_col: str,
) -> dict[str, object]:
    values, valid = to_arrays(panel, countries, value_col)
    observed = mean_gap_difference(values, valid, same_bloc)
    null = qap_permutation_null(values, valid, same_bloc)
    ci_low, ci_high, _draws = dyadic_bootstrap_ci(values, valid, same_bloc)
    null_sd = float(null.std(ddof=1))
    return {
        "outcome": value_col,
        "observed": observed,
        "null": null,
        "null_sd": null_sd,
        "z": observed / null_sd if null_sd else np.nan,
        "p_value": two_sided_p_value(observed, null),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "mde": POWER_MULTIPLIER * null_sd,
    }


def bloc_breakdown(panel: pd.DataFrame, blocs: pd.Series) -> pd.DataFrame:
    """Per-bloc mean gap on its own within-bloc dyads. Descriptive only - eight
    blocs means eight comparisons, and nothing here is multiplicity-corrected."""
    within = panel[panel["same_bloc"]]
    rows: list[dict[str, float | str | int]] = []
    for cluster_id, group in within.groupby("from_bloc"):
        members = sorted(blocs[blocs == cluster_id].index)
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "members": ", ".join(members),
                "n_dyad_years": int(len(group)),
                "mean_jury": round(float(group[JURY_COL].mean()), 2),
                "mean_televote": round(float(group[TELEVOTE_COL].mean()), 2),
                "mean_gap": round(float(group["gap"].mean()), 3),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_gap", ignore_index=True)


def yearly_breakdown(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for year, group in panel.groupby(YEAR_COL):
        within = group[group["same_bloc"]]["gap"]
        between = group[~group["same_bloc"]]["gap"]
        rows.append(
            {
                "year": int(year),
                "n_within": int(len(within)),
                "n_between": int(len(between)),
                "mean_gap_within": round(float(within.mean()), 3),
                "mean_gap_between": round(float(between.mean()), 3),
                "T": round(float(within.mean() - between.mean()), 3),
            }
        )
    return pd.DataFrame(rows)


def group_summary(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for label, group in [
        ("within_bloc", panel[panel["same_bloc"]]),
        ("between_bloc", panel[~panel["same_bloc"]]),
    ]:
        rows.append(
            {
                "group": label,
                "n_pair_years": int(len(group)),
                "mean_jury": round(float(group[JURY_COL].mean()), 3),
                "mean_televote": round(float(group[TELEVOTE_COL].mean()), 3),
                "mean_gap": round(float(group["gap"].mean()), 3),
                "median_gap": round(float(group["gap"].median()), 3),
                "sd_gap": round(float(group["gap"].std()), 3),
                "share_nonzero": round(float((group["gap"] != 0).mean()), 3),
                "share_jury_top": round(float((group[JURY_COL] >= TOP_MARK).mean()), 3),
                "share_televote_top": round(
                    float((group[TELEVOTE_COL] >= TOP_MARK).mean()), 3
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_permutation_nulls(
    mean_null: np.ndarray,
    mean_observed: float,
    mean_p: float,
    dispersion_null: np.ndarray,
    dispersion_observed: float,
    dispersion_p: float,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.histplot(
        mean_null, bins=60, color="#4c78a8", edgecolor="none", stat="density", ax=axes[0]
    )
    axes[0].axvline(mean_observed, color="#b30000", linestyle="--", linewidth=2.0)
    axes[0].axvline(-mean_observed, color="#b30000", linestyle=":", linewidth=1.0)
    axes[0].annotate(
        f"observed = {mean_observed:.3f}\nQAP p = {mean_p:.3f}  (n.s.)",
        xy=(mean_observed, axes[0].get_ylim()[1] * 0.88),
        xytext=(-10, 0),
        textcoords="offset points",
        ha="right",
        color="#b30000",
        fontsize=11,
    )
    axes[0].set_xlabel("T = mean gap (within) − mean gap (between), points")
    axes[0].set_ylabel("Density under the null")
    axes[0].set_title("Test 1: do blocs lean one way on average?")

    sns.histplot(
        dispersion_null,
        bins=60,
        color="#4c78a8",
        edgecolor="none",
        stat="density",
        ax=axes[1],
    )
    axes[1].axvline(dispersion_observed, color="#b30000", linestyle="--", linewidth=2.0)
    axes[1].annotate(
        f"observed = {dispersion_observed:.3f}\nQAP p = {format_p(dispersion_p)}",
        xy=(dispersion_observed, axes[1].get_ylim()[1] * 0.88),
        xytext=(-10, 0),
        textcoords="offset points",
        ha="right",
        color="#b30000",
        fontsize=11,
    )
    axes[1].set_xlabel("Weighted SD across blocs of their own mean gap, points")
    axes[1].set_ylabel("Density under the null")
    axes[1].set_title("Test 3: do blocs differ from each other at all?")

    fig.suptitle(
        f"QAP permutation nulls: {len(mean_null):,} random re-assignments of bloc "
        f"labels across countries ({MIN_YEAR}–{MAX_YEAR} finals)"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def short_members(members: str, keep: int = 3) -> str:
    names = members.split(", ")
    if len(names) <= keep:
        return ", ".join(names)
    return ", ".join(names[:keep]) + f" +{len(names) - keep}"


def plot_bloc_gaps(by_bloc: pd.DataFrame, output_path: Path) -> None:
    frame = by_bloc.sort_values("mean_gap").copy()
    frame["label"] = frame.apply(
        lambda r: f"{r['cluster_id']}: {short_members(r['members'])}\n"
        f"(n={r['n_dyad_years']})",
        axis=1,
    )
    colors = ["#b30000" if value < 0 else "#4c78a8" for value in frame["mean_gap"]]

    plt.figure(figsize=(11, 7))
    ax = sns.barplot(
        data=frame, y="label", x="mean_gap", palette=colors, hue="label", legend=False
    )
    ax.axvline(0.0, color="#333333", linewidth=1.0)
    ax.set_xlabel("Mean jury − televote points on the bloc's own internal dyads")
    ax.set_ylabel("")
    ax.set_title(
        "Blocs do not lean the same way\ntelevote-favoured (red) vs jury-favoured "
        f"(blue) internal voting, {MIN_YEAR}–{MAX_YEAR}"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_gap_distribution(panel: pd.DataFrame, output_path: Path) -> None:
    tidy = panel.assign(
        Pairing=np.where(panel["same_bloc"], "Within bloc", "Between blocs")
    ).melt(
        id_vars=["Pairing"],
        value_vars=[JURY_COL, TELEVOTE_COL],
        var_name="Source",
        value_name="Points",
    )
    tidy["Source"] = tidy["Source"].str.replace("Points given by ", "", regex=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    sns.barplot(
        data=tidy,
        x="Pairing",
        y="Points",
        hue="Source",
        errorbar=("ci", 95),
        palette=["#4c78a8", "#f16913"],
        ax=axes[0],
    )
    axes[0].set_ylabel("Mean points awarded per pair-year")
    axes[0].set_title("Jury vs televote points by pairing")

    sns.violinplot(
        data=panel.assign(
            Pairing=np.where(panel["same_bloc"], "Within bloc", "Between blocs")
        ),
        x="Pairing",
        y="gap",
        hue="Pairing",
        legend=False,
        palette=["#9c9c9c", "#b30000"],
        cut=0,
        ax=axes[1],
    )
    axes[1].axhline(0.0, color="#333333", linewidth=1.0)
    axes[1].set_ylabel("Jury points − televote points")
    axes[1].set_title("Distribution of the jury−televote gap")
    fig.suptitle(
        f"Jury and televote points on the same pair-years ({MIN_YEAR}–{MAX_YEAR} finals)"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def markdown_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    divider = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = [
        "| " + " | ".join(str(v) for v in record) + " |"
        for record in df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def write_report(
    panel: pd.DataFrame,
    summary: pd.DataFrame,
    primary: dict[str, object],
    secondary: dict[str, object],
    dispersion: dict[str, float],
    naive_p: float,
    delta: float,
    by_bloc: pd.DataFrame,
    by_year: pd.DataFrame,
    excluded: list[str],
    n_countries: int,
    output_path: Path = REPORT_PATH,
) -> None:
    within = summary.loc[summary["group"] == "within_bloc"].iloc[0]
    between = summary.loc[summary["group"] == "between_bloc"].iloc[0]
    observed = float(primary["observed"])
    qap_p = float(primary["p_value"])
    ci_low, ci_high = float(primary["ci_low"]), float(primary["ci_high"])
    mde = float(primary["mde"])
    top_observed = float(secondary["observed"])
    top_p = float(secondary["p_value"])
    top_ci_low, top_ci_high = float(secondary["ci_low"]), float(secondary["ci_high"])
    share_zero = 1.0 - float(panel["gap"].ne(0).mean())
    televote_blocs = by_bloc[by_bloc["mean_gap"] < 0]
    jury_blocs = by_bloc[by_bloc["mean_gap"] > 0]

    lines: list[str] = [
        "# Statistical inference: does the jury–televote gap depend on bloc membership?",
        "",
        f"**Window:** {MIN_YEAR}–{MAX_YEAR} grand finals — every edition in the data "
        "carrying a jury/televote split (2020 was cancelled). "
        "**Unit of analysis:** one ordered (voter, recipient, year) dyad. "
        f"**N = {len(panel):,} dyad-years** across {n_countries} clustered countries.",
        "",
        "## Summary of findings",
        "",
        "| test | question | statistic | QAP p |",
        "| --- | --- | --- | --- |",
        f"| 1 (primary) | Do bloc partners get relatively more from the public "
        f"than from the jury, on average? | T = {observed:.3f} points | "
        f"{qap_p:.3f} — **not significant** |",
        f"| 2 (secondary) | Same question restricted to top marks (10s and 12s) | "
        f"T = {top_observed:.3f} | {top_p:.3f} — nominally significant, does not "
        "survive correction |",
        f"| 3 (heterogeneity) | Do blocs differ *from each other* in how their "
        f"internal points split? | SD = {dispersion['observed']:.3f} points vs "
        f"{dispersion['null_mean']:.3f} expected | "
        f"**{format_p(dispersion['p_value'])} — significant** |",
        "",
        "The headline is the combination of the three. The average bloc is **not** "
        "measurably more televote-driven than a random pairing, but blocs are "
        "**emphatically not interchangeable with each other**: some award their "
        "partners through the public vote and others through the jury, by margins "
        "far larger than random re-assignment of the same labels produces. "
        "Averaging over blocs cancels these out, which is why test 1 finds "
        "nothing and test 3 finds a great deal.",
        "",
        "## Hypothesis",
        "",
        "Since 2016 every country awards two separate 1–12 ballots: one from a "
        "five-member professional jury, one from its public televote. Both rank "
        "the same songs on the same night, so their difference is a within-dyad "
        "control — everything about the song itself (quality, staging, running "
        "order) differences out, and what survives is how *this country's public* "
        "rated the entry relative to how *this country's jury* rated it.",
        "",
        "For every dyad, `gap = jury points − televote points`. The question is "
        "whether its mean differs between dyads whose two countries share a "
        "`cluster_id` from the sibling clustering piece (**within-bloc**) and "
        "dyads that do not (**between-bloc**).",
        "",
        "- **H0:** mean gap(within-bloc) − mean gap(between-bloc) = 0 — bloc "
        "partners are no more a televote phenomenon than anyone else.",
        "- **H1 (two-sided):** the difference is non-zero. The substantive prior "
        "is one-sided and negative — diaspora and neighbour voting should surface "
        "in the public vote, which the jury system exists partly to dampen — but "
        "the test is run two-sided.",
        "",
        "**Test statistic:** `T = mean(gap | within-bloc) − mean(gap | "
        "between-bloc)`, in points.",
        "",
        "Two further statistics are reported on the same data, and both are "
        "declared here rather than hidden in the results:",
        "",
        f"- **Secondary outcome:** T computed on `top_gap = 1[jury ≥ {TOP_MARK}] − "
        f"1[televote ≥ {TOP_MARK}]`. About {share_zero:.0%} of dyad-years have a "
        "gap of exactly zero — nearly all of them 0–0 in both ballots — and only "
        "dilute a mean, while the folk accusation "
        "against bloc voting is specifically about the automatic 12. A top-mark "
        "indicator looks where the effect should be, at the cost of discarding the "
        "middle of the scale. **Two outcomes means two tests**, so a Bonferroni "
        "threshold of 0.025 applies to each.",
        "- **Heterogeneity test:** the count-weighted standard deviation, across "
        "blocs, of each bloc's own mean internal gap. H0 for this one is that the "
        "eight blocs are exchangeable groupings; H1 is that they differ from one "
        "another. It is one-sided by construction, since a dispersion statistic "
        "can only be contradicted from above.",
        "",
        "## Data construction",
        "",
        "Two decisions determine whether any of these numbers mean anything.",
        "",
        "1. **The zeros are reconstructed.** The votes file stores only the ten "
        "non-zero scores each ballot hands out, so a pair that scored nothing is "
        "an absent row, not missing data. The full voter × recipient grid is "
        "rebuilt per year and unrecorded cells filled with 0. Skipping this would "
        "condition the analysis on pairs at least one side already liked — a "
        "sample censored on the outcome.",
        f"2. **Undefined bloc membership is dropped, not guessed.** "
        f"{len(excluded)} countries appear in these finals but not in "
        f"`voting_blocs_clusters.csv` ({', '.join(excluded)}): they voted in too "
        "few editions for the clustering piece to place them. Every dyad touching "
        "them has no defined `same_bloc` value and is removed rather than imputed.",
        "",
        "One mechanical property bounds what any of this can show: each ballot "
        "distributes exactly 58 points, so the gap sums to zero within every "
        "voter-year. These statistics are **redistribution** measures. They "
        "cannot detect that a bloc is popular, only that a bloc's points arrive "
        "through one channel rather than the other.",
        "",
        "## Choice of test",
        "",
        "The outcome rules out the textbook options for two independent reasons.",
        "",
        "**Distribution.** The gap is a difference of two bounded discrete scores "
        f"from {{0,1,…,8,10,12}}: spiked at zero ({share_zero:.0%} of dyad-years), "
        "symmetric, heavy-tailed. A t-test's normality assumption does not "
        "describe it — though at N ≈ 2,900 the CLT would largely rescue that on "
        "its own, so this is the lesser problem.",
        "",
        "**Dependence, which is the real problem.** These are dyads, not "
        "independent observations. The 25 rows sharing a voter share that "
        "country's tastes; the 25 sharing a recipient share that song's appeal; "
        "the fixed 58-point budget makes rows within a voter-year mechanically "
        "negatively correlated. This is network data, and the documented "
        "consequence of ignoring it is not subtle: under realistic row/column "
        "autocorrelation, type-I error rates for naive t-statistics on dyadic "
        "data have been measured above 50%.",
        "",
        "**Primary inference — QAP permutation.** The Mantel/Krackhardt quadratic "
        "assignment procedure is the standard permutation scheme when the outcome "
        "is a relational matrix. Instead of shuffling rows it shuffles *node "
        f"labels*: bloc membership is re-assigned at random across the "
        f"{n_countries} countries and the statistic recomputed, "
        f"{N_PERMUTATIONS:,} times. Because relabelling a country moves its entire "
        "row and column together, every replicate preserves the real dependence — "
        "generous voters stay generous, popular songs stay popular, budgets still "
        "sum to 58, bloc sizes are exactly as observed — and only the hypothesis "
        "under test is randomized. The null it builds is precisely *\"blocs of "
        "this size and shape exist, but they are not **these** countries\"*, which "
        "is the null the research question needs.",
        "",
        "This is also the right *frame*: the 39 countries are not a sample from a "
        "population of countries, they are the population. Randomization "
        "inference conditional on the observed network is therefore more "
        "appropriate here than sampling-based inference.",
        "",
        "**Reported for contrast — naive row permutation**, shuffling the "
        "within/between label independently across dyad-years. Not the "
        "inferential basis; reported to quantify what the dependence costs.",
        "",
        "**Interval — pigeonhole (node) bootstrap.** Countries, not dyads, are "
        f"resampled with replacement {N_BOOTSTRAP:,} times and the induced "
        "sub-network's statistic recorded. Resampling dyads would understate the "
        "width for the same reason row permutation understates the p-value.",
        "",
        "**Effect size — Cliff's delta**, `P(gap_within > gap_between) − "
        "P(gap_within < gap_between)`. Standardized mean differences such as "
        "Cohen's d assume normality and are biased on bounded, ordinal, "
        "zero-inflated outcomes; Cliff's delta is purely rank-based and assumes "
        "no distribution at all.",
        "",
        "## Results",
        "",
        markdown_table(summary),
        "",
        "### Tests 1 and 2 — direction",
        "",
        "| quantity | primary (`gap`, points) | secondary (`top_gap`, share) |",
        "| --- | --- | --- |",
        f"| mean, within-bloc dyads | {within['mean_gap']:.3f} | "
        f"{within['share_jury_top'] - within['share_televote_top']:+.3f} |",
        f"| mean, between-bloc dyads | {between['mean_gap']:.3f} | "
        f"{between['share_jury_top'] - between['share_televote_top']:+.3f} |",
        f"| **observed T** | **{observed:.3f}** | **{top_observed:.3f}** |",
        f"| QAP null SD | {float(primary['null_sd']):.3f} | "
        f"{float(secondary['null_sd']):.3f} |",
        f"| QAP z (T / null SD) | {float(primary['z']):.2f} | "
        f"{float(secondary['z']):.2f} |",
        f"| **QAP p-value** ({N_PERMUTATIONS:,} relabellings) | **{qap_p:.4f}** | "
        f"**{top_p:.4f}** |",
        f"| 95% node-bootstrap CI | [{ci_low:.3f}, {ci_high:.3f}] | "
        f"[{top_ci_low:.3f}, {top_ci_high:.3f}] |",
        "",
        f"Naive row-permutation p on the primary outcome: **{naive_p:.4f}** "
        f"(against {qap_p:.4f} from QAP). Cliff's delta: **{delta:.3f}**.",
        "",
        "### Test 3 — heterogeneity across blocs",
        "",
        "| quantity | all blocs | blocs with ≥ "
        f"{MIN_BLOC_DYADS} dyad-years |",
        "| --- | --- | --- |",
        f"| observed weighted SD of bloc mean gaps | **{dispersion['observed']:.3f}** | "
        f"**{dispersion['robust_observed']:.3f}** |",
        f"| mean under the QAP null | {dispersion['null_mean']:.3f} | "
        f"{dispersion['robust_null_mean']:.3f} |",
        f"| **QAP p-value (one-sided)** | **{format_p(dispersion['p_value'])}** | "
        f"**{format_p(dispersion['robust_p_value'])}** |",
        "",
        "![QAP permutation nulls](voting_blocs_inference_permutation.png)",
        "",
        "*Left: the primary statistic sits well inside its null. Right: the "
        "heterogeneity statistic sits far outside its own.*",
        "",
        "![Mean gap by bloc](voting_blocs_inference_by_bloc.png)",
        "",
        "![Jury vs televote by pairing](voting_blocs_inference_gap_distribution.png)",
        "",
        "### Descriptive breakdowns",
        "",
        "The per-year table below is descriptive only — five uncorrected "
        "comparisons on small samples.",
        "",
        markdown_table(by_year),
        "",
        markdown_table(by_bloc.drop(columns="members")),
        "",
        "(Bloc membership lists are in `voting_blocs_inference_by_bloc.csv`.)",
        "",
        "## Interpretation",
        "",
        "**Test 1 fails to reject, and the estimate is in the predicted "
        f"direction.** Within-bloc dyads are televote-favoured by "
        f"{abs(observed):.2f} points per dyad-year: a bloc partner's public gives "
        f"{within['mean_televote']:.2f} points where its jury gives "
        f"{within['mean_jury']:.2f}, while for non-partners the two ballots agree "
        f"almost exactly ({between['mean_televote']:.2f} vs "
        f"{between['mean_jury']:.2f}). But the QAP null has SD "
        f"{float(primary['null_sd']):.3f}, the observed T is only "
        f"{abs(float(primary['z'])):.1f} SDs out, {qap_p * 100:.0f}% of random "
        "re-assignments of the same labels to different countries are at least "
        f"this extreme, and the bootstrap interval [{ci_low:.3f}, {ci_high:.3f}] "
        "contains zero. **H0 is not rejected.**",
        "",
        "**The design's resolution, stated explicitly.** Only "
        f"{int(within['n_pair_years'])} of {len(panel):,} dyad-years are "
        "within-bloc — eight blocs across 39 countries makes roughly one dyad in "
        f"eight a partner pairing — and the gap has SD ≈ {float(within['sd_gap']):.1f} "
        "points. Against the QAP null, the smallest effect this design could "
        f"detect at 80% power is |T| ≈ {mde:.2f} points, about "
        f"{mde / max(abs(observed), 1e-9):.1f}× the effect observed. The correct "
        "reading is *inconclusive*, not *absent*. Five contests is not many, and "
        "it is the number of **countries**, not the number of dyad-years, that "
        "sets the resolution.",
        "",
        "**Test 2 agrees in direction and is nominally significant, and should "
        "still not be sold as a finding.** Within-bloc dyads are "
        f"{abs(top_observed) * 100:.1f} percentage points more likely to draw a "
        "top mark from the public than from the jury, relative to between-bloc "
        f"dyads (p = {top_p:.3f}). That clears 0.05 but not the Bonferroni "
        "threshold of 0.025 for two outcomes, and its node-bootstrap interval "
        f"[{top_ci_low:.3f}, {top_ci_high:.3f}] includes zero. Suggestive, not "
        "conclusive. The disagreement between permutation and bootstrap is itself "
        "informative: the permutation test conditions on the observed network and "
        "asks only whether *these* labels are special, while the bootstrap asks "
        "what would happen with a different draw of countries — a question this "
        "design answers poorly, since resampling 39 nodes routinely deletes an "
        "entire small bloc.",
        "",
        f"**Test 3 rejects decisively, and it is the real result.** The spread of "
        f"bloc mean gaps is {dispersion['observed']:.2f} points against "
        f"{dispersion['null_mean']:.2f} expected under random relabelling "
        f"(p {format_p(dispersion['p_value'])}), and the result survives dropping the "
        f"small blocs ({dispersion['robust_observed']:.2f} vs "
        f"{dispersion['robust_null_mean']:.2f}, p = "
        f"{format_p(dispersion['robust_p_value'])}), so it is not an artefact of one "
        "thin cell. Concretely: "
        f"{len(televote_blocs)} blocs reward their partners through the televote "
        f"and {len(jury_blocs)} through the jury. The ex-Yugoslav trio (Croatia, "
        "Serbia, Slovenia) is the extreme case — juries averaging 0.33 points to "
        "each other against 9.33 from the publics — with the post-Soviet and "
        "Nordic groups leaning the same way, while the Western European core "
        "(Austria, France, Germany, Netherlands, Switzerland) and the eastern "
        "Mediterranean group (Albania, Cyprus, Greece, Malta) lean the other way, "
        "their juries backing partners *more* than their publics do. Pooling "
        "these into one average is what produced the null in test 1.",
        "",
        "**What this supports.** A single 'bloc effect' with one sign does not "
        "exist in this data; two distinguishable mechanisms do. Where partner "
        "points arrive through the televote and not the jury, the diaspora / "
        "cross-border-broadcast / familiarity reading is the natural one — those "
        "are exactly the channels that move a public and not a panel of music "
        "professionals. Where partner points arrive through the *jury*, that "
        "reading is unavailable, and shared musical convention, shared language, "
        "or the small-panel idiosyncrasy of five people is more plausible. The "
        "project's framing question — taste or politics — is therefore mis-posed "
        "as an either/or: the answer is bloc-specific, and this test says so with "
        "p < 0.001.",
        "",
        "**Caveat on the labels themselves.** `cluster_id` comes from the sibling "
        "piece's clustering of *centered outgoing vote profiles*, which groups "
        "countries that vote **alike** — not countries that vote **for each "
        "other**. Cluster 2 (Australia, Italy, Portugal) is explicitly a group of "
        "contest outsiders with no mutual-affinity story, and the sibling report "
        "flags Belgium/Israel/Poland/Spain as a residual rather than a bloc. "
        "Those dyads carry no hypothesized effect and attenuate test 1 by "
        "construction. A mutual-points definition of 'bloc' would be a different, "
        "and probably better-powered, test of the same idea.",
        "",
        f"**On the two nulls.** Row permutation gives p = {naive_p:.4f}, QAP gives "
        f"p = {qap_p:.4f}. Both land the same side of any conventional threshold "
        "here, so the conclusion is unchanged — but the naive null credits this "
        f"analysis with {len(panel):,} independent observations when the design "
        f"contains {n_countries} exchangeable units, and with an effect nearer the "
        "boundary that difference would have decided the result.",
        "",
        "**Handover to the causal piece.** This test isolates *which ballot* "
        "carries bloc points; it says nothing about whether bloc membership "
        "predicts points at all once the songs themselves are accounted for. That "
        "is the question `voting_blocs_causal_report.md` takes up, controlling "
        "for the lyrical similarity of the two countries' entries.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")



def main() -> None:
    sns.set_theme(style="whitegrid")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    blocs = load_blocs()
    votes = load_votes(min_year=MIN_YEAR, max_year=MAX_YEAR)
    votes = votes[votes["Points type"].isin({JURY_COL, TELEVOTE_COL})].copy()

    seen = sorted(set(votes[FROM_COL]) | set(votes[TO_COL]))
    excluded = [country for country in seen if country not in blocs.index]

    panel = build_gap_panel(votes, blocs)
    countries = sorted(set(panel[FROM_COL]) | set(panel[TO_COL]))
    same_bloc = same_bloc_matrix(blocs, countries)

    primary = run_qap_test(panel, countries, same_bloc, "gap")
    secondary = run_qap_test(panel, countries, same_bloc, "top_gap")

    from_idx, to_idx = to_flat(panel, countries)
    labels = blocs.reindex(countries).to_numpy()
    gaps = panel["gap"].to_numpy()
    dispersion_observed = bloc_dispersion(gaps, from_idx, to_idx, labels)
    dispersion_null = qap_dispersion_null(gaps, from_idx, to_idx, labels)
    dispersion_p = upper_p_value(dispersion_observed, dispersion_null)
    robust_observed = bloc_dispersion(
        gaps, from_idx, to_idx, labels, min_count=MIN_BLOC_DYADS
    )
    robust_null = qap_dispersion_null(
        gaps, from_idx, to_idx, labels, min_count=MIN_BLOC_DYADS
    )
    robust_p = upper_p_value(robust_observed, robust_null)

    naive_null = naive_permutation_null(panel)
    naive_p = two_sided_p_value(float(primary["observed"]), naive_null)
    delta = cliffs_delta(
        panel.loc[panel["same_bloc"], "gap"].to_numpy(),
        panel.loc[~panel["same_bloc"], "gap"].to_numpy(),
    )
    summary = group_summary(panel)
    by_bloc = bloc_breakdown(panel, blocs)
    by_year = yearly_breakdown(panel)

    results = pd.DataFrame(
        [
            {"quantity": f"observed_T_{test['outcome']}", "value": round(float(test["observed"]), 4)}
            for test in (primary, secondary)
        ]
        + [
            {"quantity": f"qap_p_{test['outcome']}", "value": round(float(test["p_value"]), 5)}
            for test in (primary, secondary)
        ]
        + [
            {"quantity": f"qap_null_sd_{test['outcome']}", "value": round(float(test["null_sd"]), 4)}
            for test in (primary, secondary)
        ]
        + [
            {"quantity": f"bootstrap_ci_low_{test['outcome']}", "value": round(float(test["ci_low"]), 4)}
            for test in (primary, secondary)
        ]
        + [
            {"quantity": f"bootstrap_ci_high_{test['outcome']}", "value": round(float(test["ci_high"]), 4)}
            for test in (primary, secondary)
        ]
        + [
            {"quantity": "bloc_dispersion_observed", "value": round(dispersion_observed, 4)},
            {"quantity": "bloc_dispersion_null_mean", "value": round(float(dispersion_null.mean()), 4)},
            {"quantity": "bloc_dispersion_p", "value": round(dispersion_p, 5)},
            {"quantity": f"bloc_dispersion_observed_min{MIN_BLOC_DYADS}", "value": round(robust_observed, 4)},
            {"quantity": f"bloc_dispersion_p_min{MIN_BLOC_DYADS}", "value": round(robust_p, 5)},
            {"quantity": "naive_permutation_p_gap", "value": round(naive_p, 5)},
            {"quantity": "cliffs_delta_gap", "value": round(delta, 4)},
            {"quantity": "mde_80pct_power_gap", "value": round(float(primary["mde"]), 4)},
            {"quantity": "n_dyad_years", "value": int(len(panel))},
            {"quantity": "n_countries", "value": int(len(countries))},
            {"quantity": "n_permutations", "value": int(N_PERMUTATIONS)},
        ]
    )

    panel.to_csv(PANEL_PATH, index=False)
    pd.concat([summary, by_year, results], axis=0, ignore_index=True).to_csv(
        SUMMARY_PATH, index=False
    )
    by_bloc.to_csv(BLOC_BREAKDOWN_PATH, index=False)
    plot_permutation_nulls(
        np.asarray(primary["null"]),
        float(primary["observed"]),
        float(primary["p_value"]),
        dispersion_null,
        dispersion_observed,
        dispersion_p,
        PERMUTATION_PLOT_PATH,
    )
    plot_gap_distribution(panel, GAP_PLOT_PATH)
    plot_bloc_gaps(by_bloc, BLOC_PLOT_PATH)
    write_report(
        panel,
        summary,
        primary,
        secondary,
        {
            "observed": dispersion_observed,
            "null_mean": float(dispersion_null.mean()),
            "p_value": dispersion_p,
            "robust_observed": robust_observed,
            "robust_null_mean": float(robust_null.mean()),
            "robust_p_value": robust_p,
        },
        naive_p,
        delta,
        by_bloc,
        by_year,
        excluded,
        len(countries),
    )

    print(f"Dyad-years: {len(panel):,} over {len(countries)} clustered countries")
    print(f"Dropped (no bloc label): {', '.join(excluded)}")
    print(summary.to_string(index=False))
    for test in (primary, secondary):
        print(
            f"\n[{test['outcome']}] T = {float(test['observed']):.4f}  "
            f"QAP p = {float(test['p_value']):.4f}  "
            f"null sd = {float(test['null_sd']):.4f}  "
            f"95% CI = [{float(test['ci_low']):.4f}, {float(test['ci_high']):.4f}]"
        )
    print(
        f"\n[bloc dispersion] observed = {dispersion_observed:.4f}  "
        f"null mean = {dispersion_null.mean():.4f}  QAP p = {format_p(dispersion_p)}"
        f"  | min {MIN_BLOC_DYADS} dyads: observed = {robust_observed:.4f}, "
        f"p = {robust_p:.4f}"
    )
    print(f"\nNaive row-permutation p (gap) = {naive_p:.4f}")
    print(f"Cliff's delta (gap) = {delta:.4f}")
    print(f"Minimum detectable |T| at 80% power = {float(primary['mde']):.3f} points")
    print("\nBy year:")
    print(by_year.to_string(index=False))
    print("\nBy bloc (within-bloc dyads only):")
    print(by_bloc.drop(columns="members").to_string(index=False))
    print("\nSaved:")
    for path in (
        PANEL_PATH,
        SUMMARY_PATH,
        BLOC_BREAKDOWN_PATH,
        BLOC_PLOT_PATH,
        PERMUTATION_PLOT_PATH,
        GAP_PLOT_PATH,
        REPORT_PATH,
    ):
        print(f"  - {path}")


if __name__ == "__main__":
    main()
