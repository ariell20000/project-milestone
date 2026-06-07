from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Short codes used inside cells to keep annotations compact
COUNTRY_ABBREV = {
    "Israel": "ISR",
    "Serbia": "SRB",
    "France": "FRA",
    "Germany": "GER",
    "United Kingdom": "UK",
    "Malta": "MLT",
    "Switzerland": "SUI",
    "Sweden": "SWE",
    "Ukraine": "UKR",
    "Greece": "GRE",
    "Cyprus": "CYP",
    "Slovenia": "SVN",
}


def _abbreviate(name: str) -> str:
    """Return a short country code for in-cell annotations."""
    return COUNTRY_ABBREV.get(name, name[:3].upper())


def _spectral_order(matrix: pd.DataFrame) -> list[str]:
    """Reorder countries via the Fiedler vector so strongly-linked pairs sit together."""
    M = matrix.values
    # Treat absolute net points as a similarity measure (symmetric)
    similarity = np.abs(M) + np.abs(M.T)
    np.fill_diagonal(similarity, 0)

    # Graph Laplacian  L = D − W
    D = np.diag(similarity.sum(axis=1))
    L = D - similarity

    eigenvalues, eigenvectors = np.linalg.eigh(L)
    # The Fiedler vector (2nd-smallest eigenvalue) gives a 1-D embedding
    fiedler = eigenvectors[:, 1]
    order = np.argsort(fiedler)

    return [matrix.index[i] for i in order]


def build_relations_matrix(df: pd.DataFrame, countries: list[str]) -> pd.DataFrame:
    yearly = (
        df.groupby(["Year", "From", "To"])["Points"]
        .sum()
        .unstack("To", fill_value=0)
        .reindex(columns=countries, fill_value=0)
    )

    yearly_nets = []
    for _, group in yearly.groupby(level="Year"):
        mat = group.droplevel("Year").reindex(index=countries, fill_value=0)
        yearly_nets.append(mat - mat.T)

    if not yearly_nets:
        return pd.DataFrame(0.0, index=countries, columns=countries)

    return sum(yearly_nets) / len(yearly_nets)


def plot_relations_matrix(avg_net: pd.DataFrame, output_path: Path, min_year: int) -> None:
    # ── 1. Reorder countries so related pairs are adjacent ────────────
    ordered = _spectral_order(avg_net)
    avg_net = avg_net.loc[ordered, ordered]

    n = len(avg_net)
    countries = avg_net.index.tolist()

    # ── 2. Upper-triangle mask (hide diagonal + lower half) ──────────
    mask = np.tril(np.ones((n, n), dtype=bool))

    # ── 3. Symmetric colour range (clip at 95th percentile) ──────────
    upper_vals = avg_net.where(~mask).stack()
    abs_max = float(upper_vals.abs().quantile(0.95))
    if abs_max == 0:
        abs_max = float(upper_vals.abs().max())
    if abs_max == 0:
        abs_max = 1.0

    # ── 4. Figure & magnitude colour map ───────────────────────────
    fig, ax = plt.subplots(figsize=(32, 22))

    # Colour shows magnitude only (absolute value) — direction is
    # conveyed by the GIVER → RECEIVER arrow text inside each cell.
    avg_abs = avg_net.abs()
    cmap = "YlOrRd"

    sns.heatmap(
        avg_abs,
        mask=mask,
        cmap=cmap,
        vmin=0,
        vmax=abs_max,
        annot=False,
        linewidths=3.5,
        linecolor="white",
        square=True,
        cbar_kws={"shrink": 0.55, "aspect": 25, "pad": 0.02},
        ax=ax,
    )

    # Colorbar styling
    cbar = ax.collections[0].colorbar
    cbar.set_label("Avg Yearly |Net Points|", fontsize=22, fontweight="bold", labelpad=18)
    cbar.ax.tick_params(labelsize=17)

    # ── 5. Custom cell annotations: "GIVER → RECEIVER  value" ────────
    cmap_obj = plt.colormaps["YlOrRd"]
    norm = plt.Normalize(vmin=0, vmax=abs_max)

    for i in range(n):
        for j in range(i + 1, n):
            val = avg_net.iloc[i, j]
            abs_val = abs(val)

            # Near-zero: show a muted label
            if abs_val < 0.05:
                ax.text(j + 0.5, i + 0.5, "≈ 0",
                        ha="center", va="center", fontsize=20, color="#999999")
                continue

            # Determine who gives more
            if val > 0:
                # Row country favors column country
                giver = _abbreviate(countries[i])
                receiver = _abbreviate(countries[j])
            else:
                # Column country favors row country
                giver = _abbreviate(countries[j])
                receiver = _abbreviate(countries[i])

            direction_label = f"{giver} → {receiver}"
            value_label = f"{abs_val:.1f}"

            # Pick text colour for contrast against the cell background
            rgba = cmap_obj(norm(abs_val))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            text_color = "white" if luminance < 0.55 else "#1a1a1a"

            # Country codes & Arrow — split up so the arrow can be larger
            # Giver
            ax.text(
                j + 0.38, i + 0.35, giver,
                ha="right", va="center",
                fontsize=14, fontweight="bold",
                color=text_color,
            )
            # Big bold standard arrow (the heavier unicode caused font rendering issues)
            ax.text(
                j + 0.5, i + 0.35, "→",
                ha="center", va="center",
                fontsize=24, fontweight="bold",
                color=text_color,
            )
            # Receiver
            ax.text(
                j + 0.62, i + 0.35, receiver,
                ha="left", va="center",
                fontsize=14, fontweight="bold",
                color=text_color,
            )
            # Value — much bigger, below center (the star of the show)
            ax.text(
                j + 0.5, i + 0.68, value_label,
                ha="center", va="center",
                fontsize=22, fontweight="bold",
                color=text_color,
            )

    # ── 6. Readable axis labels ──────────────────────────────────────
    def wrap_label(name: str) -> str:
        if name == "United Kingdom": return "United\nKingdom"
        if name == "Switzerland": return "Switzer-\nland"
        if name == "Germany": return "Ger-\nmany"
        if name == "Slovenia": return "Slove-\nnia"
        return name
        
    wrapped_countries = [wrap_label(c) for c in countries]
    ax.set_xticklabels(wrapped_countries, rotation=0, ha="center",
                       fontsize=22, fontweight="bold")
    ax.set_yticklabels(countries, rotation=0, va="center",
                       fontsize=22, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")

    # ── 7. Large, descriptive title ──────────────────────────────────
    ax.set_title(
        "Eurovision Voting Relations\n"
        f"Average Yearly Net Points Between Selected Countries (Since {min_year})",
        fontsize=32, fontweight="bold", pad=40, linespacing=1.6,
    )

    # ── 8. Explanatory legend box below the chart ────────────────────
    legend_text = (
        'Each cell reads  "GIVER → RECEIVER"  —  the country that gives more net points to the other.\n'
        "Darker color = stronger voting bias between the pair."
    )
    fig.text(
        0.5, 0.005, legend_text,
        ha="center", va="bottom", fontsize=18,
        style="italic", color="#333333",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#FFFDE7",
                  edgecolor="#CCCCCC", alpha=0.9),
    )

    plt.tight_layout(rect=[0, 0.07, 0.93, 1])
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

