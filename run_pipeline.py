#!/usr/bin/env python3
"""Single entrypoint that regenerates every table, report and figure in this repo.

Run with: python3 run_pipeline.py

Skips model/generate_themes.py, the BERT lyric-theme classifier, since it is
slow (one zero-shot pass per song). This pipeline uses the already-committed
model/theme_predictions.csv instead - see README.md for how to regenerate it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

VOTING_BLOCS_DIR = ROOT / "voting_blocs"
GRAPHS_DIR = ROOT / "graphs"
MODEL_DIR = ROOT / "model"

# Order matters here: each later script reads an earlier one's output
# (clustering needs similarity's matrix, causal needs inference's helpers, etc).
VOTING_BLOCS_SCRIPTS = [
    "voting_blocs_similarity.py",
    "voting_blocs_clustering.py",
    "voting_blocs_graph.py",
    "voting_blocs_inference.py",
    "voting_blocs_causal.py",
    "voting_blocs_recsys.py",
    "voting_blocs_method_comparison.py",
]

# These are independent of each other - order doesn't matter.
GRAPH_SCRIPTS = [
    "composer.py",  # needs internet: pulls a CSV from raw.githubusercontent.com
    "friendship_clusters.py",
    "jury_vs_public_bias.py",
    "kingmaker_scatter.py",
    "tastmaker.py",
    "theme_clusters.py",
    "theme_pie_comparison.py",
    "winner_bias_timeline.py",
    "winners_by_language.py",
    "wordcloud_winners_vs_losers.py",
]


def run(script: Path, *, optional: bool = False) -> None:
    print(f"\n=== {script.relative_to(ROOT)} ===", flush=True)
    result = subprocess.run([PYTHON, str(script)])
    if result.returncode != 0:
        message = f"{script.relative_to(ROOT)} exited with code {result.returncode}"
        if optional:
            print(f"WARNING: {message} - continuing without it (needs internet access)")
        else:
            sys.exit(f"FAILED: {message}")


def main() -> None:
    print(
        "Skipping model/generate_themes.py (BERT lyric-theme classifier - slow).\n"
        "Using the committed model/theme_predictions.csv instead.\n"
        "See README.md if you need to regenerate it from scratch."
    )

    for name in VOTING_BLOCS_SCRIPTS:
        run(VOTING_BLOCS_DIR / name)

    run(MODEL_DIR / "update_theme_predictions.py")

    for name in GRAPH_SCRIPTS:
        run(GRAPHS_DIR / name, optional=(name == "composer.py"))

    print(
        "\nDone. Refreshed voting_blocs/outputs/ and output/. "
        "paper/figures2/ is not touched automatically - copy any changed "
        "figure over manually if the paper needs updating."
    )


if __name__ == "__main__":
    main()
