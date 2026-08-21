#!/usr/bin/env python3
"""Single entrypoint that regenerates every table, report and figure in this repo.

Run with: python3 run_pipeline.py

Skips theme_clustering/bert model/generate_themes.py, the BERT lyric-theme classifier, since it is
slow (one zero-shot pass per song). This pipeline uses the already-committed
theme_clustering/bert model/theme_predictions.csv instead - see README.md for how to regenerate it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

VOTING_BLOCS_DIR = ROOT / "voting_blocs"
GRAPHS_DIR = ROOT / "theme_clustering"

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
    "theme_pie_comparison.py",
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
    print("Starting pipeline...")

    for name in VOTING_BLOCS_SCRIPTS:
        run(VOTING_BLOCS_DIR / name)


    for name in GRAPH_SCRIPTS:
        run(GRAPHS_DIR / name, optional=(name == "composer.py"))

    print(
        "\nDone. Refreshed voting_blocs/outputs/ and graph_outputs/. "
        "paper/figures2/ is not touched automatically - copy any changed "
        "figure over manually if the paper needs updating."
    )


if __name__ == "__main__":
    main()
