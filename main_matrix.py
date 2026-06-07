"""Generate the net-points relations heatmap (matrix) between selected countries."""
from __future__ import annotations

from pathlib import Path

from data_loader import MIN_YEAR, prepare_data, get_target_countries
from relations_matrix import build_relations_matrix, plot_relations_matrix


def main() -> None:
    df = prepare_data()
    target_countries = get_target_countries()

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    df_matrix = df[
        df["From"].isin(target_countries) & df["To"].isin(target_countries)
    ].copy()
    net = build_relations_matrix(df_matrix, target_countries)
    plot_relations_matrix(net, output_dir / "relations_matrix.png", MIN_YEAR)

    print(f"Saved: {output_dir / 'relations_matrix.png'}")


if __name__ == "__main__":
    main()
