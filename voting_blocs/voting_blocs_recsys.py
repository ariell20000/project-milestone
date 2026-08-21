"""Implicit-feedback recommender over Eurovision ballots.

Frames country-to-country voting as a recommendation problem at the
(voter country, competing entry) grain, where an "entry" is a specific
(Year, To) song rather than a country, so that the model has to predict who a
voter rewards *this year* instead of re-deriving a static country similarity
matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

from main import clean_data, load_data

BASE_DIR = Path(__file__).resolve().parent
VOTES_PATH = BASE_DIR.parent / "dataset" / "eurovision_1957-2021.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
METRICS_PATH = OUTPUT_DIR / "voting_blocs_recsys_metrics.csv"
EXAMPLES_PATH = OUTPUT_DIR / "voting_blocs_recsys_examples.csv"
TUNING_PATH = OUTPUT_DIR / "voting_blocs_recsys_tuning.csv"
SIGNIFICANCE_PATH = OUTPUT_DIR / "voting_blocs_recsys_significance.csv"

MIN_YEAR = 2004
MAX_YEAR = 2021
MAX_POINTS = 12.0

FROM_COL = "From"
TO_COL = "To"
TYPE_COL = "Points type"
POINTS_COL = "Points"
ENTRY_COL = "Entry"
BALLOT_COL = "Ballot"
VOTER_YEAR_COL = "VoterYear"

TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
RANDOM_STATE = 42
TOP_K = 5
MIN_EXAMPLE_OPPORTUNITIES = 5
REPEAT_SEEDS = (42, 7, 13, 2021, 99)
MF_GRID = {"n_factors": (2, 4, 8, 16), "reg": (1.0, 5.0, 20.0, 50.0, 100.0)}
IMPLICIT_GRID = {
    "n_factors": (4, 8, 16),
    "reg": (1.0, 5.0, 20.0),
    "alpha": (0.5, 1.0, 4.0),
}
PAIRED_COMPARISONS = [
    ("matrix_factorization_als", "baseline_voter_mean"),
    ("matrix_factorization_als", "bias_only_entry_quality"),
    ("neighborhood_cf", "bias_only_entry_quality"),
    ("implicit_als_hkv", "bias_only_entry_quality"),
]

# Countries picked for the demo table: two known bloc pairs (Cyprus/Greece,
# Romania/Moldova), one post-Soviet voter, one Nordic voter and two large
# Western voters with no obvious bloc.
EXAMPLE_COUNTRIES = [
    "Cyprus",
    "Greece",
    "Romania",
    "Moldova",
    "Russia",
    "Sweden",
    "United Kingdom",
    "Israel",
]


def harmonize_country(name: str) -> str:
    # The votes file spells voters with hyphens ("The-Netherlands") and
    # recipients with spaces ("The Netherlands"); voter and recipient identities
    # have to live in one namespace or a country cannot be matched to its entry.
    return name.replace("-", " ")


def load_votes(votes_path: Path = VOTES_PATH) -> pd.DataFrame:
    votes = clean_data(load_data(votes_path))
    votes = votes[votes["Year"].between(MIN_YEAR, MAX_YEAR)].copy()
    votes["Year"] = votes["Year"].astype(int)
    votes[FROM_COL] = votes[FROM_COL].map(harmonize_country)
    votes[TO_COL] = votes[TO_COL].map(harmonize_country)
    return votes.reset_index(drop=True)


def build_interactions(votes: pd.DataFrame) -> pd.DataFrame:
    """Expand the awarded points into the full ballot x candidate grid.

    A Eurovision ballot only reports its ten favourite entries, so an absent
    (From, To, Year) triple means zero points - but only for voters that
    actually cast a ballot in that year and round. Rounds differ (2016+ splits
    each country into a jury ballot and a televote ballot, and the two rounds do
    not always cover the same recipients), so voters and candidates are taken
    per (Year, Points type) group rather than per year.
    """
    frames: list[pd.DataFrame] = []
    for (year, points_type), group in votes.groupby(["Year", TYPE_COL], sort=True):
        voters = np.sort(group[FROM_COL].unique())
        candidates = np.sort(group[TO_COL].unique())
        grid = pd.MultiIndex.from_product(
            [voters, candidates], names=[FROM_COL, TO_COL]
        ).to_frame(index=False)
        grid = grid[grid[FROM_COL] != grid[TO_COL]].copy()
        grid["Year"] = year
        grid[TYPE_COL] = points_type

        awarded = group.groupby([FROM_COL, TO_COL], as_index=False)[POINTS_COL].sum()
        grid = grid.merge(awarded, on=[FROM_COL, TO_COL], how="left")
        grid[POINTS_COL] = grid[POINTS_COL].fillna(0.0)
        frames.append(grid)

    cells = pd.concat(frames, ignore_index=True)
    cells[ENTRY_COL] = cells["Year"].astype(str) + " " + cells[TO_COL]
    cells[BALLOT_COL] = (
        cells["Year"].astype(str) + " " + cells[TYPE_COL] + " " + cells[FROM_COL]
    )
    cells[VOTER_YEAR_COL] = cells["Year"].astype(str) + " " + cells[FROM_COL]
    return cells


@dataclass
class Encoder:
    """Maps voter countries and entries onto contiguous matrix indices."""

    voters: list[str]
    entries: list[str]
    voter_index: dict[str, int] = field(init=False)
    entry_index: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.voter_index = {name: i for i, name in enumerate(self.voters)}
        self.entry_index = {name: i for i, name in enumerate(self.entries)}

    @property
    def n_voters(self) -> int:
        return len(self.voters)

    @property
    def n_entries(self) -> int:
        return len(self.entries)

    def encode(self, cells: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        rows = cells[FROM_COL].map(self.voter_index).to_numpy(dtype=float)
        cols = cells[ENTRY_COL].map(self.entry_index).to_numpy(dtype=float)
        return rows, cols


def make_encoder(cells: pd.DataFrame) -> Encoder:
    return Encoder(
        voters=sorted(cells[FROM_COL].unique()),
        entries=sorted(cells[ENTRY_COL].unique()),
    )


def split_by_voter_year(
    cells: pd.DataFrame, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out whole (From, Year) voter-years.

    Splitting single cells would leak: the other nine points of the same ballot
    (and, in 2016+, the same country's other round) pin down what is left.
    """
    groups = np.sort(cells[VOTER_YEAR_COL].unique())
    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(groups)
    n_test = max(1, int(round(len(groups) * test_size)))
    test_groups = set(shuffled[:n_test])
    is_test = cells[VOTER_YEAR_COL].isin(test_groups)
    return cells[~is_test].copy(), cells[is_test].copy()


def aggregate_cells(cells: pd.DataFrame) -> pd.DataFrame:
    """Collapse the jury and televote rounds of one voter-year into one cell.

    Pre-2016 a country casts a single combined ballot, so averaging the two
    rounds keeps the feedback on one comparable 0-12 scale across the window.
    """
    return cells.groupby([FROM_COL, ENTRY_COL], as_index=False)[POINTS_COL].mean()


def fit_biases(
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
    encoder: Encoder,
    reg_voter: float = 10.0,
    reg_entry: float = 5.0,
    n_iter: int = 5,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Damped alternating means: mu + voter bias + entry bias."""
    mu = float(values.mean())
    voter_bias = np.zeros(encoder.n_voters)
    entry_bias = np.zeros(encoder.n_entries)
    voter_counts = np.bincount(rows, minlength=encoder.n_voters)
    entry_counts = np.bincount(cols, minlength=encoder.n_entries)

    for _ in range(n_iter):
        resid = values - mu - voter_bias[rows]
        entry_bias = np.bincount(cols, weights=resid, minlength=encoder.n_entries) / (
            entry_counts + reg_entry
        )
        resid = values - mu - entry_bias[cols]
        voter_bias = np.bincount(rows, weights=resid, minlength=encoder.n_voters) / (
            voter_counts + reg_voter
        )
    return mu, voter_bias, entry_bias


def lookup(table: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Table lookup that scores unknown entities (NaN keys) as zero."""
    return np.where(np.isnan(keys), 0.0, table[np.nan_to_num(keys).astype(int)])


def _group_positions(keys: np.ndarray, n_groups: int) -> list[np.ndarray]:
    order = np.argsort(keys, kind="stable")
    counts = np.bincount(keys, minlength=n_groups)
    return np.split(order, np.cumsum(counts)[:-1])


def weighted_als(
    rows: np.ndarray,
    cols: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    n_voters: int,
    n_entries: int,
    n_factors: int,
    reg: float,
    n_iter: int = 15,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Alternating least squares over the observed cells only.

    Only cells in the ballot grid enter the loss, so entries a voter never had
    the chance to reward contribute nothing (unlike a plain SVD of a
    zero-filled matrix, which would score those as genuine zero preferences).
    """
    rng = np.random.default_rng(random_state)
    voter_factors = rng.normal(0.0, 0.05, size=(n_voters, n_factors))
    entry_factors = rng.normal(0.0, 0.05, size=(n_entries, n_factors))
    voter_groups = _group_positions(rows, n_voters)
    entry_groups = _group_positions(cols, n_entries)
    eye = np.eye(n_factors)

    def solve_side(
        groups: list[np.ndarray], other: np.ndarray, other_keys: np.ndarray
    ) -> np.ndarray:
        factors = np.zeros((len(groups), n_factors))
        for idx, positions in enumerate(groups):
            if positions.size == 0:
                continue
            design = other[other_keys[positions]]
            w = weights[positions]
            gram = design.T @ (design * w[:, None]) + reg * eye
            rhs = design.T @ (w * targets[positions])
            factors[idx] = np.linalg.solve(gram, rhs)
        return factors

    for _ in range(n_iter):
        voter_factors = solve_side(voter_groups, entry_factors, cols)
        entry_factors = solve_side(entry_groups, voter_factors, rows)
    return voter_factors, entry_factors


class GlobalMeanRecommender:
    """Predicts the average points awarded per candidate."""

    def fit(self, train: pd.DataFrame, encoder: Encoder) -> "GlobalMeanRecommender":
        self.mu_ = float(train[POINTS_COL].mean())
        return self

    def predict(self, cells: pd.DataFrame) -> np.ndarray:
        return np.full(len(cells), self.mu_)


class VoterMeanRecommender:
    """Naive baseline: each voter's historical average points given."""

    def fit(self, train: pd.DataFrame, encoder: Encoder) -> "VoterMeanRecommender":
        self.mu_ = float(train[POINTS_COL].mean())
        self.voter_mean_ = train.groupby(FROM_COL)[POINTS_COL].mean()
        return self

    def predict(self, cells: pd.DataFrame) -> np.ndarray:
        return cells[FROM_COL].map(self.voter_mean_).fillna(self.mu_).to_numpy()


class BiasRecommender:
    """Global mean plus voter and entry biases - "how good was the song"."""

    def __init__(self, reg_voter: float = 10.0, reg_entry: float = 5.0) -> None:
        self.reg_voter = reg_voter
        self.reg_entry = reg_entry

    def fit(self, train: pd.DataFrame, encoder: Encoder) -> "BiasRecommender":
        self.encoder_ = encoder
        agg = aggregate_cells(train)
        rows, cols = encoder.encode(agg)
        self.mu_, self.voter_bias_, self.entry_bias_ = fit_biases(
            rows.astype(int),
            cols.astype(int),
            agg[POINTS_COL].to_numpy(dtype=float),
            encoder,
            reg_voter=self.reg_voter,
            reg_entry=self.reg_entry,
        )
        return self

    def _bias_prediction(self, cells: pd.DataFrame) -> np.ndarray:
        rows, cols = self.encoder_.encode(cells)
        return self.mu_ + lookup(self.voter_bias_, rows) + lookup(self.entry_bias_, cols)

    def predict(self, cells: pd.DataFrame) -> np.ndarray:
        return np.clip(self._bias_prediction(cells), 0.0, MAX_POINTS)


class MatrixFactorizationRecommender(BiasRecommender):
    """Biased matrix factorization fitted by ALS on the observed ballot grid."""

    def __init__(
        self,
        n_factors: int = 4,
        reg: float = 5.0,
        n_iter: int = 15,
        reg_voter: float = 10.0,
        reg_entry: float = 5.0,
        random_state: int = RANDOM_STATE,
    ) -> None:
        super().__init__(reg_voter=reg_voter, reg_entry=reg_entry)
        self.n_factors = n_factors
        self.reg = reg
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, train: pd.DataFrame, encoder: Encoder) -> "MatrixFactorizationRecommender":
        super().fit(train, encoder)
        agg = aggregate_cells(train)
        rows, cols = encoder.encode(agg)
        rows = rows.astype(int)
        cols = cols.astype(int)
        residual = (
            agg[POINTS_COL].to_numpy(dtype=float)
            - self.mu_
            - self.voter_bias_[rows]
            - self.entry_bias_[cols]
        )
        self.voter_factors_, self.entry_factors_ = weighted_als(
            rows,
            cols,
            residual,
            np.ones_like(residual),
            encoder.n_voters,
            encoder.n_entries,
            n_factors=self.n_factors,
            reg=self.reg,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        return self

    def predict(self, cells: pd.DataFrame) -> np.ndarray:
        rows, cols = self.encoder_.encode(cells)
        known = ~(np.isnan(rows) | np.isnan(cols))
        interaction = np.zeros(len(cells))
        r = np.nan_to_num(rows).astype(int)
        c = np.nan_to_num(cols).astype(int)
        interaction[known] = np.einsum(
            "ij,ij->i", self.voter_factors_[r[known]], self.entry_factors_[c[known]]
        )
        return np.clip(self._bias_prediction(cells) + interaction, 0.0, MAX_POINTS)


class ImplicitALSRecommender:
    """Hu-Koren-Volinsky implicit ALS: binary preference, confidence = 1 + alpha * points.

    Points are treated as a confidence signal rather than a rating, which is the
    standard treatment for implicit count feedback. The resulting score is a
    ranking quantity, so a linear calibration fitted on the training cells maps
    it back onto the 0-12 points scale before RMSE is computed.
    """

    def __init__(
        self,
        n_factors: int = 8,
        reg: float = 5.0,
        alpha: float = 1.0,
        n_iter: int = 15,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.n_factors = n_factors
        self.reg = reg
        self.alpha = alpha
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, train: pd.DataFrame, encoder: Encoder) -> "ImplicitALSRecommender":
        self.encoder_ = encoder
        agg = aggregate_cells(train)
        rows, cols = encoder.encode(agg)
        rows = rows.astype(int)
        cols = cols.astype(int)
        points = agg[POINTS_COL].to_numpy(dtype=float)
        preference = (points > 0).astype(float)
        confidence = 1.0 + self.alpha * points
        self.voter_factors_, self.entry_factors_ = weighted_als(
            rows,
            cols,
            preference,
            confidence,
            encoder.n_voters,
            encoder.n_entries,
            n_factors=self.n_factors,
            reg=self.reg,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        raw = self._raw_score(rows, cols)
        self.calibration_ = LinearRegression().fit(raw.reshape(-1, 1), points)
        return self

    def _raw_score(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        return np.einsum("ij,ij->i", self.voter_factors_[rows], self.entry_factors_[cols])

    def predict(self, cells: pd.DataFrame) -> np.ndarray:
        rows, cols = self.encoder_.encode(cells)
        known = ~(np.isnan(rows) | np.isnan(cols))
        raw = np.zeros(len(cells))
        r = np.nan_to_num(rows).astype(int)
        c = np.nan_to_num(cols).astype(int)
        raw[known] = self._raw_score(r[known], c[known])
        return np.clip(self.calibration_.predict(raw.reshape(-1, 1)), 0.0, MAX_POINTS)


class VoterNeighborhoodRecommender(BiasRecommender):
    """User-based CF: voters similar to you rewarded this entry."""

    def __init__(
        self,
        n_neighbors: int = 10,
        shrinkage: float = 25.0,
        min_overlap: int = 30,
        reg_voter: float = 10.0,
        reg_entry: float = 5.0,
    ) -> None:
        super().__init__(reg_voter=reg_voter, reg_entry=reg_entry)
        self.n_neighbors = n_neighbors
        self.shrinkage = shrinkage
        self.min_overlap = min_overlap

    def fit(self, train: pd.DataFrame, encoder: Encoder) -> "VoterNeighborhoodRecommender":
        super().fit(train, encoder)
        agg = aggregate_cells(train)
        rows, cols = encoder.encode(agg)
        rows = rows.astype(int)
        cols = cols.astype(int)

        residual = np.zeros((encoder.n_voters, encoder.n_entries))
        mask = np.zeros_like(residual)
        residual[rows, cols] = (
            agg[POINTS_COL].to_numpy(dtype=float)
            - self.mu_
            - self.voter_bias_[rows]
            - self.entry_bias_[cols]
        )
        mask[rows, cols] = 1.0

        dot = residual @ residual.T
        norms = np.sqrt(np.diag(dot))
        denom = np.outer(norms, norms)
        similarity = np.divide(dot, denom, out=np.zeros_like(dot), where=denom > 0)
        overlap = mask @ mask.T
        # Shrink towards zero so two voters that only ever co-voted a handful of
        # times cannot look like a bloc.
        similarity *= overlap / (overlap + self.shrinkage)
        similarity[overlap < self.min_overlap] = 0.0
        np.fill_diagonal(similarity, 0.0)

        self.residual_ = residual
        self.mask_ = mask
        self.similarity_ = similarity
        return self

    def _neighbor_term(self, voter: int, entry: int) -> float:
        sims = self.similarity_[voter] * self.mask_[:, entry]
        if not np.any(sims):
            return 0.0
        top = np.argsort(-np.abs(sims))[: self.n_neighbors]
        weights = sims[top]
        denom = np.abs(weights).sum()
        if denom == 0:
            return 0.0
        return float(weights @ self.residual_[top, entry] / denom)

    def predict(self, cells: pd.DataFrame) -> np.ndarray:
        rows, cols = self.encoder_.encode(cells)
        base = self._bias_prediction(cells)
        adjust = np.zeros(len(cells))
        for i, (r, c) in enumerate(zip(rows, cols)):
            if np.isnan(r) or np.isnan(c):
                continue
            adjust[i] = self._neighbor_term(int(r), int(c))
        return np.clip(base + adjust, 0.0, MAX_POINTS)


def ndcg_at_k(actual: np.ndarray, predicted: np.ndarray, k: int) -> float:
    order = np.argsort(-predicted, kind="stable")[:k]
    discounts = 1.0 / np.log2(np.arange(2, len(order) + 2))
    dcg = float(actual[order] @ discounts)
    ideal_order = np.sort(actual)[::-1][:k]
    idcg = float(ideal_order @ discounts[: len(ideal_order)])
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(actual: np.ndarray, predicted: np.ndarray, k: int) -> float:
    predicted_top = set(np.argsort(-predicted, kind="stable")[:k].tolist())
    actual_top = set(np.argsort(-actual, kind="stable")[:k].tolist())
    return len(predicted_top & actual_top) / k


def evaluate(
    cells: pd.DataFrame,
    predictions: np.ndarray,
    k: int = TOP_K,
    random_state: int = RANDOM_STATE,
) -> dict[str, float]:
    """RMSE over the ballot grid plus ranking quality per held-out ballot.

    Ties are broken by a seeded jitter so that a constant-prediction baseline
    scores like a random ranking instead of like the input ordering.
    """
    rng = np.random.default_rng(random_state)
    scored = cells.copy()
    scored["prediction"] = predictions
    scored["ranking_score"] = predictions + rng.normal(0.0, 1e-9, size=len(predictions))

    errors = scored["prediction"] - scored[POINTS_COL]
    awarded = scored[POINTS_COL] > 0

    precisions: list[float] = []
    ndcgs: list[float] = []
    ndcgs_10: list[float] = []
    for _, ballot in scored.groupby(BALLOT_COL):
        actual = ballot[POINTS_COL].to_numpy(dtype=float)
        predicted = ballot["ranking_score"].to_numpy(dtype=float)
        precisions.append(precision_at_k(actual, predicted, k))
        ndcgs.append(ndcg_at_k(actual, predicted, k))
        ndcgs_10.append(ndcg_at_k(actual, predicted, 2 * k))

    return {
        "n_cells": float(len(scored)),
        "n_ballots": float(scored[BALLOT_COL].nunique()),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "rmse_awarded_only": float(np.sqrt(np.mean(errors[awarded] ** 2))),
        f"precision_at_{k}": float(np.mean(precisions)),
        f"ndcg_at_{k}": float(np.mean(ndcgs)),
        f"ndcg_at_{2 * k}": float(np.mean(ndcgs_10)),
    }


def per_ballot_scores(
    cells: pd.DataFrame,
    predictions: np.ndarray,
    k: int = TOP_K,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """NDCG@k and mean squared error for every held-out ballot."""
    rng = np.random.default_rng(random_state)
    scored = cells.copy()
    scored["prediction"] = predictions
    scored["ranking_score"] = predictions + rng.normal(0.0, 1e-9, size=len(predictions))

    rows: list[dict[str, float | str]] = []
    for ballot, group in scored.groupby(BALLOT_COL):
        actual = group[POINTS_COL].to_numpy(dtype=float)
        rows.append(
            {
                BALLOT_COL: ballot,
                f"ndcg_at_{k}": ndcg_at_k(
                    actual, group["ranking_score"].to_numpy(dtype=float), k
                ),
                "mse": float(np.mean((group["prediction"].to_numpy() - actual) ** 2)),
            }
        )
    return pd.DataFrame(rows).set_index(BALLOT_COL)


def compare_models_paired(
    test: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    comparisons: list[tuple[str, str]] = PAIRED_COMPARISONS,
    k: int = TOP_K,
) -> pd.DataFrame:
    """Paired t-tests over held-out ballots.

    Ballots, not cells, are the unit: the ten awards on one ballot are forced to
    sum to 58 points, so cell-level errors within a ballot are anything but
    independent.
    """
    ballot_scores = {
        name: per_ballot_scores(test, preds) for name, preds in predictions.items()
    }
    rows: list[dict[str, float | str]] = []
    for model, reference in comparisons:
        left, right = ballot_scores[model], ballot_scores[reference]
        for metric, higher_is_better in ((f"ndcg_at_{k}", True), ("mse", False)):
            diff = left[metric] - right[metric]
            result = stats.ttest_rel(left[metric], right[metric])
            rows.append(
                {
                    "model": model,
                    "reference": reference,
                    "metric": metric,
                    "model_mean": float(left[metric].mean()),
                    "reference_mean": float(right[metric].mean()),
                    "mean_difference": float(diff.mean()),
                    "model_better": bool(
                        (diff.mean() > 0) if higher_is_better else (diff.mean() < 0)
                    ),
                    "t_statistic": float(result.statistic),
                    "p_value": float(result.pvalue),
                    "n_ballots": int(len(diff)),
                }
            )
    return pd.DataFrame(rows)


def repeated_split_evaluation(
    cells: pd.DataFrame,
    encoder: Encoder,
    mf_params: dict[str, float],
    implicit_params: dict[str, float],
    seeds: tuple[int, ...] = REPEAT_SEEDS,
) -> pd.DataFrame:
    """Re-run every model over several held-out voter-year splits.

    A single split leaves ~105 test ballots, so a few tenths of a point of RMSE
    could be split luck; the spread over seeds says whether the ordering holds.
    """
    rows: list[dict[str, float | str]] = []
    for seed in seeds:
        train, test = split_by_voter_year(cells, random_state=seed)
        for name, model, _ in build_models(mf_params, implicit_params):
            model.fit(train, encoder)
            rows.append(
                {"model": name, "seed": seed, **evaluate(test, model.predict(test))}
            )
    per_seed = pd.DataFrame(rows)
    summary = per_seed.groupby("model")[
        ["rmse", f"precision_at_{TOP_K}", f"ndcg_at_{TOP_K}"]
    ].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}_over_seeds" for metric, stat in summary.columns]
    return summary.reset_index()


def tune_recommender(
    name: str,
    factory: Callable[..., object],
    grid: dict[str, Sequence[float]],
    train: pd.DataFrame,
    encoder: Encoder,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Grid search on a validation split carved out of the training voter-years.

    The validation split reuses split_by_voter_year, so hyperparameters are
    never picked on cells from a ballot the inner model has already seen.
    """
    inner_train, validation = split_by_voter_year(
        train, test_size=VALIDATION_SIZE, random_state=RANDOM_STATE + 1
    )
    keys = list(grid)
    rows: list[dict[str, float | str]] = []
    for combination in product(*(grid[key] for key in keys)):
        params = dict(zip(keys, combination))
        model = factory(**params)
        model.fit(inner_train, encoder)
        scores = evaluate(validation, model.predict(validation))
        rows.append({"model": name, **params, **scores})
    results = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    best = {key: results.loc[0, key] for key in keys}
    return best, results


def format_params(params: dict[str, float]) -> str:
    return ", ".join(f"{key}={value:g}" for key, value in params.items())


def build_models(
    mf_params: dict[str, float], implicit_params: dict[str, float]
) -> list[tuple[str, object, dict[str, float]]]:
    neighborhood_params = {"n_neighbors": 10, "shrinkage": 25.0, "min_overlap": 30}
    return [
        ("baseline_global_mean", GlobalMeanRecommender(), {}),
        ("baseline_voter_mean", VoterMeanRecommender(), {}),
        ("bias_only_entry_quality", BiasRecommender(), {}),
        (
            "neighborhood_cf",
            VoterNeighborhoodRecommender(**neighborhood_params),
            neighborhood_params,
        ),
        (
            "implicit_als_hkv",
            ImplicitALSRecommender(
                n_factors=int(implicit_params["n_factors"]),
                reg=float(implicit_params["reg"]),
                alpha=float(implicit_params["alpha"]),
            ),
            implicit_params,
        ),
        (
            "matrix_factorization_als",
            MatrixFactorizationRecommender(
                n_factors=int(mf_params["n_factors"]), reg=float(mf_params["reg"])
            ),
            mf_params,
        ),
    ]


def evaluate_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
    encoder: Encoder,
    mf_params: dict[str, float],
    implicit_params: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows: list[dict[str, float | str]] = []
    predictions: dict[str, np.ndarray] = {}
    for name, model, params in build_models(mf_params, implicit_params):
        model.fit(train, encoder)
        preds = model.predict(test)
        predictions[name] = preds
        rows.append({"model": name, "params": format_params(params), **evaluate(test, preds)})
    return pd.DataFrame(rows), predictions


def build_examples(
    model: MatrixFactorizationRecommender,
    cells: pd.DataFrame,
    countries: list[str] = EXAMPLE_COUNTRIES,
    k: int = TOP_K,
    min_opportunities: int = MIN_EXAMPLE_OPPORTUNITIES,
) -> pd.DataFrame:
    """Predicted vs. actual favourite recipients, averaged over the eligible grid.

    Both columns average over every cell where the voter *could* have rewarded
    that country, so a single 12 to a one-off competitor cannot top the table;
    pairs with too few such opportunities are dropped entirely.
    """
    scored = cells.copy()
    scored["prediction"] = model.predict(cells)
    # "Lift" strips out how much everyone liked that entry, so the ranking shows
    # who a voter favours *beyond* the song's general appeal - the bloc signal
    # rather than the taste signal.
    scored["predicted_lift"] = scored["prediction"] - scored.groupby(ENTRY_COL)[
        "prediction"
    ].transform("mean")
    scored["actual_lift"] = scored[POINTS_COL] - scored.groupby(ENTRY_COL)[
        POINTS_COL
    ].transform("mean")

    per_pair = (
        scored.groupby([FROM_COL, TO_COL])
        .agg(
            predicted_mean_points=("prediction", "mean"),
            actual_mean_points=(POINTS_COL, "mean"),
            predicted_bloc_lift=("predicted_lift", "mean"),
            actual_bloc_lift=("actual_lift", "mean"),
            n_opportunities=(POINTS_COL, "size"),
        )
        .reset_index()
    )
    per_pair = per_pair[per_pair["n_opportunities"] >= min_opportunities]

    bases = [
        ("mean_points", "predicted_mean_points", "actual_mean_points"),
        ("bloc_lift", "predicted_bloc_lift", "actual_bloc_lift"),
    ]
    rows: list[dict[str, object]] = []
    for country in countries:
        subset = per_pair[per_pair[FROM_COL] == country]
        if subset.empty:
            continue
        for basis, predicted_col, actual_col in bases:
            predicted_top = subset.sort_values(predicted_col, ascending=False).head(k)
            actual_top = subset.sort_values(actual_col, ascending=False).head(k)
            actual_names = set(actual_top[TO_COL])
            for rank, (pred_row, actual_row) in enumerate(
                zip(predicted_top.itertuples(), actual_top.itertuples()), start=1
            ):
                rows.append(
                    {
                        "voter": country,
                        "basis": basis,
                        "rank": rank,
                        "predicted_country": pred_row.To,
                        "predicted_value": round(getattr(pred_row, predicted_col), 3),
                        "actual_country": actual_row.To,
                        "actual_value": round(getattr(actual_row, actual_col), 3),
                        "actual_opportunities": int(actual_row.n_opportunities),
                        f"predicted_in_actual_top{k}": pred_row.To in actual_names,
                    }
                )
    return pd.DataFrame(rows)



def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    votes = load_votes()
    cells = build_interactions(votes)
    encoder = make_encoder(cells)
    train, test = split_by_voter_year(cells)

    mf_params, mf_tuning = tune_recommender(
        "matrix_factorization_als", MatrixFactorizationRecommender, MF_GRID, train, encoder
    )
    implicit_params, implicit_tuning = tune_recommender(
        "implicit_als_hkv", ImplicitALSRecommender, IMPLICIT_GRID, train, encoder
    )
    tuning = pd.concat([mf_tuning, implicit_tuning], ignore_index=True)
    tuning.to_csv(TUNING_PATH, index=False)

    metrics, predictions = evaluate_models(train, test, encoder, mf_params, implicit_params)
    repeated = repeated_split_evaluation(cells, encoder, mf_params, implicit_params)
    metrics = metrics.merge(repeated, on="model", how="left")
    metrics.to_csv(METRICS_PATH, index=False)

    significance = compare_models_paired(test, predictions)
    significance.to_csv(SIGNIFICANCE_PATH, index=False)

    final_model = MatrixFactorizationRecommender(
        n_factors=int(mf_params["n_factors"]), reg=float(mf_params["reg"])
    ).fit(cells, encoder)
    examples = build_examples(final_model, cells)
    examples.to_csv(EXAMPLES_PATH, index=False)


    print(f"Ballot grid: {len(cells)} cells, {cells[BALLOT_COL].nunique()} ballots, "
          f"{cells[VOTER_YEAR_COL].nunique()} voter-years, {encoder.n_entries} entries")
    print(f"Best MF params: {format_params(mf_params)}")
    print(f"Best implicit-ALS params: {format_params(implicit_params)}")
    primary_cols = [
        "model",
        "rmse",
        "rmse_awarded_only",
        f"precision_at_{TOP_K}",
        f"ndcg_at_{TOP_K}",
        f"ndcg_at_{2 * TOP_K}",
    ]
    print("\nHeld-out metrics (primary split):")
    print(metrics[primary_cols].to_string(index=False))
    print(f"\nAcross {len(REPEAT_SEEDS)} splits:")
    print(
        metrics[["model", "rmse_mean_over_seeds", "rmse_std_over_seeds",
                 f"ndcg_at_{TOP_K}_mean_over_seeds"]].to_string(index=False)
    )
    print("\nPaired ballot-level tests:")
    print(significance.to_string(index=False))
    print("\nExamples (bloc-lift ranking):")
    print(examples[examples["basis"] == "bloc_lift"].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
