import numpy as np
import pandas as pd
import pytest

from src.evaluation.benchmark import capacity_sensitivity, compare_score_sources, plot_capacity_sensitivity, plot_score_source_comparison

TYPES = ["CASH-IN", "CASH-OUT", "DEBIT", "PAYMENT", "TRANSFER"]


def _make_yield_table(n_rows: int, n_fraud: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    types = rng.choice(TYPES, size=n_rows)
    df = pd.DataFrame(
        {
            "step": rng.integers(1, 30, size=n_rows),
            "amount": rng.exponential(5000, size=n_rows).round(2),
            "isFraud": 0,
            "fraud_prob_mean": rng.uniform(0, 0.05, size=n_rows),
        }
    )
    for t in TYPES:
        df[f"type_{t}"] = (types == t).astype(int)

    fraud_idx = rng.choice(n_rows, size=n_fraud, replace=False)
    df.loc[fraud_idx, "isFraud"] = 1
    df.loc[fraud_idx, "fraud_prob_mean"] = rng.uniform(0.6, 0.95, size=n_fraud)

    df["expected_yield_mean"] = df["fraud_prob_mean"] * df["amount"]
    return df.sort_values("expected_yield_mean", ascending=False).reset_index(drop=True)


@pytest.fixture
def two_yield_tables():
    bayesian_table = _make_yield_table(n_rows=3000, n_fraud=15, seed=1)
    rf_table = _make_yield_table(n_rows=3000, n_fraud=15, seed=2)
    return bayesian_table, rf_table


def test_compare_score_sources_returns_four_rows(two_yield_tables):
    bayesian_table, rf_table = two_yield_tables
    comparison = compare_score_sources(
        bayesian_table, rf_table, capacity=200.0, sector_cap_fraction=0.5,
        shortlist_size=500, random_sample_size=50,
    )
    assert len(comparison) == 4
    expected_labels = {
        "Bayesian + ILP", "Bayesian + naive top-k",
        "Random Forest + ILP", "Random Forest + naive top-k",
    }
    assert set(comparison["label"]) == expected_labels
    assert "precision" in comparison.columns


def test_capacity_sensitivity_yield_is_monotonic_nondecreasing(two_yield_tables):
    bayesian_table, _ = two_yield_tables
    capacities = [50, 100, 200, 400]
    sensitivity_df = capacity_sensitivity(
        bayesian_table, capacities=capacities, sector_cap_fraction=0.5,
        shortlist_size=500, random_sample_size=50,
    )
    assert len(sensitivity_df) == len(capacities)
    # more budget can never hurt an ILP maximizing yield subject to that budget
    yields = sensitivity_df.sort_values("capacity")["total_expected_yield"].to_numpy()
    assert (np.diff(yields) >= -1e-6).all()


def test_capacity_sensitivity_includes_capacity_column(two_yield_tables):
    bayesian_table, _ = two_yield_tables
    sensitivity_df = capacity_sensitivity(
        bayesian_table, capacities=[100, 200], sector_cap_fraction=0.5,
        shortlist_size=500, random_sample_size=50,
    )
    assert list(sensitivity_df["capacity"]) == [100, 200]


def test_capacity_sensitivity_skips_infeasible_low_capacity_gracefully(two_yield_tables, capsys):
    bayesian_table, _ = two_yield_tables
    # random_sample_size=200 with real per-row costs (0.5-1.5) means the
    # mandatory sample alone costs well over capacity=10 - this used to
    # crash the whole sweep; it should now just skip that one value.
    capacities = [10, 500]
    sensitivity_df = capacity_sensitivity(
        bayesian_table, capacities=capacities, sector_cap_fraction=0.5,
        shortlist_size=500, random_sample_size=200,
    )
    assert list(sensitivity_df["capacity"]) == [500]
    captured = capsys.readouterr()
    assert "Skipping capacity=10" in captured.out


def test_plot_functions_run_without_error(two_yield_tables, tmp_path):
    bayesian_table, rf_table = two_yield_tables
    comparison = compare_score_sources(
        bayesian_table, rf_table, capacity=200.0, sector_cap_fraction=0.5,
        shortlist_size=500, random_sample_size=50,
    )
    save_path_1 = tmp_path / "comparison.png"
    plot_score_source_comparison(comparison, save_path=save_path_1)
    assert save_path_1.exists()

    sensitivity_df = capacity_sensitivity(
        bayesian_table, capacities=[100, 200], sector_cap_fraction=0.5,
        shortlist_size=500, random_sample_size=50,
    )
    save_path_2 = tmp_path / "sensitivity.png"
    plot_capacity_sensitivity(sensitivity_df, save_path=save_path_2)
    assert save_path_2.exists()
