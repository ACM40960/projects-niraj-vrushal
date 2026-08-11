import numpy as np
import pandas as pd
import pytest

from src.optimization.ilp_optimizer import (
    DEFAULT_AUDIT_COST_BY_TYPE,
    build_candidate_pool,
    compare_sector_composition,
    naive_topk_baseline,
    sector_cap_sensitivity,
    solve_auditor_allocation,
    summarize_allocation,
)

TYPES = ["CASH-IN", "CASH-OUT", "DEBIT", "PAYMENT", "TRANSFER"]


def _make_yield_table(n_rows: int, n_fraud: int = 5, random_state: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
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
def yield_table():
    return _make_yield_table(n_rows=3000, n_fraud=10)


def test_build_candidate_pool_includes_mandatory_random_sample(yield_table):
    candidates = build_candidate_pool(yield_table, shortlist_size=1000, random_sample_size=200)
    assert len(candidates) == 1200
    assert candidates["mandatory"].sum() == 200
    mandatory_rows = candidates[candidates["mandatory"]]
    assert len(mandatory_rows) == 200


def test_build_candidate_pool_handles_small_dataset_gracefully():
    small_table = _make_yield_table(n_rows=50, n_fraud=2)
    candidates = build_candidate_pool(small_table, shortlist_size=1000, random_sample_size=500)
    # shortlist_size exceeds available rows - should just return everything,
    # with zero mandatory rows since there's no remainder to sample from.
    assert len(candidates) == 50
    assert candidates["mandatory"].sum() == 0


def test_solve_auditor_allocation_respects_capacity(yield_table):
    candidates = build_candidate_pool(yield_table, shortlist_size=500, random_sample_size=50)
    capacity = 100.0
    allocation = solve_auditor_allocation(candidates, capacity=capacity, sector_cap_fraction=0.5)

    audited = allocation[allocation["audited"] == 1]
    type_cols = [c for c in audited.columns if c.startswith("type_")]
    inferred_type = audited[type_cols].idxmax(axis=1).str.replace("type_", "", regex=False)
    costs = inferred_type.map(DEFAULT_AUDIT_COST_BY_TYPE)
    assert costs.sum() <= capacity + 1e-6


def test_solve_auditor_allocation_forces_mandatory_inclusion(yield_table):
    candidates = build_candidate_pool(yield_table, shortlist_size=200, random_sample_size=100)
    # Generous capacity so the solver isn't forced to drop mandatory rows
    # for infeasibility reasons in this test.
    allocation = solve_auditor_allocation(candidates, capacity=1000.0, sector_cap_fraction=None)

    mandatory_mask = candidates["mandatory"].values
    assert (allocation.loc[mandatory_mask, "audited"] == 1).all()


def test_solve_auditor_allocation_respects_sector_cap(yield_table):
    candidates = build_candidate_pool(yield_table, shortlist_size=1000, random_sample_size=0)
    capacity = 500.0
    sector_cap_fraction = 0.3
    allocation = solve_auditor_allocation(
        candidates, capacity=capacity, sector_cap_fraction=sector_cap_fraction
    )

    audited = allocation[allocation["audited"] == 1]
    type_cols = [c for c in audited.columns if c.startswith("type_")]
    inferred_type = audited[type_cols].idxmax(axis=1).str.replace("type_", "", regex=False)
    costs = inferred_type.map(DEFAULT_AUDIT_COST_BY_TYPE)

    for t in TYPES:
        sector_cost = costs[inferred_type == t].sum()
        assert sector_cost <= sector_cap_fraction * capacity + 1e-6


def test_mandatory_rows_exempt_from_sector_cap(yield_table):
    # Regression test: previously, if mandatory random rows concentrated in
    # one sector cost more than sector_cap_fraction * capacity on their
    # own, the ILP was infeasible even though nothing was actually wrong -
    # mandatory rows are a compliance floor and shouldn't be blocked by a
    # cap meant to limit the *optimizer's* discretionary picks.
    candidates = build_candidate_pool(yield_table, shortlist_size=200, random_sample_size=100)

    # Force every mandatory row into a single sector so its cost alone
    # would exceed a very tight cap.
    mandatory_mask = candidates["mandatory"]
    for col in [c for c in candidates.columns if c.startswith("type_")]:
        candidates.loc[mandatory_mask, col] = 0
    candidates.loc[mandatory_mask, "type_TRANSFER"] = 1

    capacity = 500.0
    tight_cap_fraction = 0.01  # sector cap = 5.0, far less than mandatory TRANSFER cost alone

    # Should NOT raise - this used to be a RuntimeError('...Infeasible...').
    allocation = solve_auditor_allocation(candidates, capacity=capacity, sector_cap_fraction=tight_cap_fraction)

    assert (allocation.loc[mandatory_mask, "audited"] == 1).all()


def test_mandatory_cost_exceeding_capacity_raises_clear_error(yield_table):
    candidates = build_candidate_pool(yield_table, shortlist_size=200, random_sample_size=100)
    with pytest.raises(ValueError, match="Mandatory random-sample"):
        solve_auditor_allocation(candidates, capacity=1.0, sector_cap_fraction=0.5)


def test_naive_topk_baseline_respects_capacity(yield_table):
    candidates = build_candidate_pool(yield_table, shortlist_size=500, random_sample_size=0)
    capacity = 100.0
    allocation = naive_topk_baseline(candidates, capacity=capacity)

    audited = allocation[allocation["audited"] == 1]
    type_cols = [c for c in audited.columns if c.startswith("type_")]
    inferred_type = audited[type_cols].idxmax(axis=1).str.replace("type_", "", regex=False)
    costs = inferred_type.map(DEFAULT_AUDIT_COST_BY_TYPE)
    assert costs.sum() <= capacity + 1e-6


def test_summarize_allocation_returns_expected_keys(yield_table):
    candidates = build_candidate_pool(yield_table, shortlist_size=200, random_sample_size=20)
    allocation = solve_auditor_allocation(candidates, capacity=50.0, sector_cap_fraction=0.5)
    summary = summarize_allocation(allocation, "test")

    for key in ["label", "n_audited", "total_expected_yield", "fraud_cases_caught", "fraud_amount_caught"]:
        assert key in summary
    assert summary["n_audited"] == int(allocation["audited"].sum())


def test_sector_cap_sensitivity_returns_one_row_per_cap(yield_table):
    candidates = build_candidate_pool(yield_table, shortlist_size=300, random_sample_size=30)
    cap_fractions = [0.3, 0.6, None]
    sensitivity_df = sector_cap_sensitivity(candidates, capacity=100.0, cap_fractions=cap_fractions)

    assert len(sensitivity_df) == len(cap_fractions)
    for col in ["total_expected_yield", "fraud_cases_caught", "fraud_amount_caught", "precision"]:
        assert col in sensitivity_df.columns
    # a tighter cap can never audit more of the highest-yield (typically
    # single-sector-concentrated) transactions than the unconstrained case
    unconstrained_yield = sensitivity_df.loc[sensitivity_df["sector_cap_fraction"].isna(), "total_expected_yield"].iloc[0]
    tightest_yield = sensitivity_df.loc[sensitivity_df["sector_cap_fraction"] == 0.3, "total_expected_yield"].iloc[0]
    assert tightest_yield <= unconstrained_yield + 1e-6


def test_compare_sector_composition_returns_per_sector_breakdown(yield_table):
    candidates = build_candidate_pool(yield_table, shortlist_size=500, random_sample_size=50)
    comparison = compare_sector_composition(candidates, capacity=200.0, cap_a=0.3, cap_b=None)

    assert "sector" in comparison.columns
    assert "delta_audited" in comparison.columns
    assert "delta_fraud" in comparison.columns
    for t in TYPES:
        assert t in comparison["sector"].values
    # precision columns should be valid rates
    precision_cols = [c for c in comparison.columns if c.startswith("precision_")]
    assert len(precision_cols) == 2
    for col in precision_cols:
        assert comparison[col].between(0, 1).all()
