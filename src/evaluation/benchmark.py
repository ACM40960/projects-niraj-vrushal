"""
End-to-end system benchmark.

Every prior commit validated one stage in isolation: the Bayesian model
was compared to baselines as a *classifier*, and the ILP was compared to
a naive top-k *allocator*. Neither test actually validates this project's
central claim - that a Bayesian posterior specifically (not just any
probability estimate) is what enables good capacity-constrained
allocation, following Canillas et al. (2020)'s framing.

This module runs two benchmarks that do test it directly:

1. **Score-source comparison**: the identical ILP formulation (same
   capacity, sector caps, mandatory sampling) driven by two different
   score sources - the Bayesian posterior mean, and the baseline Random
   Forest's predict_proba - each also compared to its own naive top-k
   baseline. This isolates whether the *scoring model* matters once
   you're already using constrained optimization, from whether
   *constrained optimization* matters at all.

2. **Capacity sensitivity**: every prior ILP run used a single capacity
   (1000). This sweeps capacity to show how the system's real-world
   outcomes scale as the auditor budget grows or shrinks.
"""

import pathlib

import numpy as np
import pandas as pd

from src.data.preprocessing import build_feature_matrix
from src.models.baseline import train_random_forest
from src.models.bayesian_model import CONTINUOUS_FEATURES, load_bayesian_artifacts
from src.models.expected_yield import build_yield_table, compute_expected_yield_summary
from src.optimization.ilp_optimizer import (
    build_candidate_pool,
    naive_topk_baseline,
    solve_auditor_allocation,
    summarize_allocation,
)

DATA_PROCESSED_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "processed"
REPORTS_DIR = pathlib.Path(__file__).resolve().parents[2] / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42


def rf_yield_table(test_df: pd.DataFrame, rf_model, X_test: pd.DataFrame) -> pd.DataFrame:
    """
    Build a yield table driven by the Random Forest baseline's
    predict_proba instead of the Bayesian posterior mean, using the
    identical build_yield_table() the Bayesian pipeline uses. From this
    point on, the only difference between the two pipelines is the score
    source - isolating its effect on the ILP's real-world outcomes.
    """
    fraud_prob = rf_model.predict_proba(X_test)[:, 1]
    yield_summary = pd.DataFrame(
        {
            "fraud_prob_mean": fraud_prob,
            "expected_yield_mean": fraud_prob * test_df["amount"].to_numpy(),
        }
    )
    return build_yield_table(test_df, yield_summary)


def compare_score_sources(
    bayesian_yield_table: pd.DataFrame,
    rf_yield_table_: pd.DataFrame,
    capacity: float,
    sector_cap_fraction: float | None,
    shortlist_size: int = 10_000,
    random_sample_size: int = 500,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Run the identical ILP pipeline (candidate pool construction, ILP
    solve, naive top-k) on both score sources and compare real-world
    outcomes side by side - a 2x2 that isolates score source (Bayesian
    vs RF) from allocation method (ILP vs naive).
    """
    rows = []
    for label, yield_table in [("Bayesian", bayesian_yield_table), ("Random Forest", rf_yield_table_)]:
        candidates = build_candidate_pool(
            yield_table,
            shortlist_size=shortlist_size,
            random_sample_size=random_sample_size,
            random_state=random_state,
        )
        ilp_allocation = solve_auditor_allocation(candidates, capacity=capacity, sector_cap_fraction=sector_cap_fraction)
        naive_allocation = naive_topk_baseline(candidates, capacity=capacity)

        rows.append(summarize_allocation(ilp_allocation, f"{label} + ILP"))
        rows.append(summarize_allocation(naive_allocation, f"{label} + naive top-k"))

    df = pd.DataFrame(rows)
    df["precision"] = df["fraud_cases_caught"] / df["n_audited"].replace(0, np.nan)
    return df


def capacity_sensitivity(
    yield_table: pd.DataFrame,
    capacities: list,
    sector_cap_fraction: float | None = 0.5,
    shortlist_size: int = 10_000,
    random_sample_size: int = 500,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Sweep auditor-capacity budget for a single score source's ILP-driven
    allocation, showing how outcomes scale with resourcing.

    A fixed-size mandatory random sample can genuinely be infeasible at
    very low capacities (e.g. 500 mandatory rows cannot fit in a 100-unit
    budget, regardless of what the optimizer does with the rest) - this
    is a real constraint, not a bug, so those capacity values are skipped
    with a printed explanation rather than crashing the whole sweep. In
    practice this itself is an informative result: it shows the minimum
    viable capacity for the mandatory-sampling requirement to be
    satisfiable at all, given random_sample_size.
    """
    candidates = build_candidate_pool(
        yield_table, shortlist_size=shortlist_size, random_sample_size=random_sample_size, random_state=random_state
    )
    rows = []
    for capacity in capacities:
        try:
            allocation = solve_auditor_allocation(candidates, capacity=capacity, sector_cap_fraction=sector_cap_fraction)
        except ValueError as e:
            print(f"Skipping capacity={capacity}: {e}")
            continue
        summary = summarize_allocation(allocation, f"capacity={capacity}")
        summary["capacity"] = capacity
        summary["precision"] = summary["fraud_cases_caught"] / summary["n_audited"] if summary["n_audited"] else 0.0
        rows.append(summary)
    return pd.DataFrame(rows)


def plot_score_source_comparison(comparison: pd.DataFrame, save_path: pathlib.Path) -> None:
    """Grouped bar chart: fraud $ caught for each (score source, allocation
    method) combination - the key figure for the score-source benchmark."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4C72B0" if "ILP" in label else "#C44E52" for label in comparison["label"]]
    ax.bar(comparison["label"], comparison["fraud_amount_caught"], color=colors)
    ax.set_ylabel("Fraud $ caught")
    ax.set_title("Score source x allocation method: real fraud $ captured")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_capacity_sensitivity(sensitivity_df: pd.DataFrame, save_path: pathlib.Path) -> None:
    """Line chart: fraud $ caught and precision vs. auditor capacity."""
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(sensitivity_df["capacity"], sensitivity_df["fraud_amount_caught"], marker="o", color="#4C72B0")
    ax1.set_xlabel("Capacity (investigator-time budget)")
    ax1.set_ylabel("Fraud $ caught", color="#4C72B0")
    ax1.tick_params(axis="y", labelcolor="#4C72B0")

    ax2 = ax1.twinx()
    ax2.plot(sensitivity_df["capacity"], sensitivity_df["precision"], marker="s", color="#C44E52")
    ax2.set_ylabel("Precision", color="#C44E52")
    ax2.tick_params(axis="y", labelcolor="#C44E52")

    ax1.set_title("Capacity sensitivity: fraud $ caught and precision vs. budget")
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    train_path, test_path = DATA_PROCESSED_DIR / "train.csv", DATA_PROCESSED_DIR / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("No processed data found. Run `python -m src.data.preprocessing` first.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    X_train, y_train = build_feature_matrix(train_df)
    X_test, y_test = build_feature_matrix(test_df)

    print("Training Random Forest baseline...")
    rf_model = train_random_forest(X_train, y_train)
    rf_table = rf_yield_table(test_df, rf_model, X_test)

    print("Loading fitted Bayesian model...")
    trace, scaler = load_bayesian_artifacts()
    cont_cols = [c for c in CONTINUOUS_FEATURES if c in X_test.columns]
    X_test_scaled = X_test.copy()
    X_test_scaled[cont_cols] = scaler.transform(X_test[cont_cols])
    bayesian_summary = compute_expected_yield_summary(trace, X_test_scaled, test_df["amount"])
    bayesian_table = build_yield_table(test_df, bayesian_summary)

    capacity = 1000.0
    sector_cap_fraction = 0.5

    print(f"\n=== Score-source comparison (capacity={capacity}, sector_cap_fraction={sector_cap_fraction}) ===")
    comparison = compare_score_sources(bayesian_table, rf_table, capacity=capacity, sector_cap_fraction=sector_cap_fraction)
    print(comparison.to_string(index=False))

    comparison_path = REPORTS_DIR / "score_source_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    print(f"\nSaved score-source comparison to {comparison_path}")

    plot_score_source_comparison(comparison, save_path=FIGURES_DIR / "score_source_comparison.png")
    print(f"Saved score-source comparison plot to {FIGURES_DIR / 'score_source_comparison.png'}")

    print(f"\n=== Capacity sensitivity (Bayesian + ILP, sector_cap_fraction={sector_cap_fraction}) ===")
    capacities = [100, 250, 500, 1000, 2000, 5000, 8000]
    # Smaller mandatory sample than the main comparison (500) so more of
    # the low-capacity range is actually solvable rather than skipped -
    # see capacity_sensitivity()'s docstring for why very low capacities
    # can be genuinely infeasible with a fixed mandatory-sample size.
    cap_sensitivity = capacity_sensitivity(
        bayesian_table, capacities=capacities, sector_cap_fraction=sector_cap_fraction, random_sample_size=100
    )
    print(cap_sensitivity[["label", "n_audited", "fraud_cases_caught", "fraud_amount_caught", "precision"]].to_string(index=False))

    cap_sensitivity_path = REPORTS_DIR / "capacity_sensitivity.csv"
    cap_sensitivity.to_csv(cap_sensitivity_path, index=False)
    print(f"\nSaved capacity sensitivity to {cap_sensitivity_path}")

    plot_capacity_sensitivity(cap_sensitivity, save_path=FIGURES_DIR / "capacity_sensitivity.png")
    print(f"Saved capacity sensitivity plot to {FIGURES_DIR / 'capacity_sensitivity.png'}")
