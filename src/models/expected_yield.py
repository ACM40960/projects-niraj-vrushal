"""
Expected financial yield derivation.

Combines the Bayesian model's posterior fraud probability with each
transaction's dollar amount into a single "expected yield" score:

    expected_yield = P(fraud) * amount

This is the objective function the ILP optimizer (next commit) maximizes
when deciding which transactions to allocate limited auditor capacity to
- directly implementing the literature review's framing (section 3): "By
modeling the probability of non-compliance alongside expected financial
discrepancies, Bayesian frameworks provide a continuous expected yield."

Because fraud probability is a full posterior distribution rather than a
point estimate, so is the resulting expected yield: this module summarizes
that distribution (mean, std, 94% HDI) per transaction rather than
collapsing straight to a single number.
"""

import pathlib

import numpy as np
import pandas as pd

from src.data.preprocessing import build_feature_matrix
from src.models.bayesian_model import (
    CONTINUOUS_FEATURES,
    RANDOM_STATE,
    load_bayesian_artifacts,
    posterior_predictive_probabilities,
)

DATA_PROCESSED_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "processed"
YIELD_TABLE_PATH = DATA_PROCESSED_DIR / "expected_yield.csv"

REPORTS_DIR = pathlib.Path(__file__).resolve().parents[2] / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def compute_expected_yield_summary(
    trace,
    X_new: pd.DataFrame,
    amounts: pd.Series,
    max_draws: int = 100,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    For every transaction in X_new, compute the posterior distribution of
    expected yield (fraud_prob * amount) and summarize it.

    Uses at most `max_draws` posterior draws (deterministically subsampled)
    in float32 to keep memory bounded on large test sets - e.g. 100 draws
    x 1.3M rows x 4 bytes ~ 500MB, versus ~10GB for the full posterior at
    float64. See posterior_predictive_probabilities() for details.
    """
    probs = posterior_predictive_probabilities(
        trace, X_new, max_draws=max_draws, dtype=np.float32, random_state=random_state
    )  # shape (n_draws, n_rows)

    amounts_arr = amounts.values.astype(np.float32)
    yield_draws = probs * amounts_arr[None, :]

    return pd.DataFrame(
        {
            "fraud_prob_mean": probs.mean(axis=0),
            "expected_yield_mean": yield_draws.mean(axis=0),
            "expected_yield_std": yield_draws.std(axis=0),
            "expected_yield_hdi_3%": np.percentile(yield_draws, 3, axis=0),
            "expected_yield_hdi_97%": np.percentile(yield_draws, 97, axis=0),
        }
    )


def build_yield_table(test_df: pd.DataFrame, yield_summary: pd.DataFrame) -> pd.DataFrame:
    """Merge expected-yield scores with transaction identifiers and ground
    truth, ranked descending by expected yield. This table is exactly what
    the ILP optimizer consumes as its objective function input."""
    table = test_df[["step", "amount", "isFraud"]].reset_index(drop=True).copy()
    table = pd.concat([table, yield_summary.reset_index(drop=True)], axis=1)
    return table.sort_values("expected_yield_mean", ascending=False).reset_index(drop=True)


def capture_curve(yield_table: pd.DataFrame, k_values: list[int]) -> pd.DataFrame:
    """
    For each k in k_values: if only the top-k transactions ranked by
    expected yield could be audited, what fraction of all fraud cases (and
    what fraction of total fraud dollar value) would be caught? This
    previews exactly the tradeoff the ILP optimizer manages under a real
    auditor-capacity constraint.
    """
    total_fraud_count = yield_table["isFraud"].sum()
    total_fraud_amount = yield_table.loc[yield_table["isFraud"] == 1, "amount"].sum()

    rows = []
    for k in k_values:
        top_k = yield_table.head(k)
        fraud_caught = int(top_k["isFraud"].sum())
        amount_caught = top_k.loc[top_k["isFraud"] == 1, "amount"].sum()
        rows.append(
            {
                "k": k,
                "fraud_cases_caught": fraud_caught,
                "fraud_capture_rate": fraud_caught / total_fraud_count if total_fraud_count else 0.0,
                "fraud_amount_caught": amount_caught,
                "fraud_amount_capture_rate": (
                    amount_caught / total_fraud_amount if total_fraud_amount else 0.0
                ),
                "precision_at_k": fraud_caught / k if k else 0.0,
            }
        )
    return pd.DataFrame(rows)


def plot_capture_curve(
    yield_table: pd.DataFrame, max_k: int, save_path: pathlib.Path, n_points: int = 50
) -> None:
    """Gains chart: cumulative fraud captured vs. number of transactions
    audited, ranked by expected yield."""
    import matplotlib.pyplot as plt

    k_values = sorted(set(np.linspace(1, max_k, n_points).astype(int).tolist()))
    curve = capture_curve(yield_table, k_values)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(curve["k"], curve["fraud_capture_rate"] * 100, marker=".", label="Fraud cases captured")
    ax.plot(
        curve["k"], curve["fraud_amount_capture_rate"] * 100, marker=".", label="Fraud $ captured"
    )
    ax.set_xlabel("Transactions audited (ranked by expected yield)")
    ax.set_ylabel("% of total fraud captured")
    ax.set_title("Capture rate vs. audit capacity")
    ax.legend()
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    test_path = DATA_PROCESSED_DIR / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(
            "No processed test data found. Run `python -m src.data.preprocessing` first."
        )

    test_df = pd.read_csv(test_path)
    X_test, y_test = build_feature_matrix(test_df)

    print("Loading fitted Bayesian model...")
    trace, scaler = load_bayesian_artifacts()

    cont_cols = [c for c in CONTINUOUS_FEATURES if c in X_test.columns]
    X_test_scaled = X_test.copy()
    X_test_scaled[cont_cols] = scaler.transform(X_test[cont_cols])

    print(f"Scoring {len(X_test):,} test transactions...")
    yield_summary = compute_expected_yield_summary(trace, X_test_scaled, test_df["amount"])
    yield_table = build_yield_table(test_df, yield_summary)

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    yield_table.to_csv(YIELD_TABLE_PATH, index=False)
    print(f"Saved yield table to {YIELD_TABLE_PATH}")

    n_fraud = int(yield_table["isFraud"].sum())
    candidate_ks = [n_fraud, n_fraud * 2, n_fraud * 5, 100, 500, 1000, max(len(yield_table) // 100, 1)]
    k_values = sorted({k for k in candidate_ks if 0 < k <= len(yield_table)})

    curve = capture_curve(yield_table, k_values)
    print("\nCapture rate by audit capacity (k):")
    print(curve.to_string(index=False))

    plot_path = FIGURES_DIR / "yield_capture_curve.png"
    plot_capture_curve(yield_table, max_k=min(len(yield_table), max(k_values) * 3), save_path=plot_path)
    print(f"\nSaved capture curve plot to {plot_path}")
