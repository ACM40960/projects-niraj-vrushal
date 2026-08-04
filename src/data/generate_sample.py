"""
Generate a small synthetic sample with the same schema as the PaySim
dataset (Lopez-Rojas et al., 2016).

This is NOT a substitute for the real dataset. Its only purpose is to let
the rest of the pipeline (loading, preprocessing, tests, notebooks) be
developed and run without requiring the ~470MB PaySim CSV to be downloaded
first. Swap it out for the real file in data/raw/ once available (see
README.md for the download link).

Usage:
    python -m src.data.generate_sample
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(seed=42)

TRANSACTION_TYPES = ["CASH-IN", "CASH-OUT", "DEBIT", "PAYMENT", "TRANSFER"]
# Real PaySim fraud only ever occurs on TRANSFER / CASH-OUT transactions.
FRAUD_ELIGIBLE_TYPES = ["TRANSFER", "CASH-OUT"]
FRAUD_RATE = 0.0013  # roughly matches the real dataset's ~0.13% fraud rate


def _make_account_id(prefix: str, rng: np.random.Generator) -> str:
    return f"{prefix}{rng.integers(10**9, 10**10)}"


def generate_sample(n_rows: int = 20_000, n_steps: int = 30) -> pd.DataFrame:
    """Generate a synthetic transaction dataset matching PaySim's schema."""
    steps = RNG.integers(1, n_steps + 1, size=n_rows)
    types = RNG.choice(TRANSACTION_TYPES, size=n_rows, p=[0.22, 0.35, 0.03, 0.30, 0.10])
    amounts = np.round(RNG.exponential(scale=8000, size=n_rows) + 1, 2)

    old_bal_orig = np.round(RNG.uniform(0, 50_000, size=n_rows), 2)
    # Balance decreases by the transaction amount
    new_bal_orig = np.clip(old_bal_orig - amounts, 0, None)

    old_bal_dest = np.round(RNG.uniform(0, 50_000, size=n_rows), 2)
    new_bal_dest = old_bal_dest + amounts

    name_orig = [_make_account_id("C", RNG) for _ in range(n_rows)]
    name_dest = [_make_account_id("C", RNG) for _ in range(n_rows)]

    is_fraud = np.zeros(n_rows, dtype=int)
    eligible_idx = np.where(np.isin(types, FRAUD_ELIGIBLE_TYPES))[0]
    n_fraud = max(1, int(len(eligible_idx) * FRAUD_RATE / (1 - FRAUD_RATE)))
    fraud_idx = RNG.choice(eligible_idx, size=min(n_fraud, len(eligible_idx)), replace=False)
    is_fraud[fraud_idx] = 1

    # Fraudulent transactions tend to drain the origin account fully.
    new_bal_orig[fraud_idx] = 0.0

    is_flagged_fraud = np.zeros(n_rows, dtype=int)
    big_transfer = (types == "TRANSFER") & (amounts > 200_000)
    is_flagged_fraud[big_transfer] = 1

    df = pd.DataFrame(
        {
            "step": steps,
            "type": types,
            "amount": amounts,
            "nameOrig": name_orig,
            "oldbalanceOrg": old_bal_orig,
            "newbalanceOrig": new_bal_orig,
            "nameDest": name_dest,
            "oldbalanceDest": old_bal_dest,
            "newbalanceDest": new_bal_dest,
            "isFraud": is_fraud,
            "isFlaggedFraud": is_flagged_fraud,
        }
    )
    return df.sort_values("step").reset_index(drop=True)


if __name__ == "__main__":
    import pathlib

    out_path = pathlib.Path(__file__).resolve().parents[2] / "data" / "raw" / "sample_paysim.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_sample()
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df):,} synthetic rows to {out_path}")
    print(f"Fraud rate: {df['isFraud'].mean():.4%}")
