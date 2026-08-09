"""
Bayesian risk-profiling model.

Implements the core contribution named in the literature review (section 3):
rather than a deterministic classifier's binary fraud/not-fraud flag, this
model produces a full posterior distribution over each transaction's fraud
probability, formally quantifying uncertainty (following the framing of
Canillas et al., 2020). The posterior mean becomes a continuous risk score;
its spread quantifies confidence. Combined with transaction amount, this is
what feeds the "expected financial yield" objective for the ILP optimizer
added in a later commit.

A Bayesian logistic regression is used: interpretable, and its posterior
over coefficients has a direct probabilistic reading. Fitting method scales
with data size:
  - NUTS (full MCMC): accurate posterior, but does not scale to millions of
    rows. Use for the synthetic sample / small subsets / testing.
  - ADVI (variational inference): scales to the full ~6.3M-row PaySim
    dataset; trades some posterior accuracy for tractable runtime. This is
    the default for `fit_bayesian_logistic`.
"""

import pathlib

import numpy as np
import pandas as pd
import pymc as pm
from scipy.special import expit
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

# Continuous columns get standardized before modeling; one-hot / binary
# columns (type_*, orig_balance_drained) are left as-is.
CONTINUOUS_FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "orig_balance_discrepancy",
    "dest_balance_discrepancy",
    "amount_to_orig_balance_ratio",
]


def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Standardize continuous columns (fit on train only); leave binary/
    one-hot columns untouched. Returns scaled copies plus the fitted scaler
    (needed to transform new data consistently at inference time)."""
    cont_cols = [c for c in CONTINUOUS_FEATURES if c in X_train.columns]
    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[cont_cols] = scaler.fit_transform(X_train[cont_cols])
    X_test_scaled[cont_cols] = scaler.transform(X_test[cont_cols])

    return X_train_scaled, X_test_scaled, scaler


def build_bayesian_logistic_model(
    X: np.ndarray, y: np.ndarray, batch_size: int | None = None
) -> pm.Model:
    """
    Bayesian logistic regression:
        fraud ~ Bernoulli(sigmoid(intercept + X @ coefs))
    with weakly informative Normal priors on the intercept and coefficients.

    If batch_size is given, uses minibatch ADVI (pm.Minibatch + total_size)
    so each optimization step only touches `batch_size` rows regardless of
    the full dataset size — essential for scaling to the full ~6.3M-row
    PaySim dataset, where full-batch gradients would be far too slow to
    compute every iteration. Ignored for NUTS sampling (full data is used).
    """
    n_features = X.shape[1]
    with pm.Model() as model:
        intercept = pm.Normal("intercept", mu=0, sigma=5)
        coefs = pm.Normal("coefs", mu=0, sigma=2, shape=n_features)

        if batch_size is not None:
            X_mb, y_mb = pm.Minibatch(X, y, batch_size=batch_size)
            logit_p = intercept + pm.math.dot(X_mb, coefs)
            pm.Bernoulli("obs", logit_p=logit_p, observed=y_mb, total_size=len(X))
        else:
            X_data = pm.Data("X_data", X)
            y_data = pm.Data("y_data", y)
            logit_p = intercept + pm.math.dot(X_data, coefs)
            pm.Bernoulli("obs", logit_p=logit_p, observed=y_data)

    return model


def fit_bayesian_logistic(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str = "advi",
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 2,
    advi_iterations: int = 20_000,
    batch_size: int | None = None,
    random_state: int = RANDOM_STATE,
    progressbar: bool = True,
):
    """
    Fit the Bayesian logistic regression and return (model, trace).

    method="advi" (default): variational inference, scales to the full
    dataset. Pass batch_size (e.g. 5000-10000) when fitting on the full
    ~6.3M-row PaySim dataset so each step only touches a minibatch rather
    than computing a gradient over all rows.
    method="nuts": full MCMC, far more accurate but only practical on
    small subsets (thousands of rows, not millions) - batch_size is
    ignored in this case.

    progressbar=True (default) shows a live progress bar with an ETA -
    without it, ADVI/NUTS give zero visual feedback while running, which
    can look identical to a hang, especially when PyTensor has fallen
    back to pure Python (no C compiler available). Tests pass
    progressbar=False to keep output quiet.
    """
    X_values = X_train.values.astype(float)
    y_values = y_train.values.astype(float)
    model = build_bayesian_logistic_model(
        X_values, y_values, batch_size=batch_size if method == "advi" else None
    )

    with model:
        if method == "advi":
            approx = pm.fit(
                n=advi_iterations,
                method="advi",
                random_seed=random_state,
                progressbar=progressbar,
            )
            trace = approx.sample(draws)
        elif method == "nuts":
            trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                random_seed=random_state,
                progressbar=progressbar,
            )
        else:
            raise ValueError(f"Unknown method: {method!r}. Use 'advi' or 'nuts'.")

    return model, trace


def posterior_predictive_probabilities(trace, X_new: pd.DataFrame) -> np.ndarray:
    """
    Compute the posterior distribution over fraud probability for each row
    in X_new, using every posterior draw of (intercept, coefs).

    Returns an array of shape (n_posterior_draws, n_rows): each row is one
    posterior draw's fraud-probability estimate for every transaction. This
    is the "full posterior distribution" the literature review calls for,
    not just a point estimate.
    """
    intercept = trace.posterior["intercept"].values.reshape(-1)
    coefs = trace.posterior["coefs"].values.reshape(-1, trace.posterior["coefs"].shape[-1])

    X_values = X_new.values.astype(float)
    logits = intercept[:, None] + coefs @ X_values.T
    probs = expit(logits)  # numerically stable sigmoid (avoids exp overflow)
    return probs


def summarize_posterior_probabilities(probs: np.ndarray) -> pd.DataFrame:
    """
    Collapse the (n_draws, n_rows) posterior probability array into a
    per-transaction summary: posterior mean (the point risk score), std
    (uncertainty), and a 94% credible interval.
    """
    return pd.DataFrame(
        {
            "fraud_prob_mean": probs.mean(axis=0),
            "fraud_prob_std": probs.std(axis=0),
            "fraud_prob_hdi_3%": np.percentile(probs, 3, axis=0),
            "fraud_prob_hdi_97%": np.percentile(probs, 97, axis=0),
        }
    )


if __name__ == "__main__":
    from src.data.preprocessing import build_feature_matrix

    processed_dir = pathlib.Path(__file__).resolve().parents[2] / "data" / "processed"
    train_path, test_path = processed_dir / "train.csv", processed_dir / "test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "No processed train/test data found. Run "
            "`python -m src.data.preprocessing` first."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train, y_train = build_feature_matrix(train_df)
    X_test, y_test = build_feature_matrix(test_df)

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Auto-select minibatching based on training set size. Full-batch ADVI
    # is fine (and converges best, per earlier testing) on the small
    # synthetic sample, but on the real ~5M-row PaySim training set every
    # iteration would otherwise need a gradient over the whole dataset -
    # intractable, especially without a C compiler available. 10,000 is a
    # reasonable batch size: large enough to see fraud examples reasonably
    # often even at PaySim's real ~0.13% fraud rate, small enough to keep
    # each iteration fast.
    n_train = len(X_train_scaled)
    batch_size = 10_000 if n_train > 50_000 else None
    print(f"Training set size: {n_train:,} rows -> "
          f"{'minibatch ADVI (batch_size=' + str(batch_size) + ')' if batch_size else 'full-batch ADVI'}")

    print("Fitting Bayesian logistic regression via ADVI...")
    model, trace = fit_bayesian_logistic(
        X_train_scaled, y_train, method="advi", batch_size=batch_size
    )

    probs = posterior_predictive_probabilities(trace, X_test_scaled)
    summary = summarize_posterior_probabilities(probs)

    print(summary.describe())
    print(f"\nMean posterior fraud probability for actual fraud cases: "
          f"{summary.loc[y_test.values == 1, 'fraud_prob_mean'].mean():.4f}")
    print(f"Mean posterior fraud probability for legitimate cases:    "
          f"{summary.loc[y_test.values == 0, 'fraud_prob_mean'].mean():.4f}")

    # Save results in the same format as the baseline classifiers, so all
    # models are directly comparable in reports/metrics.json. A 0.5
    # threshold on the posterior mean gives a hard label for this
    # comparison; the model's real advantage (a continuous, uncertainty-
    # aware score) is used directly in the ILP optimizer in a later commit.
    from sklearn.metrics import (
        average_precision_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    from src.evaluation.report_utils import FIGURES_DIR, plot_calibration_curve, save_metrics

    y_pred = (summary["fraud_prob_mean"] >= 0.5).astype(int)
    metrics = {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, summary["fraud_prob_mean"]),
        "average_precision": average_precision_score(y_test, summary["fraud_prob_mean"]),
    }
    save_metrics("Bayesian Logistic Regression", metrics)

    plot_calibration_curve(
        y_test.values,
        summary["fraud_prob_mean"].values,
        title="Bayesian Logistic Regression — Calibration",
        save_path=FIGURES_DIR / "bayesian_calibration.png",
    )
    print(f"\nSaved metrics and calibration plot to reports/")
