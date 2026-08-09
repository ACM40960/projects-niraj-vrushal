"""
Utilities for persisting model evaluation results to reports/.

Keeps a running record (reports/metrics.json) of every model's headline
metrics so results can be compared across commits and pulled directly into
the final write-up, plus a saved confusion-matrix plot per model.
"""

import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

REPORTS_DIR = pathlib.Path(__file__).resolve().parents[2] / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_PATH = REPORTS_DIR / "metrics.json"


def _load_existing_metrics() -> dict:
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            return json.load(f)
    return {}


def save_metrics(model_name: str, metrics: dict, plot_confusion: bool = True) -> None:
    """
    Merge a model's metrics into reports/metrics.json (keyed by model name,
    overwriting any previous entry for that name) and, by default, save a
    confusion-matrix heatmap to reports/figures/.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics = _load_existing_metrics()
    # classification_report is a formatted string; everything else here is
    # already JSON-serializable (floats, ints, list-of-lists).
    all_metrics[model_name] = metrics

    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)

    if plot_confusion and "confusion_matrix" in metrics:
        plot_confusion_matrix(
            np.array(metrics["confusion_matrix"]),
            title=f"{model_name} — Confusion Matrix",
            save_path=FIGURES_DIR / f"{_slugify(model_name)}_confusion_matrix.png",
        )


def plot_confusion_matrix(
    cm: np.ndarray,
    title: str,
    save_path: pathlib.Path,
    labels: tuple[str, str] = ("Legitimate", "Fraud"),
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar=False,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str,
    save_path: pathlib.Path,
    n_bins: int = 10,
) -> None:
    """
    Reliability diagram: bins predictions by predicted probability and
    plots the actual fraud rate in each bin against the predicted one. A
    well-calibrated probabilistic model (the entire point of the Bayesian
    approach over a hard classifier) should track the diagonal.
    """
    from sklearn.calibration import calibration_curve

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.plot(prob_pred, prob_true, marker="o", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraud rate")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def _slugify(name: str) -> str:
    return name.lower().replace(" ", "_")


def load_all_metrics() -> dict:
    """Read back every model's saved metrics, e.g. for a comparison table."""
    return _load_existing_metrics()
