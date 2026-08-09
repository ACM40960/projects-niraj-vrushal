import json

import numpy as np
import pytest

import src.evaluation.report_utils as report_utils


@pytest.fixture(autouse=True)
def isolated_reports_dir(tmp_path, monkeypatch):
    """Redirect all report output to a temp dir so tests never touch the
    real reports/ folder."""
    reports_dir = tmp_path / "reports"
    figures_dir = reports_dir / "figures"
    monkeypatch.setattr(report_utils, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(report_utils, "FIGURES_DIR", figures_dir)
    monkeypatch.setattr(report_utils, "METRICS_PATH", reports_dir / "metrics.json")
    return reports_dir, figures_dir


def _dummy_metrics():
    return {
        "precision": 0.95,
        "recall": 0.90,
        "f1": 0.9245,
        "confusion_matrix": [[100, 2], [3, 20]],
        "classification_report": "dummy report text",
    }


def test_save_metrics_writes_json(isolated_reports_dir):
    reports_dir, _ = isolated_reports_dir
    report_utils.save_metrics("Test Model", _dummy_metrics())

    metrics_path = reports_dir / "metrics.json"
    assert metrics_path.exists()
    with open(metrics_path) as f:
        saved = json.load(f)
    assert "Test Model" in saved
    assert saved["Test Model"]["precision"] == 0.95


def test_save_metrics_merges_multiple_models(isolated_reports_dir):
    report_utils.save_metrics("Model A", _dummy_metrics())
    report_utils.save_metrics("Model B", _dummy_metrics())

    all_metrics = report_utils.load_all_metrics()
    assert "Model A" in all_metrics
    assert "Model B" in all_metrics


def test_save_metrics_creates_confusion_matrix_plot(isolated_reports_dir):
    _, figures_dir = isolated_reports_dir
    report_utils.save_metrics("Test Model", _dummy_metrics())

    plot_path = figures_dir / "test_model_confusion_matrix.png"
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0


def test_plot_confusion_matrix_runs_without_error(tmp_path):
    cm = np.array([[500, 3], [2, 40]])
    save_path = tmp_path / "cm.png"
    report_utils.plot_confusion_matrix(cm, title="Test", save_path=save_path)
    assert save_path.exists()


def test_load_all_metrics_returns_empty_dict_when_no_file(isolated_reports_dir):
    assert report_utils.load_all_metrics() == {}
