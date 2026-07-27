"""Metrics behave sanely on tiny toy arrays (no model/data needed)."""
import numpy as np
import pandas as pd
import pytest

import bootstrap_metrics
from lib import metrics


def test_detection_perfect_with_scores():
    m = metrics.detection_metrics([0, 0, 1, 1], [0, 0, 1, 1], [0.1, 0.2, 0.9, 0.8])
    assert m["accuracy"] == 1.0
    assert m["auroc"] == 1.0
    assert m["auprc"] == 1.0
    assert m["cohen_kappa"] == 1.0
    assert m["n"] == 4


def test_detection_without_scores_skips_threshold_free():
    m = metrics.detection_metrics([0, 1], [0, 0])
    assert "auroc" not in m
    assert m["n"] == 2


def test_attribution_basic():
    m = metrics.attribution_metrics(["A", "A", "B", "B"], ["A", "B", "B", "B"])
    assert m["n"] == 4
    assert 0.0 <= m["top1_accuracy"] <= 1.0
    assert set(m["labels"]) == {"A", "B"}
    assert len(m["confusion_matrix"]) == 2
    assert m["cohen_kappa"] == pytest.approx(0.5)


def test_predictive_entropy_orders_certainty():
    ent = metrics.predictive_entropy(np.array([[0.5, 0.5], [1.0, 0.0]]))
    assert ent[0] > ent[1]


def test_false_known_rate():
    assert metrics.false_known_rate([0.9, 0.2, 0.6], 0.5) == 2 / 3
    assert metrics.false_known_rate([], 0.5) == 0.0


def test_label_flip_rate():
    assert metrics.label_flip_rate(["a", "b", "c"], ["a", "x", "c"]) == 1 / 3


def test_performance_and_confidence_drop():
    assert metrics.performance_drop(0.9, 0.7) == pytest.approx(0.2)
    assert metrics.confidence_drop([0.8, 0.9], [0.6, 0.5]) == pytest.approx(0.3)


def test_dct_schema_bootstrap_includes_kappa():
    class Logger:
        def info(self, *args, **kwargs):
            pass

    df = pd.DataFrame({
        "y_true": [0, 0, 1, 1],
        "pred": [0, 1, 1, 1],
        "score": [-1.0, 0.2, 0.5, 1.0],
        "generator": ["R", "R", "F", "F"],
    })
    assert bootstrap_metrics._detect_mode(df, "auto") == "detection"
    result = bootstrap_metrics.bootstrap_detection(df, 20, 42, Logger())
    assert "cohen_kappa" in result["overall"]
    assert result["overall"]["cohen_kappa"]["point"] == pytest.approx(0.5)
