"""Metrics behave sanely on tiny toy arrays (no model/data needed)."""
import numpy as np
import pytest

from lib import metrics


def test_detection_perfect_with_scores():
    m = metrics.detection_metrics([0, 0, 1, 1], [0, 0, 1, 1], [0.1, 0.2, 0.9, 0.8])
    assert m["accuracy"] == 1.0
    assert m["auroc"] == 1.0
    assert m["auprc"] == 1.0
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
