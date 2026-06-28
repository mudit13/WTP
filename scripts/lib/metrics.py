"""
Evaluation metrics for detection, attribution, out-of-set behavior, and robustness.

Detection (binary real-vs-fake):
    AUROC, AUPRC, balanced accuracy, accuracy, precision, recall, macro-F1.
Attribution (multi-class generator id):
    top-1 accuracy, macro-F1, balanced accuracy, per-class report, confusion matrix.
Out-of-set:
    predictive entropy, false-known rate (forced confident known-class assignment).
Robustness:
    performance drop, confidence drop, label-flip rate.

All functions take numpy arrays and return plain dicts/arrays so results serialize cleanly.
ASCII-only; Python 3.9.
"""
from typing import Dict, List, Optional, Sequence

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def detection_metrics(y_true: Sequence[int],
                      y_pred: Sequence[int],
                      y_score: Optional[Sequence[float]] = None) -> Dict[str, float]:
    """Binary detection metrics. Convention: fake = 1 (positive), real = 0.

    y_score is the probability/confidence of the positive (fake) class; if omitted,
    threshold-free metrics (AUROC/AUPRC) are skipped.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "n": int(len(y_true)),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        y_score = np.asarray(y_score, dtype=float)
        out["auroc"] = float(roc_auc_score(y_true, y_score))
        out["auprc"] = float(average_precision_score(y_true, y_score))
    return out


def attribution_metrics(y_true: Sequence,
                        y_pred: Sequence,
                        labels: Optional[Sequence] = None) -> Dict[str, object]:
    """Multi-class attribution metrics + confusion matrix."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = sorted(set(list(np.unique(y_true)) + list(np.unique(y_pred))))
    labels = list(labels)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = {}
    for idx, lab in enumerate(labels):
        support = int(cm[idx].sum())
        correct = int(cm[idx, idx])
        per_class[str(lab)] = {
            "support": support,
            "recall": float(correct / support) if support else 0.0,
        }
    return {
        "top1_accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "labels": [str(x) for x in labels],
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "n": int(len(y_true)),
    }


def attribution_slice(y_true: Sequence,
                      y_pred: Sequence,
                      labels: Sequence,
                      keep_labels: Sequence,
                      other_label: str = "__other__") -> Dict[str, object]:
    """Score attribution on a SUBSET of classes (a 'slice').

    Rows whose true label is in `keep_labels` are retained; predictions that fall OUTSIDE
    `keep_labels` are folded into a single synthetic `other_label` bucket so out-of-slice
    predictions count as wrong without polluting the per-class report with every excluded
    class. Returns the same shape of dict as attribution_metrics, but the confusion matrix /
    per-class report are over (sorted(keep_labels) + [other_label]).

    Used by the GAN-only attribution headline (keep = GAN classes + reals, diffusion folded
    to 'diffusion_mismatch') and the all-class number keeps using attribution_metrics.
    """
    keep_set = set(str(k) for k in keep_labels)
    y_true = np.asarray([str(x) for x in y_true])
    y_pred = np.asarray([str(x) for x in y_pred])
    mask = np.array([t in keep_set for t in y_true])
    yt = y_true[mask]
    yp = np.where(np.isin(y_pred[mask], list(keep_set)), y_pred[mask], other_label)
    slice_labels = sorted(keep_set)
    if other_label not in slice_labels:
        slice_labels = slice_labels + [other_label]
    # Edge case: no true label is in the keep set -> score an empty slice (avoid sklearn's
    # empty-input crash). Per-class report is empty; accuracies are 0 by convention.
    if len(yt) == 0:
        return {
            "top1_accuracy": 0.0, "macro_f1": 0.0, "balanced_accuracy": 0.0,
            "labels": slice_labels, "confusion_matrix": [[0] * len(slice_labels)] * len(slice_labels),
            "per_class": {str(l): {"support": 0, "recall": 0.0} for l in slice_labels},
            "n": 0,
        }
    return attribution_metrics(yt, yp, labels=slice_labels)


def predictive_entropy(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Shannon entropy (natural log) of each row of a probability matrix.

    Higher entropy -> the classifier is less certain, which we expect on out-of-set
    generators. Returns one value per sample.
    """
    probs = np.asarray(probs, dtype=float)
    probs = np.clip(probs, eps, 1.0)
    return -np.sum(probs * np.log(probs), axis=1)


def false_known_rate(max_confidence: Sequence[float], threshold: float = 0.5) -> float:
    """Fraction of (out-of-set) samples assigned a known class with confidence above
    the threshold. A high value means the closed-set model is confidently wrong on
    unseen generators - the core limitation the supervisors flagged.
    """
    conf = np.asarray(max_confidence, dtype=float)
    if conf.size == 0:
        return 0.0
    return float(np.mean(conf >= threshold))


def label_flip_rate(pred_clean: Sequence, pred_perturbed: Sequence) -> float:
    """Fraction of samples whose predicted label changes after perturbation."""
    a = np.asarray(pred_clean)
    b = np.asarray(pred_perturbed)
    if a.size == 0:
        return 0.0
    return float(np.mean(a != b))


def performance_drop(clean_metric: float, perturbed_metric: float) -> float:
    """Absolute drop in a scalar metric (clean - perturbed)."""
    return float(clean_metric - perturbed_metric)


def confidence_drop(conf_clean: Sequence[float], conf_perturbed: Sequence[float]) -> float:
    """Mean drop in confidence after perturbation."""
    a = np.asarray(conf_clean, dtype=float)
    b = np.asarray(conf_perturbed, dtype=float)
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return float(np.mean(a[:n] - b[:n]))


def save_confusion_matrix(cm: np.ndarray,
                          labels: List[str],
                          png_path: str,
                          csv_path: Optional[str] = None,
                          title: str = "Confusion matrix",
                          normalize: bool = False) -> None:
    """Persist a confusion matrix as a PNG figure and (optionally) a CSV table."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    cm = np.asarray(cm, dtype=float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_display = cm / row_sums
    else:
        cm_display = cm

    if csv_path is not None:
        pd.DataFrame(cm, index=labels, columns=labels).to_csv(csv_path)

    fig, ax = plt.subplots(figsize=(1.4 * len(labels) + 2, 1.4 * len(labels) + 2))
    im = ax.imshow(cm_display, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fmt = ".2f" if normalize else ".0f"
    thresh = cm_display.max() / 2.0 if cm_display.size else 0.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, format(cm_display[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm_display[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
