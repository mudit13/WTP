"""robustness_perturb.score() must compute accuracy/AUROC drop for BOTH DE-FAKE-style
per-image CSVs (schema.LABEL string column) and DCT-style per-image CSVs (dct_svm.py's
dct_per_image.csv, which has no `label` column at all - only a numeric `y_true`). Regression
test for the gap where DCT's drop JSON silently ended up with only n/label_flip_rate."""
import json
from argparse import Namespace

import pandas as pd

import robustness_perturb as rp
from lib import schema


class _StubLogger:
    """score() only calls logger.info once at the end; avoid writing to logs/ in tests."""

    def info(self, *args, **kwargs):
        pass


def _args(tmp_path, clean_csv, pert_csv, pred_col, conf_col):
    return Namespace(
        clean=str(clean_csv), perturbed=str(pert_csv), source_index=None,
        pred_col=pred_col, conf_col=conf_col, out=str(tmp_path / "drop.json"),
    )


def test_score_defake_style_computes_accuracy_and_auroc(tmp_path):
    clean = pd.DataFrame({
        schema.PATH: ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
        schema.LABEL: ["real", "real", "fake", "fake"],
        schema.DEFAKE_PREDICT: [0, 0, 1, 1],
        schema.PROB_FAKE: [0.1, 0.2, 0.8, 0.9],
    })
    pert = clean.copy()
    pert[schema.DEFAKE_PREDICT] = [0, 1, 1, 0]  # b and d flip
    pert[schema.PROB_FAKE] = [0.15, 0.6, 0.7, 0.4]
    clean_csv, pert_csv = tmp_path / "clean.csv", tmp_path / "pert.csv"
    clean.to_csv(clean_csv, index=False)
    pert.to_csv(pert_csv, index=False)

    args = _args(tmp_path, clean_csv, pert_csv, schema.DEFAKE_PREDICT, schema.PROB_FAKE)
    rp.score(args, _StubLogger())
    out = json.loads((tmp_path / "drop.json").read_text())

    assert out["n"] == 4
    assert out["label_flip_rate"] == 0.5
    assert "accuracy_clean" in out and "accuracy_perturbed" in out
    assert out["accuracy_clean"] == 1.0
    assert "auroc_clean" in out and "auroc_perturbed" in out and "auroc_drop" in out


def test_score_dct_style_computes_accuracy_and_auroc(tmp_path):
    """Same scenario, but with dct_svm.py's actual per-image schema: full_path, generator,
    y_true (int, 1=fake), score (SVM decision function), pred (int). No `label` column."""
    clean = pd.DataFrame({
        schema.PATH: ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
        "generator": ["CelebA", "CelebA", "SD1.5", "SD1.5"],
        "y_true": [0, 0, 1, 1],
        "score": [-0.4, -0.2, 0.8, 0.9],
        "pred": [0, 0, 1, 1],
    })
    pert = clean.copy()
    pert["pred"] = [0, 1, 1, 0]  # b and d flip
    pert["score"] = [-0.3, 0.3, 0.5, -0.1]
    clean_csv, pert_csv = tmp_path / "dct_clean.csv", tmp_path / "dct_pert.csv"
    clean.to_csv(clean_csv, index=False)
    pert.to_csv(pert_csv, index=False)

    args = _args(tmp_path, clean_csv, pert_csv, "pred", "score")
    rp.score(args, _StubLogger())
    out = json.loads((tmp_path / "drop.json").read_text())

    assert out["n"] == 4
    assert out["label_flip_rate"] == 0.5
    # This is the regression check: before the fix, label_col never resolved for the DCT
    # schema (no `label` column), so NONE of the keys below were ever written.
    assert "accuracy_clean" in out and "accuracy_perturbed" in out
    assert out["accuracy_clean"] == 1.0
    assert "performance_drop" in out
    assert "auroc_clean" in out and "auroc_perturbed" in out and "auroc_drop" in out
