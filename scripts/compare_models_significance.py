#!/usr/bin/env python3
"""
Paired significance test: DE-FAKE vs DCT-SVM binary detection on a SHARED test set.

A fair model comparison must be paired (same images), not two independently reported numbers.
DE-FAKE runs on every image; DCT-SVM reports a held-out test split. We align the two on
`full_path` (the DCT test images), then report:

  - McNemar exact test on the discordant pairs (are the error patterns significantly different?)
  - paired bootstrap 95% CI of the AUROC and balanced-accuracy DIFFERENCE (DE-FAKE - DCT)

Inputs:
  --defake : DE-FAKE detection predictions CSV (full_path, label, defake_predict, prob_fake)
  --dct    : dct_per_image.csv from dct_svm.py (full_path, y_true, score, pred)

Note: DE-FAKE is pretrained (never trained on these images) and DCT's test split was held out of
DCT training, so evaluating both on the DCT test paths is fair to each.

Usage:
  $WTP_PY_DEFAKE scripts/compare_models_significance.py \
      --defake results/defake_detection_aspect_predictions.csv \
      --dct results/dct_svm_aspect/dct_per_image.csv \
      --out results/ci/defake_vs_dct_aspect.json
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import io_utils, schema  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, roc_auc_score  # noqa: E402


def _mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value (binomial on discordant pairs, p=0.5).
    b = DE-FAKE correct & DCT wrong; c = DE-FAKE wrong & DCT correct."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return float(min(1.0, 2.0 * tail))


def _auroc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def main(args):
    logger = io_utils.setup_logging("compare_models_significance")
    de = pd.read_csv(args.defake)
    dct = pd.read_csv(args.dct)

    de = de[pd.to_numeric(de[schema.DEFAKE_PREDICT], errors="coerce") != -1].copy()
    de_truth = schema.is_fake_label(de[schema.LABEL]).astype(int)
    de_pred = schema.is_fake_predict(de[schema.DEFAKE_PREDICT]).astype(int)
    de_score = pd.to_numeric(de[schema.PROB_FAKE], errors="coerce")
    de_slim = pd.DataFrame({
        schema.PATH: de[schema.PATH].astype(str),
        "de_truth": de_truth.values, "de_pred": de_pred.values, "de_score": de_score.values,
    })

    dct_slim = pd.DataFrame({
        schema.PATH: dct["full_path"].astype(str),
        "dct_truth": dct["y_true"].astype(int).values,
        "dct_pred": dct["pred"].astype(int).values,
        "dct_score": pd.to_numeric(dct["score"], errors="coerce").values,
    })

    m = de_slim.merge(dct_slim, on=schema.PATH, how="inner").dropna(
        subset=["de_score", "dct_score"])
    if m.empty:
        raise SystemExit("No shared full_path rows between the two prediction files.")
    if not (m["de_truth"].values == m["dct_truth"].values).all():
        logger.warning("Truth mismatch on some shared rows; using DE-FAKE label as ground truth.")
    y = m["de_truth"].to_numpy()
    logger.info("Shared test rows: %d (real=%d fake=%d)",
                len(y), int((y == 0).sum()), int((y == 1).sum()))

    de_correct = (m["de_pred"].to_numpy() == y).astype(int)
    dct_correct = (m["dct_pred"].to_numpy() == y).astype(int)
    b = int(np.sum((de_correct == 1) & (dct_correct == 0)))
    c = int(np.sum((de_correct == 0) & (dct_correct == 1)))
    p_mcnemar = _mcnemar_exact(b, c)

    de_s = m["de_score"].to_numpy()
    dct_s = m["dct_score"].to_numpy()
    auroc_de, auroc_dct = _auroc(y, de_s), _auroc(y, dct_s)
    bal_de = float(balanced_accuracy_score(y, m["de_pred"].to_numpy()))
    bal_dct = float(balanced_accuracy_score(y, m["dct_pred"].to_numpy()))

    # Paired bootstrap of the differences (stratified by truth so both classes stay present).
    rng = np.random.default_rng(args.seed)
    real_idx = np.where(y == 0)[0]
    fake_idx = np.where(y == 1)[0]
    d_auroc, d_bal = [], []
    for _ in range(args.n_boot):
        samp = np.concatenate([rng.choice(real_idx, len(real_idx), replace=True),
                               rng.choice(fake_idx, len(fake_idx), replace=True)])
        yy = y[samp]
        d_auroc.append(_auroc(yy, de_s[samp]) - _auroc(yy, dct_s[samp]))
        d_bal.append(balanced_accuracy_score(yy, m["de_pred"].to_numpy()[samp])
                     - balanced_accuracy_score(yy, m["dct_pred"].to_numpy()[samp]))

    def _ci(vals):
        a = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
        lo, hi = float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))
        # bootstrap two-sided p: 2 * min(P(diff<=0), P(diff>=0))
        p = 2.0 * min(float(np.mean(a <= 0)), float(np.mean(a >= 0)))
        return {"lo": lo, "hi": hi, "p_bootstrap": float(min(1.0, p))}

    out = {
        "n_shared": int(len(y)),
        "mcnemar": {"b_defake_only_correct": b, "c_dct_only_correct": c, "p_value": p_mcnemar},
        "auroc": {"defake": auroc_de, "dct": auroc_dct,
                  "diff_defake_minus_dct": dict(point=auroc_de - auroc_dct, **_ci(d_auroc))},
        "balanced_accuracy": {"defake": bal_de, "dct": bal_dct,
                              "diff_defake_minus_dct": dict(point=bal_de - bal_dct, **_ci(d_bal))},
        "n_boot": args.n_boot, "seed": args.seed,
    }
    io_utils.ensure_dir(os.path.dirname(os.path.abspath(args.out)))
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    logger.info("McNemar p=%.4g | dAUROC=%.3f %s | wrote %s",
                p_mcnemar, out["auroc"]["diff_defake_minus_dct"]["point"],
                _ci(d_auroc), args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paired DE-FAKE vs DCT significance test.")
    parser.add_argument("--defake", required=True, help="DE-FAKE detection predictions CSV")
    parser.add_argument("--dct", required=True, help="dct_per_image.csv from dct_svm.py")
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True, help="Output JSON path")
    main(parser.parse_args())
