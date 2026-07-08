#!/usr/bin/env python3
"""
Bootstrap confidence intervals for detection and attribution metrics.

The per-generator test support in this project is small (~22 images/fake class), so point
estimates like "StyleGAN3 recall 0.82" are fragile. This script attaches 95% bootstrap CIs to
the headline metrics so the report can state uncertainty honestly (GOLD: scientific reliability
over point accuracy).

Stratified bootstrap: each resample draws WITH REPLACEMENT within every true class, preserving
per-class support (keeps both classes present for AUROC and stabilizes tiny attribution classes).

Auto-detects the input:
  - detection    : a predictions CSV with `label` + `defake_predict` (+ optional `prob_fake`).
  - attribution  : a per-image CSV with `true_generator` + `pred_generator` (+ optional `in_set`).

Usage:
  $WTP_PY_DEFAKE scripts/bootstrap_metrics.py \
      --predictions results/defake_detection_aspect_predictions.csv \
      --out results/ci/defake_detection_aspect_ci.json
  $WTP_PY_DEFAKE scripts/bootstrap_metrics.py \
      --predictions results/attr_eval_aspect/attribution_per_image.csv \
      --subset in_set --out results/ci/attr_eval_aspect_ci.json
"""
import argparse
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import io_utils, metrics, schema  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Bootstrap resamples can omit a class present in the fixed label list; sklearn warns benignly.
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")


def _percentile_ci(samples, alpha=0.05):
    """95% percentile CI (+ std) from a list of bootstrap replicates, skipping non-finite."""
    arr = np.asarray([s for s in samples if s is not None and np.isfinite(s)], dtype=float)
    if arr.size == 0:
        return {"lo": None, "hi": None, "std": None, "n_boot_valid": 0}
    return {
        "lo": float(np.percentile(arr, 100 * alpha / 2.0)),
        "hi": float(np.percentile(arr, 100 * (1.0 - alpha / 2.0))),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n_boot_valid": int(arr.size),
    }


def _strat_resample(strat, rng):
    """Indices of a stratified bootstrap resample: within each class, sample len(class) with
    replacement. Preserves per-class support (works on int or string label arrays)."""
    parts = []
    for c in np.unique(strat):
        idx = np.where(strat == c)[0]
        parts.append(rng.choice(idx, size=len(idx), replace=True))
    return np.concatenate(parts)


def bootstrap_detection(df, n_boot, seed, logger):
    y_true = schema.is_fake_label(df[schema.LABEL]).astype(int).to_numpy()
    y_pred = schema.is_fake_predict(df[schema.DEFAKE_PREDICT]).astype(int).to_numpy()
    y_score = None
    if schema.PROB_FAKE in df.columns and df[schema.PROB_FAKE].notna().any():
        y_score = pd.to_numeric(df[schema.PROB_FAKE], errors="coerce").to_numpy()
    rng = np.random.default_rng(seed)

    def _m(idx):
        sc = y_score[idx] if y_score is not None else None
        return metrics.detection_metrics(y_true[idx], y_pred[idx], sc)

    full = _m(np.arange(len(y_true)))
    keys = ["balanced_accuracy", "macro_f1", "accuracy"]
    if "auroc" in full:
        keys += ["auroc", "auprc"]
    dist = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = _strat_resample(y_true, rng)
        m = _m(idx)
        for k in keys:
            dist[k].append(m.get(k))
    overall = {k: dict(point=float(full[k]), **_percentile_ci(dist[k])) for k in keys}
    overall["n"] = int(len(y_true))

    per_gen = {}
    if schema.GENERATOR in df.columns:
        for g, grp in df.groupby(schema.GENERATOR):
            gy = schema.is_fake_label(grp[schema.LABEL]).astype(int).to_numpy()
            gp = schema.is_fake_predict(grp[schema.DEFAKE_PREDICT]).astype(int).to_numpy()
            correct = (gy == gp).astype(int)
            gidx = np.arange(len(correct))
            reps = [float(correct[rng.choice(gidx, size=len(gidx), replace=True)].mean())
                    for _ in range(n_boot)]
            per_gen[str(g)] = {
                "class": "fake" if (len(gy) and gy[0] == 1) else "real",
                "n": int(len(gy)),
                "detection_rate": dict(point=float(correct.mean()), **_percentile_ci(reps)),
            }
    logger.info("Detection CIs: %s", json.dumps({k: overall[k] for k in keys}))
    return {"mode": "detection", "n_boot": n_boot, "seed": seed,
            "overall": overall, "per_generator": per_gen}


def bootstrap_attribution(df, n_boot, seed, subset, true_col, pred_col, logger):
    d = df.copy()
    if "in_set" in d.columns and subset != "all":
        mask = d["in_set"].astype(str).str.lower() == "true"
        d = d[mask if subset == "in_set" else ~mask]
    if len(d) == 0:
        raise SystemExit("No rows after subset=%s; check the in_set column." % subset)
    y_true = d[true_col].astype(str).to_numpy()
    y_pred = d[pred_col].astype(str).to_numpy()
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    rng = np.random.default_rng(seed)

    full = metrics.attribution_metrics(y_true, y_pred, labels=labels)
    keys = ["top1_accuracy", "macro_f1", "balanced_accuracy"]
    dist = {k: [] for k in keys}
    pc_dist = {str(l): [] for l in labels}
    for _ in range(n_boot):
        idx = _strat_resample(y_true, rng)
        m = metrics.attribution_metrics(y_true[idx], y_pred[idx], labels=labels)
        for k in keys:
            dist[k].append(m[k])
        for l in labels:
            pc = m["per_class"].get(str(l))
            pc_dist[str(l)].append(pc["recall"] if pc and pc["support"] > 0 else None)

    overall = {k: dict(point=float(full[k]), **_percentile_ci(dist[k])) for k in keys}
    overall["n"] = int(len(y_true))
    per_class = {}
    for l in labels:
        pc = full["per_class"].get(str(l), {"support": 0, "recall": 0.0})
        per_class[str(l)] = dict(support=int(pc["support"]),
                                 recall=dict(point=float(pc["recall"]), **_percentile_ci(pc_dist[str(l)])))
    logger.info("Attribution CIs (subset=%s): %s", subset,
                json.dumps({k: overall[k] for k in keys}))
    return {"mode": "attribution", "subset": subset, "n_boot": n_boot, "seed": seed,
            "overall": overall, "per_class": per_class}


def _detect_mode(df, forced):
    if forced != "auto":
        return forced
    if schema.DEFAKE_PREDICT in df.columns and schema.LABEL in df.columns:
        return "detection"
    if "true_generator" in df.columns and "pred_generator" in df.columns:
        return "attribution"
    raise SystemExit("Could not auto-detect mode; pass --mode detection|attribution.")


def main(args):
    logger = io_utils.setup_logging("bootstrap_metrics")
    df = pd.read_csv(args.predictions)
    if schema.DEFAKE_PREDICT in df.columns:
        df = df[pd.to_numeric(df[schema.DEFAKE_PREDICT], errors="coerce") != -1].copy()
    mode = _detect_mode(df, args.mode)
    logger.info("Mode=%s rows=%d n_boot=%d", mode, len(df), args.n_boot)

    if mode == "detection":
        out = bootstrap_detection(df, args.n_boot, args.seed, logger)
    else:
        out = bootstrap_attribution(df, args.n_boot, args.seed, args.subset,
                                    args.true_col, args.pred_col, logger)

    io_utils.ensure_dir(os.path.dirname(os.path.abspath(args.out)))
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    logger.info("Wrote %s", args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap 95% CIs for detection/attribution.")
    parser.add_argument("--predictions", required=True,
                        help="Detection predictions CSV or attribution per-image CSV.")
    parser.add_argument("--mode", choices=["auto", "detection", "attribution"], default="auto")
    parser.add_argument("--subset", choices=["in_set", "out_of_set", "all"], default="in_set",
                        help="Attribution only: which population to score (uses the in_set column).")
    parser.add_argument("--true_col", default="true_generator")
    parser.add_argument("--pred_col", default="pred_generator")
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True, help="Output JSON path")
    main(parser.parse_args())
