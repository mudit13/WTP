#!/usr/bin/env python3
"""
Binary real-vs-fake detection with a linear-kernel SVM on log-DCT features (Frank2020).

This is the secondary detector that complements DE-FAKE. Per the GOLD review, the SVM is a
LINEAR-kernel 2-class classifier (real vs fake) - documented explicitly to avoid the
"what kind of classifier is this" confusion from the interim meeting.

Two evaluation modes:
  random        : stratified random train/test split (standard in-set sanity check)
  out_of_set    : hold out one or more generators entirely from training and test only on
                  them (measures detection generalization to unseen generators)

Outputs metrics JSON + a fitted model (joblib). Uses the SVM decision function as the
"fake" score for AUROC/AUPRC.

Usage:
  $WTP_PY_DEFAKE scripts/dct_svm.py --features results/dct_features_scaled.npz \
      --out_dir results/dct_svm_scaled/ --mode random
  $WTP_PY_DEFAKE scripts/dct_svm.py --features results/dct_features_scaled.npz \
      --out_dir results/dct_svm_oos/ --mode out_of_set \
      --holdout_generators "FLUX.1-schnell" "StyleGAN3-FFHQ"
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import io_utils, metrics  # noqa: E402

import numpy as np  # noqa: E402


def _load(features_path):
    data = np.load(features_path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    label = data["label"].astype(str)
    generator = data["generator"].astype(str)
    dataset = data["dataset"].astype(str)
    # paths present in features written by dct_extract_features.py; sentinel if an older cache.
    paths = data["paths"].astype(str) if "paths" in data else np.array([""] * len(label))
    y = (label == "fake").astype(int)  # fake = positive = 1
    return X, y, label, generator, dataset, paths


def _fit_eval(X_tr, y_tr, X_te, y_te, seed):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    clf = make_pipeline(
        StandardScaler(with_mean=True),
        LinearSVC(C=1.0, class_weight="balanced", random_state=seed, max_iter=5000),
    )
    clf.fit(X_tr, y_tr)
    scores = clf.decision_function(X_te)
    preds = (scores >= 0).astype(int)
    result = metrics.detection_metrics(y_te, preds, y_score=scores)
    return clf, result, scores, preds


def _write_per_image(out_dir, paths, generator, y_true, scores, preds):
    """Per-image test predictions, so a paired DE-FAKE-vs-DCT significance test can align on
    full_path. Columns: full_path, generator, y_true(1=fake), score(SVM decision), pred."""
    out_csv = os.path.join(out_dir, "dct_per_image.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["full_path", "generator", "y_true", "score", "pred"])
        for p, g, yt, sc, pr in zip(paths, generator, y_true, scores, preds):
            w.writerow([p, g, int(yt), float(sc), int(pr)])
    return out_csv


def main(args):
    logger = io_utils.setup_logging("dct_svm")
    io_utils.ensure_dir(args.out_dir)
    X, y, _, generator, _, paths = _load(args.features)
    logger.info("Loaded X=%s, positives(fake)=%d, negatives(real)=%d",
                X.shape, int(y.sum()), int((1 - y).sum()))

    summary = {"mode": args.mode, "n_total": int(len(y))}
    te_idx = None  # test-set indices, for the per-image dump

    if args.mode == "random":
        from sklearn.model_selection import train_test_split
        idx = np.arange(len(y))
        tr, te = train_test_split(idx, test_size=args.test_size,
                                  stratify=y, random_state=args.seed)
        clf, result, scores, preds = _fit_eval(X[tr], y[tr], X[te], y[te], args.seed)
        te_idx = te
        summary["test"] = result
        logger.info("Random split test metrics: %s", json.dumps(result))

    elif args.mode == "out_of_set":
        if not args.holdout_generators:
            raise SystemExit("--holdout_generators required for out_of_set mode")
        held = set(args.holdout_generators)
        is_held = np.array([g in held for g in generator])
        # Train on everything not held out (reals + non-held fakes); test on held fakes
        # plus a reserved slice of reals so the test set has both classes.
        train_mask = ~is_held
        # Reserve some reals for testing alongside the held-out fakes.
        real_idx = np.where((y == 0))[0]
        rng = np.random.RandomState(args.seed)
        rng.shuffle(real_idx)
        n_real_test = max(1, int(len(real_idx) * args.test_size))
        real_test = set(real_idx[:n_real_test].tolist())
        train_mask = np.array([train_mask[i] and (i not in real_test)
                               for i in range(len(y))])
        test_mask = np.array([(is_held[i] and y[i] == 1) or (i in real_test)
                              for i in range(len(y))])
        logger.info("Out-of-set: train=%d test=%d (held generators=%s)",
                    int(train_mask.sum()), int(test_mask.sum()), sorted(held))
        clf, result, scores, preds = _fit_eval(X[train_mask], y[train_mask],
                                               X[test_mask], y[test_mask], args.seed)
        te_idx = np.where(test_mask)[0]
        summary["holdout_generators"] = sorted(held)
        summary["test"] = result
        logger.info("Out-of-set test metrics: %s", json.dumps(result))
    else:
        raise SystemExit("Unknown mode: %s" % args.mode)

    if te_idx is not None:
        per_image = _write_per_image(args.out_dir, paths[te_idx], generator[te_idx],
                                     y[te_idx], scores, preds)
        logger.info("Wrote per-image test predictions to %s", per_image)

    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    try:
        import joblib
        joblib.dump(clf, os.path.join(args.out_dir, "dct_svm.joblib"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not save model: %s", exc)
    logger.info("Wrote results to %s", args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear-SVM real/fake detector on DCT.")
    parser.add_argument("--features", required=True, help=".npz from dct_extract_features.py")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--mode", choices=["random", "out_of_set"], default="random")
    parser.add_argument("--holdout_generators", nargs="*", default=None)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
