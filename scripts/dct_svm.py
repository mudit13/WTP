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
    y = (label == "fake").astype(int)  # fake = positive = 1
    return X, y, label, generator, dataset


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
    return clf, result


def main(args):
    logger = io_utils.setup_logging("dct_svm")
    io_utils.ensure_dir(args.out_dir)
    X, y, _, generator, _ = _load(args.features)
    logger.info("Loaded X=%s, positives(fake)=%d, negatives(real)=%d",
                X.shape, int(y.sum()), int((1 - y).sum()))

    summary = {"mode": args.mode, "n_total": int(len(y))}

    if args.mode == "random":
        from sklearn.model_selection import train_test_split
        idx = np.arange(len(y))
        tr, te = train_test_split(idx, test_size=args.test_size,
                                  stratify=y, random_state=args.seed)
        clf, result = _fit_eval(X[tr], y[tr], X[te], y[te], args.seed)
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
        clf, result = _fit_eval(X[train_mask], y[train_mask],
                                X[test_mask], y[test_mask], args.seed)
        summary["holdout_generators"] = sorted(held)
        summary["test"] = result
        logger.info("Out-of-set test metrics: %s", json.dumps(result))
    else:
        raise SystemExit("Unknown mode: %s" % args.mode)

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
