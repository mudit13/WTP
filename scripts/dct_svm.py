#!/usr/bin/env python3
"""
Binary real-vs-fake detection with a linear-kernel SVM on log-DCT features (Frank2020).

This is the secondary detector that complements DE-FAKE. Per the GOLD review, the SVM is a
LINEAR-kernel 2-class classifier (real vs fake) - documented explicitly to avoid the
"what kind of classifier is this" confusion from the interim meeting.

Modes:
  random        : stratified random train/test split (standard in-set sanity check). Pass
                  --test_index to make the held-out set EXACTLY the rows of an existing split
                  CSV (e.g. results/test_index.csv) instead of drawing a fresh internal split -
                  see the leakage note below.
  out_of_set    : hold out one or more generators entirely from training and test only on
                  them (measures detection generalization to unseen generators)
  predict       : load a PREVIOUSLY fitted model (--model dct_svm.joblib) and score an external
                  feature set (e.g. perturbed images for the robustness pipeline). No retrain;
                  writes dct_per_image.csv (full_path,generator,y_true,score,pred) that
                  robustness_perturb.py --mode score consumes (--pred_col pred --conf_col score).

random/out_of_set output metrics JSON + a fitted model (joblib). Uses the SVM decision function
as the "fake" score for AUROC/AUPRC.

LEAKAGE NOTE (fixed via --test_index): the robustness pipeline (make_split.py -> test_index.csv)
stratifies its split on the 12-class GENERATOR column, while plain `--mode random` here
stratifies on the BINARY real/fake label. Same seed + same test_size do NOT guarantee the same
partition when the stratification column differs, so a fraction of `test_index.csv` rows can
land in this script's internal TRAIN split. robustness_perturb.py then scores perturbed copies
of those rows with `--mode predict`, which is partly evaluating the SVM on (perturbed) training
data -> an inflated "clean" baseline that every robustness delta in the report is measured
against. Always pass --test_index results/test_index.csv for any DCT-SVM run whose test rows
feed the robustness pipeline, so the SVM's train/test boundary is IDENTICAL to the shared split.

Usage:
  $WTP_PY_DEFAKE scripts/dct_svm.py --features results/dct_features_scaled.npz \
      --out_dir results/dct_svm_scaled/ --mode random
  # Robustness-safe: train/test boundary matches results/{train,test}_index.csv exactly.
  $WTP_PY_DEFAKE scripts/dct_svm.py --features results/dct_features_aspect.npz \
      --out_dir results/dct_svm_aspect/ --mode random --test_index results/test_index.csv
  $WTP_PY_DEFAKE scripts/dct_svm.py --features results/dct_features_scaled.npz \
      --out_dir results/dct_svm_oos/ --mode out_of_set \
      --holdout_generators "FLUX.1-schnell" "StyleGAN3-FFHQ"
  $WTP_PY_DEFAKE scripts/dct_svm.py --mode predict \
      --model results/dct_svm_aspect/dct_svm.joblib \
      --features results/robust/dct_jpeg30.npz --out_dir results/robust/dct_jpeg30/
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import io_utils, metrics, schema  # noqa: E402

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

    if args.mode == "predict":
        if not args.model:
            raise SystemExit("--model (a saved dct_svm.joblib) required for predict mode")
        import joblib
        clf = joblib.load(args.model)
        scores = clf.decision_function(X)
        preds = (scores >= 0).astype(int)
        # Metrics are only meaningful if the external set carries both classes (it may not, e.g.
        # a single-perturbation slice); guard so the per-image CSV is always written.
        if len(np.unique(y)) == 2:
            summary["test"] = metrics.detection_metrics(y, preds, y_score=scores)
            logger.info("Predict metrics: %s", json.dumps(summary["test"]))
        else:
            summary["note"] = "single-class external set; per-image predictions only"
        per_image = _write_per_image(args.out_dir, paths, generator, y, scores, preds)
        logger.info("Wrote per-image predictions to %s", per_image)
        with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        logger.info("Wrote results to %s", args.out_dir)
        return

    if args.mode == "random":
        if args.test_index:
            import pandas as pd
            test_paths = set(pd.read_csv(args.test_index)[schema.PATH].astype(str))
            is_test = np.array([p in test_paths for p in paths])
            n_matched = int(is_test.sum())
            if n_matched == 0:
                raise SystemExit(
                    "--test_index %s matched 0 rows in --features %s; check that both were "
                    "built from the same index/variant." % (args.test_index, args.features))
            te = np.where(is_test)[0]
            tr = np.where(~is_test)[0]
            clf, result, scores, preds = _fit_eval(X[tr], y[tr], X[te], y[te], args.seed)
            te_idx = te
            summary["split_source"] = args.test_index
            summary["n_test_index_rows"] = len(test_paths)
            summary["n_test_matched"] = n_matched
            summary["test"] = result
            logger.info("Fixed split (--test_index, matched %d/%d rows) test metrics: %s",
                        n_matched, len(test_paths), json.dumps(result))
        else:
            from sklearn.model_selection import train_test_split
            idx = np.arange(len(y))
            tr, te = train_test_split(idx, test_size=args.test_size,
                                      stratify=y, random_state=args.seed)
            clf, result, scores, preds = _fit_eval(X[tr], y[tr], X[te], y[te], args.seed)
            te_idx = te
            summary["split_source"] = "internal_binary_stratified_random_split"
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
    parser.add_argument("--mode", choices=["random", "out_of_set", "predict"], default="random")
    parser.add_argument("--model", default=None,
                        help="Saved dct_svm.joblib to load (required for --mode predict).")
    parser.add_argument("--holdout_generators", nargs="*", default=None)
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Ignored when --test_index is given (mode=random only).")
    parser.add_argument("--test_index", default=None,
                        help="mode=random only: CSV (e.g. results/test_index.csv) whose "
                             "full_path rows define the held-out test set exactly, instead of "
                             "an internal binary-stratified split. Use this whenever the DCT "
                             "test set feeds the robustness pipeline (see LEAKAGE NOTE above).")
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
