#!/usr/bin/env python3
"""
Train the GAN Fingerprints (Yu2019) attribution classifier on residual/spectrum features
(scripts/lib/ganfp.py). The classifier is the existing defake_head._MLPHead; this script is
the GAN-fp analogue of finetune_defake_head.py (which trains the CLIP/DE-FAKE head).

Two data modes:
  --sample_dir <dir>   LOCAL PROTOTYPE. Scans <dir>/<generator>/*.{png,jpg,jpeg}; the folder
                       name is the generator; label = real if the generator is in
                       config.attribution.real_generators, else fake. No master index needed.
  --index <csv>        SERVER / FULL RUN. Reads master_metadata.csv (schema columns); features
                       are cached via ganfp.build_features (--features_cache).

Reports multi-class attribution + secondary binary detection metrics, a confusion matrix, and
per-image test predictions; saves the head as ganfp_head.pt (consumed by run_ganfp_infer.py).

Local CPU prototype:
  python scripts/train_ganfp.py --config configs/config.yaml --sample_dir ganfp_sample \
      --out_dir results/ganfp_local --device cpu

Server (full run, DE-FAKE interpreter venv_sd15):
  $WTP_PY_DEFAKE scripts/train_ganfp.py --config configs/config.yaml \
      --index results/index_scaled.csv --out_dir results/ganfp_scaled \
      --features_cache results/ganfp_feats_scaled.npz
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import io_utils, metrics, defake_head, ganfp, schema  # noqa: E402

import numpy as np  # noqa: E402


def main(args):
    logger = io_utils.setup_logging("train_ganfp")
    io_utils.ensure_dir(args.out_dir)
    config = io_utils.load_config(args.config)
    seed = int(config.get("seed", 42))

    gcfg = config.get("ganfp", {}) or {}
    common_size = int(args.common_size if args.common_size is not None
                      else gcfg.get("common_size", config.get("common_size", 256)))
    feat_size = int(args.feat_size if args.feat_size is not None else gcfg.get("feat_size", 32))
    mode = str(args.mode if args.mode is not None else gcfg.get("mode", "both"))
    epochs = args.epochs if args.epochs is not None else int(gcfg.get("epochs", 40))
    lr = args.lr if args.lr is not None else float(gcfg.get("lr", 1e-3))
    logger.info("ganfp: common_size=%d feat_size=%d mode=%s epochs=%d lr=%g",
                common_size, feat_size, mode, epochs, lr)

    aug_cfg = config.get("augmentation", {}) or {}
    if args.jpeg_aug == "on":
        jpeg_aug = True
    elif args.jpeg_aug == "off":
        jpeg_aug = False
    else:  # auto -> follow config
        jpeg_aug = bool(aug_cfg.get("jpeg_train", False))
    qrange = tuple(aug_cfg.get("jpeg_quality_range", [30, 100]))
    augment = None
    if jpeg_aug:
        from lib import image_ops
        augment = image_ops.make_jpeg_augmenter(qrange, seed)
    logger.info("JPEG augmentation: %s (q %s)", jpeg_aug, list(qrange))

    real_set = set((config.get("attribution", {}) or {}).get("real_generators", []))

    if args.sample_dir:
        paths, generators = ganfp.scan_sample_dir(args.sample_dir)
        labels = [schema.REAL if g in real_set else schema.FAKE for g in generators]
        logger.info("Sample dir: %d images across %d classes", len(paths), len(set(generators)))
        X, generator, label, path_arr = ganfp.features_from_samples(
            paths, generators, labels, common_size, feat_size, mode, augment, logger=logger)
    elif args.index:
        X, generator, label, path_arr = ganfp.build_features(
            args.index, args.features_cache, common_size, feat_size, mode,
            jpeg_aug, qrange, seed, force=args.recompute_features, logger=logger)
    else:
        raise SystemExit("Provide --sample_dir (local prototype) or --index (full run).")
    logger.info("Features: %s, classes present: %s", X.shape, sorted(set(generator)))

    if len(X) == 0:
        raise SystemExit("No features extracted; check --sample_dir/--index paths.")

    # Class space: real + all fake generators present (unless restricted via --classes).
    classes = list(args.classes) if args.classes else sorted(set(generator))
    keep = np.array([g in set(classes) for g in generator])
    X, generator, label, path_arr = X[keep], generator[keep], label[keep], path_arr[keep]
    y = defake_head.encode_labels(generator, classes)
    logger.info("Training over %d classes: %s", len(classes), classes)

    # Content-stable split keyed on full_path (same scheme as finetune_defake_head.py) so the
    # GAN-fp and DE-FAKE test sets are the SAME images -> the benchmark comparison is valid.
    tr, va, te = defake_head.stratified_split(
        y, test_size=config.get("test_size", 0.2),
        val_size=config.get("val_size", 0.1), seed=seed, keys=path_arr)
    logger.info("Split sizes: train=%d val=%d test=%d", len(tr), len(va), len(te))

    cw = defake_head.compute_class_weights(y[tr], len(classes))
    head = defake_head._MLPHead(in_dim=X.shape[1], num_classes=len(classes),
                                device=args.device, seed=seed)
    head.fit(X[tr], y[tr], X[va], y[va], epochs=epochs, lr=lr,
             class_weights=cw, logger=logger)

    # Multi-class attribution on the test split.
    proba = head.predict_proba(X[te])
    pred = proba.argmax(axis=1)
    y_true_names = [classes[i] for i in y[te]]
    y_pred_names = [classes[i] for i in pred]
    res = metrics.attribution_metrics(y_true_names, y_pred_names, classes)
    metrics.save_confusion_matrix(
        np.array(res["confusion_matrix"]), res["labels"],
        png_path=os.path.join(args.out_dir, "cm_ganfp_test.png"),
        csv_path=os.path.join(args.out_dir, "cm_ganfp_test.csv"),
        title="GAN-fp attribution (test)", normalize=True)
    logger.info("GAN-fp test attribution: top1=%.3f macroF1=%.3f balAcc=%.3f",
                res["top1_accuracy"], res["macro_f1"], res["balanced_accuracy"])

    # Secondary binary detection (real vs fake): fake = any non-real class.
    fake_idx = [i for i, c in enumerate(classes) if c not in real_set]
    y_true_bin = np.array([1 if t not in real_set else 0 for t in y_true_names])
    y_pred_bin = np.array([1 if p not in real_set else 0 for p in y_pred_names])
    y_score = proba[:, fake_idx].sum(axis=1) if fake_idx else None
    det = metrics.detection_metrics(y_true_bin, y_pred_bin, y_score=y_score)
    logger.info("GAN-fp test detection: balAcc=%.3f auroc=%s",
                det["balanced_accuracy"], det.get("auroc"))

    with open(os.path.join(args.out_dir, "ganfp_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "data_mode": "sample_dir" if args.sample_dir else "index",
            "classes": classes,
            "feat_dim": int(X.shape[1]),
            "common_size": common_size,
            "feat_size": feat_size,
            "mode": mode,
            "test_attribution": res,
            "test_detection": det,
        }, fh, indent=2)

    # Per-image export (confidence + entropy) for out-of-set analysis / eval_defake_attribution.
    import pandas as pd  # noqa: E402
    ent = metrics.predictive_entropy(proba)
    pd.DataFrame({
        schema.PATH: path_arr[te],
        "true_generator": y_true_names,
        "pred_generator": y_pred_names,
        "confidence": proba.max(axis=1),
        "entropy": ent,
    }).to_csv(os.path.join(args.out_dir, "ganfp_per_image.csv"), index=False)

    head.save(os.path.join(args.out_dir, "ganfp_head.pt"), classes)
    logger.info("Saved head + metrics + per-image to %s", args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the GAN-fp attribution head.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--index", default=None,
                        help="Index CSV (full_path,generator,label,...) - full run mode")
    parser.add_argument("--sample_dir", default=None,
                        help="Local prototype: <dir>/<generator>/* image folders")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--features_cache", default=None, help="GAN-fp feature .npz cache path")
    parser.add_argument("--classes", nargs="*", default=None,
                        help="Restrict to these generator classes (default: all present)")
    parser.add_argument("--jpeg_aug", choices=["auto", "on", "off"], default="off",
                        help="JPEG-augment features (auto = use config.augmentation.jpeg_train)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override config.ganfp.epochs (default 40)")
    parser.add_argument("--lr", type=float, default=None, help="Override config.ganfp.lr")
    parser.add_argument("--common_size", type=int, default=None,
                        help="Override config.ganfp.common_size")
    parser.add_argument("--feat_size", type=int, default=None,
                        help="Override config.ganfp.feat_size (lower = less overfitting)")
    parser.add_argument("--mode", choices=["residual", "spectrum", "both"], default=None,
                        help="Override config.ganfp.mode")
    parser.add_argument("--device", default="cpu",
                        help="torch device (cpu for local; cuda on the server)")
    parser.add_argument("--recompute_features", action="store_true")
    main(parser.parse_args())
