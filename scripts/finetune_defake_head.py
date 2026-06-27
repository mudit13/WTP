#!/usr/bin/env python3
"""
Phase E: fine-tune the DE-FAKE attribution head on frozen CLIP features to ADD the
out-of-set generators (FLUX, StyleGAN3) as proper classes, instead of forcing them into
the pretrained label space.

Pipeline:
  1. Build/cache CLIP image embeddings for an index (master_metadata.csv or a variant index).
  2. Define the class space (generators present, optionally restricted via config).
  3. Stratified train/val/test split; train a small MLP head (CLIP frozen).
  4. Report attribution metrics + confusion matrix on the test split.
  5. Export per-image test predictions + softmax confidence for the out-of-set analysis.

Run with the DE-FAKE interpreter (venv_sd15 on the server: CLIP + torch live there):
  $WTP_PY_DEFAKE scripts/finetune_defake_head.py --config configs/config.yaml \
      --index results/index_scaled.csv --out_dir results/finetune_scaled/ \
      --features_cache results/clip_feats_scaled.npz \
      --captions_csv /pitsec_sose26_topic8/dataset/defake_predictions_all.csv
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import io_utils, metrics, features_cache, defake_head, schema  # noqa: E402

import numpy as np  # noqa: E402


def main(args):
    logger = io_utils.setup_logging("finetune_defake_head")
    io_utils.ensure_dir(args.out_dir)
    config = io_utils.load_config(args.config)
    seed = int(config.get("seed", 42))

    aug_cfg = config.get("augmentation", {}) or {}
    if args.jpeg_aug == "on":
        jpeg_aug = True
    elif args.jpeg_aug == "off":
        jpeg_aug = False
    else:  # auto -> follow config
        jpeg_aug = bool(aug_cfg.get("jpeg_train", False))
    qrange = tuple(aug_cfg.get("jpeg_quality_range", [30, 100]))
    logger.info("JPEG augmentation: %s (q %s)", jpeg_aug, list(qrange))

    X, generator, label, paths = features_cache.build_features(
        args.index, args.features_cache, device=args.device,
        force=args.recompute_features, captions_csv=args.captions_csv,
        jpeg_aug=jpeg_aug, jpeg_quality_range=qrange, seed=seed)
    logger.info("Features: %s, classes present: %s", X.shape, sorted(set(generator)))

    # Class space: real + all fake generators present (unless restricted via --classes).
    if args.classes:
        classes = list(args.classes)
    else:
        classes = sorted(set(generator))
    keep = np.array([g in set(classes) for g in generator])
    X, generator, label, paths = X[keep], generator[keep], label[keep], paths[keep]
    y = defake_head.encode_labels(generator, classes)
    logger.info("Training over %d classes: %s", len(classes), classes)

    tr, va, te = defake_head.stratified_split(
        y, test_size=config.get("test_size", 0.2),
        val_size=config.get("val_size", 0.1), seed=seed)
    logger.info("Split sizes: train=%d val=%d test=%d", len(tr), len(va), len(te))

    cw = defake_head.compute_class_weights(y[tr], len(classes))
    head = defake_head._MLPHead(in_dim=X.shape[1], num_classes=len(classes),
                                device=args.device, seed=seed)
    head.fit(X[tr], y[tr], X[va], y[va], epochs=args.epochs, lr=args.lr,
             class_weights=cw, logger=logger)

    proba = head.predict_proba(X[te])
    pred = proba.argmax(axis=1)
    y_true_names = [classes[i] for i in y[te]]
    y_pred_names = [classes[i] for i in pred]
    res = metrics.attribution_metrics(y_true_names, y_pred_names, classes)
    metrics.save_confusion_matrix(
        np.array(res["confusion_matrix"]), res["labels"],
        png_path=os.path.join(args.out_dir, "cm_finetuned_test.png"),
        csv_path=os.path.join(args.out_dir, "cm_finetuned_test.csv"),
        title="Fine-tuned DE-FAKE head (test)", normalize=True)
    logger.info("Fine-tuned test: top1=%.3f macroF1=%.3f balAcc=%.3f",
                res["top1_accuracy"], res["macro_f1"], res["balanced_accuracy"])

    with open(os.path.join(args.out_dir, "finetune_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump({"classes": classes, "test": res}, fh, indent=2)

    # Per-image export (with confidence + entropy) for the out-of-set analysis.
    import pandas as pd
    ent = metrics.predictive_entropy(proba)
    export = pd.DataFrame({
        schema.PATH: paths[te],
        "true_generator": y_true_names,
        "pred_generator": y_pred_names,
        "confidence": proba.max(axis=1),
        "entropy": ent,
    })
    export.to_csv(os.path.join(args.out_dir, "finetune_per_image.csv"), index=False)
    head.save(os.path.join(args.out_dir, "defake_head.pt"), classes)
    logger.info("Saved head + per-image predictions to %s", args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DE-FAKE head on CLIP features.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--index", required=True, help="Index CSV with image_path,generator,...")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--features_cache", default=None, help="CLIP feature .npz cache path")
    parser.add_argument("--captions_csv", default=None,
                        help="Optional predictions CSV (full_path,blip_caption) for faithful "
                             "1024-dim image+text features")
    parser.add_argument("--classes", nargs="*", default=None,
                        help="Restrict to these generator classes (default: all present)")
    parser.add_argument("--jpeg_aug", choices=["auto", "on", "off"], default="auto",
                        help="JPEG-augment features (auto = use config.augmentation.jpeg_train)")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--recompute_features", action="store_true")
    main(parser.parse_args())
