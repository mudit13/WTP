#!/usr/bin/env python3
"""
Phase E: fine-tune the DE-FAKE attribution head on frozen CLIP features to ADD the
out-of-set generators (FLUX, StyleGAN3) as proper classes, instead of forcing them into
the pretrained label space.

Pipeline:
  1. Build/cache CLIP image embeddings for an index (master_metadata.csv or a variant index).
  2. Define the TRAINED class space from config = reals present + (in_set_generators UNION
     finetune_new_classes). Generators outside that set are treated as genuinely UNSEEN: not
     trained, only force-scored later. (Override the whole set with --classes.)
  3. Content-stable (path-hashed) stratified train/val/test split; train a small MLP head
     (CLIP frozen), selecting the checkpoint by BALANCED val accuracy.
  4. Report attribution metrics + confusion matrix on the held-out test split.
  5. Export per-image predictions + confidence + entropy for BOTH the in-set test split and
     the force-scored unseen generators (tagged by an `in_set` flag) for the out-of-set analysis.

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

    # Class space. Default = reals present + the configured TRAINED fake set
    # (attribution.in_set_generators UNION attribution.finetune_new_classes). Any other
    # generator in the index is treated as GENUINELY UNSEEN: it is NOT trained, only
    # force-scored afterwards for the out-of-set analysis. This keeps the in-set/out-of-set
    # contract honest instead of silently turning every generator present into a trained class.
    attr = config.get("attribution", {}) or {}
    real_generators = set(attr.get("real_generators", []))
    present = set(generator)
    if args.classes:
        classes = sorted(set(args.classes))
    else:
        trained_fakes = list(dict.fromkeys(
            list(attr.get("in_set_generators", [])) + list(attr.get("finetune_new_classes", []))))
        allowed = real_generators | set(trained_fakes)
        classes = sorted(g for g in present if g in allowed)
    class_set = set(classes)
    if not class_set:
        raise SystemExit("Empty class space; check --classes or config.attribution lists.")

    in_mask = np.array([g in class_set for g in generator])
    unseen_mask = ~in_mask
    unseen_present = sorted(set(generator[unseen_mask]))
    if unseen_present:
        logger.warning("Genuinely-unseen generators in index (NOT trained; force-scored for "
                       "the out-of-set analysis): %s", unseen_present)
    logger.info("Training over %d classes: %s", len(classes), classes)

    Xi, gi, pi = X[in_mask], generator[in_mask], paths[in_mask]
    y = defake_head.encode_labels(gi, classes)

    # Group-aware split: keep every crop sharing a source (e.g. an OpenForensics source photo's
    # real+fake crop pair) on the SAME split side. Auto-loads openforensics_groups.csv unless
    # --group_map overrides it; an empty map is a no-op (every row falls back to a singleton
    # group = its own path), so this is a no-op for indices with no OpenForensics coupling.
    group_map_paths = args.group_map if args.group_map else io_utils.default_group_map_paths(config)
    group_map = io_utils.load_group_map(group_map_paths, logger)
    groups = io_utils.apply_group_map(pi, group_map, logger=logger) if group_map else None
    if group_map:
        logger.info("Group-aware split: %d path->group entries loaded from %s",
                    len(group_map), group_map_paths)

    # Content-stable split keyed on full_path (reproducible regardless of row order / dropouts).
    tr, va, te = defake_head.stratified_split(
        y, test_size=config.get("test_size", 0.2),
        val_size=config.get("val_size", 0.1), seed=seed, keys=pi, groups=groups)
    logger.info("Split sizes: train=%d val=%d test=%d", len(tr), len(va), len(te))

    cw = defake_head.compute_class_weights(y[tr], len(classes))
    head = defake_head._MLPHead(in_dim=Xi.shape[1], num_classes=len(classes),
                                device=args.device, seed=seed)
    head.fit(Xi[tr], y[tr], Xi[va], y[va], epochs=args.epochs, lr=args.lr,
             class_weights=cw, logger=logger)

    proba = head.predict_proba(Xi[te])
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
        json.dump({"classes": classes,
                   "trained_fake_classes": sorted(class_set - real_generators),
                   "unseen_generators": unseen_present,
                   "test": res}, fh, indent=2)

    # Per-image export (confidence + entropy). Combine the held-out TEST split (in-set,
    # trained classes) with the force-scored genuinely-unseen rows (out-of-set), tagged by an
    # `in_set` flag, so downstream eval + out-of-set analysis get BOTH populations with correct
    # ground truth (no reliance on a static config list).
    import pandas as pd
    ent = metrics.predictive_entropy(proba)
    frames = [pd.DataFrame({
        schema.PATH: pi[te],
        "true_generator": y_true_names,
        "pred_generator": y_pred_names,
        "confidence": proba.max(axis=1),
        "entropy": ent,
        "in_set": True,
    })]
    if unseen_mask.any():
        proba_u = head.predict_proba(X[unseen_mask])
        pred_u = [classes[i] for i in proba_u.argmax(axis=1)]
        rows_out = pd.DataFrame({
            schema.PATH: paths[unseen_mask],
            "true_generator": generator[unseen_mask],
            "pred_generator": pred_u,
            "confidence": proba_u.max(axis=1),
            "entropy": metrics.predictive_entropy(proba_u),
            "in_set": False,
        })
        rows_out.to_csv(os.path.join(args.out_dir, "finetune_unseen_per_image.csv"), index=False)
        frames.append(rows_out)
    export = pd.concat(frames, ignore_index=True)
    export.to_csv(os.path.join(args.out_dir, "finetune_per_image.csv"), index=False)
    head.save(os.path.join(args.out_dir, "defake_head.pt"), classes)
    logger.info("Saved head + per-image predictions (in-set test=%d, unseen=%d) to %s",
                len(te), int(unseen_mask.sum()), args.out_dir)


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
                        help="Override the trained class space (default: reals + "
                             "in_set_generators + finetune_new_classes from config)")
    parser.add_argument("--group_map", nargs="*", default=None,
                        help="Path(s) to full_path,source_image_id sidecar CSV(s) (e.g. "
                             "openforensics_groups.csv) for group-aware splitting. Default: "
                             "auto-load <dataset_root>/openforensics/openforensics_groups.csv "
                             "if present.")
    parser.add_argument("--jpeg_aug", choices=["auto", "on", "off"], default="auto",
                        help="JPEG-augment features (auto = use config.augmentation.jpeg_train)")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--recompute_features", action="store_true")
    main(parser.parse_args())
