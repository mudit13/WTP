#!/usr/bin/env python3
"""
Fine-tune the DE-FAKE attribution head on frozen CLIP features.

Pipeline:
  1. Build/cache CLIP image embeddings for an index (master_metadata.csv or a variant index).
  2. Primary mode: exactly eight configured fake generators. Joint mode: the same eight plus
     one source-balanced merged Real class. OpenForensics-fake remains test-only in both.
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
from lib import (attribution_taxonomy, defake_head, features_cache, io_utils,
                 metrics, schema)  # noqa: E402

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

    # Keep validation/test/OOS CLEAN. When JPEG control is enabled, build a companion feature
    # matrix for training rows only; the old pipeline augmented every row before splitting,
    # inadvertently evaluating on augmented test images despite calling this "training-time".
    X, generator, label, paths = features_cache.build_features(
        args.index, args.features_cache, device=args.device,
        force=args.recompute_features, captions_csv=args.captions_csv,
        jpeg_aug=False, jpeg_quality_range=qrange, seed=seed)
    X_fit = X
    if jpeg_aug:
        X_fit, gen_aug, label_aug, paths_aug = features_cache.build_features(
            args.index, features_cache.training_aug_cache_path(args.features_cache),
            device=args.device, force=args.recompute_features,
            captions_csv=args.captions_csv, jpeg_aug=True,
            jpeg_quality_range=qrange, seed=seed)
        if not (np.array_equal(generator, gen_aug)
                and np.array_equal(label, label_aug)
                and np.array_equal(paths, paths_aug)):
            raise SystemExit("Clean and training-augmentation feature rows do not align.")
    logger.info("Features: %s, classes present: %s", X.shape, sorted(set(generator)))

    # Primary = eight fake generators only. Optional joint mode adds ONE merged Real class,
    # sampled deterministically and source-balanced. OpenForensics-fake is OOS in both modes.
    mode = attribution_taxonomy.class_mode(config, args.class_mode)
    real_generators = set(attribution_taxonomy.real_generators(config))
    mapped_generator = attribution_taxonomy.remap_reals(generator, config, mode)
    if args.classes:
        classes = list(dict.fromkeys(args.classes))
        in_mask = np.isin(mapped_generator, classes)
        oos_mask = np.isin(
            generator, attribution_taxonomy.out_of_set_generators(config))
        population = {
            "mode": mode,
            "real_mask": np.isin(generator, list(real_generators)) & in_mask,
            "fake_classes": [c for c in classes
                             if c != attribution_taxonomy.real_class_name(config)],
        }
    else:
        try:
            population = attribution_taxonomy.prepare_population(
                generator, paths, config, mode=mode, seed=seed,
                require_all_fakes=not args.allow_missing_fake_classes)
        except ValueError as exc:
            raise SystemExit(str(exc))
        classes = population["classes"]
        in_mask = population["train_mask"]
        oos_mask = population["oos_mask"]
        mapped_generator = population["mapped_generators"]
    class_set = set(classes)
    if not class_set:
        raise SystemExit("Empty class space; check --classes or config.attribution lists.")
    if np.any(in_mask & oos_mask):
        leaked = sorted(set(generator[in_mask & oos_mask]))
        raise SystemExit("Out-of-set generator(s) entered the training population: %s"
                         % ", ".join(leaked))

    unseen_present = sorted(set(generator[oos_mask]))
    if unseen_present:
        logger.warning("Genuinely-unseen generators in index (NOT trained; force-scored for "
                       "the out-of-set analysis): %s", unseen_present)
    excluded = ~(in_mask | oos_mask)
    logger.info("Class mode=%s; training over %d classes: %s", mode, len(classes), classes)
    logger.info("Population: train=%d oos=%d excluded=%d (merged-real selected=%d)",
                int(in_mask.sum()), int(oos_mask.sum()), int(excluded.sum()),
                int(population["real_mask"].sum()))

    Xi, Xi_fit, gi, pi = (
        X[in_mask], X_fit[in_mask], mapped_generator[in_mask], paths[in_mask])
    y = defake_head.encode_labels(gi, classes)

    # Group-aware split: keep every crop sharing a source (e.g. an OpenForensics source photo's
    # real+fake crop pair) on the SAME split side. Auto-loads openforensics_groups.csv unless
    # --group_map overrides it; an empty map is a no-op (every row falls back to a singleton
    # group = its own path), so this is a no-op for indices with no OpenForensics coupling.
    group_map_paths = args.group_map if args.group_map else io_utils.default_group_map_paths(config)
    group_map = io_utils.load_group_map(group_map_paths, logger)
    # Lookup via source_path when args.index is a variant/perturbed index (its full_path points
    # at a derived file the sidecar never knew about) - see io_utils.load_group_lookup_map.
    lookup_map = io_utils.load_group_lookup_map(args.index)
    groups = (io_utils.apply_group_map_with_lookup(pi, lookup_map, group_map, logger=logger)
             if group_map else None)
    if group_map:
        logger.info("Group-aware split: %d path->group entries loaded from %s",
                    len(group_map), group_map_paths)

    # Content-stable split keyed on full_path (reproducible regardless of row order / dropouts).
    tr, va, te = defake_head.stratified_split(
        y, test_size=config.get("test_size", 0.2),
        val_size=config.get("val_size", 0.1), seed=seed, keys=pi, groups=groups)
    n_checked = defake_head.assert_no_group_straddle(
        groups, {"train": tr, "val": va, "test": te}, keys=pi)
    logger.info("Post-split group assertion passed (%d explicit groups)", n_checked)
    logger.info("Split sizes: train=%d val=%d test=%d", len(tr), len(va), len(te))

    cw = defake_head.compute_class_weights(y[tr], len(classes))
    head = defake_head._MLPHead(in_dim=Xi.shape[1], num_classes=len(classes),
                                device=args.device, seed=seed)
    head.fit(Xi_fit[tr], y[tr], Xi[va], y[va], epochs=args.epochs, lr=args.lr,
             class_weights=cw, logger=logger)

    proba = head.predict_proba(Xi[te])
    pred = proba.argmax(axis=1)
    y_true_names = [classes[i] for i in y[te]]
    y_pred_names = [classes[i] for i in pred]
    res = metrics.attribution_metrics(y_true_names, y_pred_names, classes)
    metrics.save_confusion_matrix(
        np.array(res["confusion_matrix"]),
        attribution_taxonomy.display_names(config, res["labels"]),
        png_path=os.path.join(args.out_dir, "cm_finetuned_test.png"),
        csv_path=os.path.join(args.out_dir, "cm_finetuned_test.csv"),
        title="Fine-tuned DE-FAKE head (test)", normalize=True)
    logger.info("Fine-tuned test: top1=%.3f macroF1=%.3f balAcc=%.3f",
                res["top1_accuracy"], res["macro_f1"], res["balanced_accuracy"])

    with open(os.path.join(args.out_dir, "finetune_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump({"class_mode": mode,
                   "classes": classes,
                   "trained_fake_classes": population["fake_classes"],
                   "merged_real_class": (attribution_taxonomy.real_class_name(config)
                                         if mode == attribution_taxonomy.JOINT else None),
                   "n_train_population": int(in_mask.sum()),
                   "n_oos_population": int(oos_mask.sum()),
                   "n_excluded_population": int(excluded.sum()),
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
        "group_id": (groups[te] if groups is not None else pi[te]),
        "in_set": True,
    })]
    if oos_mask.any():
        proba_u = head.predict_proba(X[oos_mask])
        pred_u = [classes[i] for i in proba_u.argmax(axis=1)]
        rows_out = pd.DataFrame({
            schema.PATH: paths[oos_mask],
            "true_generator": generator[oos_mask],
            "pred_generator": pred_u,
            "confidence": proba_u.max(axis=1),
            "entropy": metrics.predictive_entropy(proba_u),
            "group_id": (
                io_utils.apply_group_map_with_lookup(
                    paths[oos_mask], lookup_map, group_map, logger=logger)
                if group_map else paths[oos_mask]),
            "in_set": False,
        })
        rows_out.to_csv(os.path.join(args.out_dir, "finetune_unseen_per_image.csv"), index=False)
        frames.append(rows_out)
    export = pd.concat(frames, ignore_index=True)
    export.to_csv(os.path.join(args.out_dir, "finetune_per_image.csv"), index=False)
    head.save(os.path.join(args.out_dir, "defake_head.pt"), classes)
    logger.info("Saved head + per-image predictions (in-set test=%d, unseen=%d) to %s",
                len(te), int(oos_mask.sum()), args.out_dir)


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
    parser.add_argument("--class_mode", choices=attribution_taxonomy.VALID_MODES, default=None,
                        help="fake_only (primary eight-way) or joint (eight fakes + merged Real). "
                             "Default: attribution.primary_mode from config.")
    parser.add_argument("--allow_missing_fake_classes", action="store_true",
                        help="Development-only: do not fail when a configured fake class is absent.")
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
