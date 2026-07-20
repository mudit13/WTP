#!/usr/bin/env python3
"""
Leave-one-generator-out (LOGO): the strict out-of-set generalization test.

For each of the eight configured fake generators, train the DE-FAKE head on exactly the other
seven fake classes and test on the held-out generator. Optional joint mode also includes one
merged Real class. OpenForensics-fake and every other nondeclared class are always excluded.
Because the held-out class is absent from training, the closed-set head MUST force every image
into a known label.
We record how those forced labels distribute and how confident they are - the central
research question the supervisors emphasized.

The default target list is the complete configured fake class space. `--all_trained_classes`
only differs in joint mode, where it also holds out the merged Real class as an auxiliary check.

Run with the DE-FAKE interpreter (venv_sd15 on the server):
  $WTP_PY_DEFAKE scripts/leave_one_generator_out.py --config configs/config.yaml \
      --index results/index_scaled.csv --out_dir results/logo_scaled/ \
      --features_cache results/clip_feats_scaled.npz \
      --class_mode fake_only
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import (attribution_taxonomy, defake_head, features_cache, io_utils,
                 metrics, schema)  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def main(args):
    logger = io_utils.setup_logging("leave_one_generator_out")
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
    all_generators = sorted(set(generator))
    mode = attribution_taxonomy.class_mode(config, args.class_mode)
    try:
        population = attribution_taxonomy.prepare_population(
            generator, paths, config, mode=mode, seed=seed,
            require_all_fakes=not args.allow_missing_fake_classes)
    except ValueError as exc:
        raise SystemExit(str(exc))
    mapped_generator = population["mapped_generators"]
    base_classes = population["classes"]
    fake_classes = population["fake_classes"]

    if args.all_trained_classes:
        targets = list(base_classes)
    else:
        targets = args.targets or list(fake_classes)
    targets = [t for t in targets if t in set(mapped_generator.tolist())]
    invalid_targets = [t for t in targets if t not in base_classes]
    if invalid_targets:
        raise SystemExit("LOGO target(s) are outside the declared training space: %s"
                         % ", ".join(invalid_targets))
    logger.info("Class mode=%s; generators present=%s; trainable=%s; LOGO targets=%s",
                mode, all_generators, base_classes, targets)

    # Group-aware split (same-source-photo coupling fix, e.g. OpenForensics real+fake crop
    # pairs); see finetune_defake_head.py for details. No-op when no sidecar is found.
    group_map_paths = args.group_map if args.group_map else io_utils.default_group_map_paths(config)
    group_map = io_utils.load_group_map(group_map_paths, logger)
    # Lookup via source_path when --index is a variant/perturbed index (its full_path points at
    # a derived file the sidecar never knew about) - see io_utils.load_group_lookup_map.
    lookup_map = io_utils.load_group_lookup_map(args.index) if args.index else {}
    groups = (io_utils.apply_group_map_with_lookup(
        paths, lookup_map, group_map, logger=logger) if group_map else paths.copy())

    merged_real = attribution_taxonomy.real_class_name(config)
    per_image_rows = []
    summary = {}
    for target in targets:
        train_mask = population["train_mask"] & (mapped_generator != target)
        test_mask = population["train_mask"] & (mapped_generator == target)
        if test_mask.sum() == 0:
            logger.warning("No samples for target %s; skipping", target)
            continue

        # If a held-out img2img output shares a London identity with a Real training row,
        # exclude that whole identity from the fold's training population. Grouping a validation
        # split alone is insufficient because LOGO's target set is external to that split.
        explicit_target_groups = {
            str(g) for g, p in zip(groups[test_mask], paths[test_mask]) if str(g) != str(p)}
        if explicit_target_groups:
            overlap = train_mask & np.isin(groups, list(explicit_target_groups))
            if overlap.any():
                logger.info("LOGO %s: excluded %d training row(s) sharing held-out groups",
                            target, int(overlap.sum()))
                train_mask &= ~overlap
        n_external_groups = defake_head.assert_no_group_straddle(
            groups,
            {"train_population": np.where(train_mask)[0],
             "heldout": np.where(test_mask)[0]},
            keys=paths)

        expected_classes = [c for c in base_classes if c != target]
        train_classes = [c for c in expected_classes if np.any(mapped_generator[train_mask] == c)]
        missing_train = [c for c in expected_classes if c not in train_classes]
        if missing_train:
            raise SystemExit("LOGO %s missing training class(es): %s"
                             % (target, ", ".join(missing_train)))
        y_train = defake_head.encode_labels(mapped_generator[train_mask], train_classes)
        X_train = X_fit[train_mask]
        X_val_clean = X[train_mask]

        # Carve a content-stable 10% micro-val from the LOGO training data so the head can select
        # its best checkpoint (early stopping), matching finetune_defake_head.py. Without this,
        # best_state is never set and the head keeps the last-epoch weights (fixed-epoch training).
        train_paths = paths[train_mask]
        train_groups = groups[train_mask] if group_map else None
        sub_tr, sub_va, _ = defake_head.stratified_split(
            y_train, test_size=0.0, val_size=0.1, seed=seed, keys=train_paths,
            groups=train_groups)
        n_fit_groups = defake_head.assert_no_group_straddle(
            train_groups, {"train": sub_tr, "val": sub_va}, keys=train_paths)
        logger.info("LOGO %s group assertions passed (external=%d fit=%d)",
                    target, n_external_groups, n_fit_groups)
        cw = defake_head.compute_class_weights(y_train[sub_tr], len(train_classes))
        head = defake_head._MLPHead(in_dim=X.shape[1], num_classes=len(train_classes),
                                    device=args.device, seed=seed)
        head.fit(X_train[sub_tr], y_train[sub_tr], X_val_clean[sub_va], y_train[sub_va],
                 epochs=args.epochs, lr=args.lr, class_weights=cw, logger=logger)

        proba = head.predict_proba(X[test_mask])
        pred_idx = proba.argmax(axis=1)
        pred_names = [train_classes[i] for i in pred_idx]
        conf = proba.max(axis=1)
        ent = metrics.predictive_entropy(proba)

        dist = Counter(pred_names)
        fkr = metrics.false_known_rate(conf, threshold=args.conf_threshold)
        summary[target] = {
            "class_mode": mode,
            "n_held_out": int(test_mask.sum()),
            "n_train": int(train_mask.sum()),
            "is_real_class": target == merged_real,
            "train_classes": train_classes,
            "forced_label_distribution": dict(dist),
            "mean_confidence": float(np.mean(conf)),
            "mean_entropy": float(np.mean(ent)),
            "false_known_rate@%.2f" % args.conf_threshold: fkr,
        }
        logger.info("LOGO %-10s -> forced labels %s | meanConf=%.3f meanEnt=%.3f FKR=%.3f",
                    target, dict(dist), np.mean(conf), np.mean(ent), fkr)

        for p, pn, c, e in zip(paths[test_mask], pred_names, conf, ent):
            per_image_rows.append({"held_out_generator": target, schema.PATH: p,
                                   "pred_generator": pn, "confidence": float(c),
                                   "entropy": float(e)})

    with open(os.path.join(args.out_dir, "logo_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    pd.DataFrame(per_image_rows).to_csv(
        os.path.join(args.out_dir, "logo_per_image.csv"), index=False)
    logger.info("Wrote logo_summary.json and logo_per_image.csv to %s", args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Leave-one-generator-out generalization test.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--features_cache", default=None)
    parser.add_argument("--captions_csv", default=None,
                        help="Optional predictions CSV for faithful 1024-dim image+text features")
    parser.add_argument("--targets", nargs="*", default=None,
                        help="Generators to hold out (default: all eight configured fake classes).")
    parser.add_argument("--all_trained_classes", action="store_true",
                        help="Hold out every class in the selected mode; joint mode also holds "
                             "out the merged Real class. Overrides --targets.")
    parser.add_argument("--class_mode", choices=attribution_taxonomy.VALID_MODES, default=None,
                        help="fake_only (primary eight-fake LOGO) or joint.")
    parser.add_argument("--allow_missing_fake_classes", action="store_true",
                        help="Development-only: skip the fail-fast check for absent fake classes.")
    parser.add_argument("--group_map", nargs="*", default=None,
                        help="Path(s) to full_path,source_image_id sidecar CSV(s) for "
                             "group-aware splitting. Default: auto-load "
                             "<dataset_root>/openforensics/openforensics_groups.csv if present.")
    parser.add_argument("--jpeg_aug", choices=["auto", "on", "off"], default="auto",
                        help="JPEG-augment features (auto = use config.augmentation.jpeg_train)")
    parser.add_argument("--conf_threshold", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--recompute_features", action="store_true")
    main(parser.parse_args())
