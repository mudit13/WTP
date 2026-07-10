#!/usr/bin/env python3
"""
Leave-one-generator-out (LOGO): the strict out-of-set generalization test.

For each target generator, train the DE-FAKE head on ALL other classes (real + remaining
generators) and then test on the held-out generator's images. Because the held-out class is
absent from training, the closed-set head MUST force every held-out image into a known label.
We record how those forced labels distribute and how confident they are - the central
research question the supervisors emphasized.

NAMING CAVEAT: this script is GENERIC (any generator in the index can be a --targets value,
including a REAL class like "CelebA"), but --targets defaults to only the two
finetune_new_classes (FLUX.1-schnell, StyleGAN3-FFHQ) - generators the regular head IS trained
on. That default run is more accurately "leave-new-class-out": it tests whether the two most
recently added classes could be dropped and still be recognised, not a full leave-one-out sweep
over every trained class. For an actual LOGO report, pass --all_trained_classes (below) or an
explicit --targets list covering every real + fake class the head is normally trained on
(e.g. also "CelebA", "London-DB", "FFHQ", "OpenForensics", "SD1.5") so the report can state a
false-known rate averaged over ALL trained classes, not just the two most favorable ones.

Run with the DE-FAKE interpreter (venv_sd15 on the server):
  $WTP_PY_DEFAKE scripts/leave_one_generator_out.py --config configs/config.yaml \
      --index results/index_scaled.csv --out_dir results/logo_scaled/ \
      --features_cache results/clip_feats_scaled.npz \
      --targets "FLUX.1-schnell" "StyleGAN3-FFHQ"

  # Proper LOGO: hold out EVERY trained class in turn (reals included), one flag:
  $WTP_PY_DEFAKE scripts/leave_one_generator_out.py --config configs/config.yaml \
      --index results/index_aspect.csv --out_dir results/logo_full_aspect/ \
      --features_cache results/clip_feats_aspect_jpegaug.npz \
      --captions_csv $WTP_ROOT/dataset/defake_predictions_all.csv --all_trained_classes
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import io_utils, metrics, features_cache, defake_head, schema  # noqa: E402

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
        jpeg_aug=jpeg_aug, jpeg_quality_range=qrange, seed=seed)
    all_generators = sorted(set(generator))
    # Default targets = the TRAINED fake set (in_set + finetune_new_classes): holding one of
    # these out is the real generalization ablation. (A never-trained generator is already
    # "out" by construction - finetune force-scores those.) Override with --targets.
    attr = config.get("attribution", {}) or {}
    if args.all_trained_classes:
        # Proper LOGO: every class the regular head is normally trained on (reals included),
        # not just the two finetune_new_classes. See the module docstring's naming caveat.
        targets = list(dict.fromkeys(
            list(attr.get("real_generators", [])) + list(attr.get("in_set_generators", []))
            + list(attr.get("finetune_new_classes", []))))
        logger.info("--all_trained_classes: LOGO will hold out every trained class in turn "
                    "(%d classes, reals included) -> this is the actual leave-one-*generator*-"
                    "out sweep, not leave-new-class-out.", len(targets))
    else:
        default_targets = list(dict.fromkeys(
            list(attr.get("in_set_generators", [])) + list(attr.get("finetune_new_classes", []))))
        targets = args.targets or default_targets
        if targets == default_targets and not args.targets:
            logger.warning(
                "Using the DEFAULT LOGO targets (%s) = finetune_new_classes only. This is "
                "'leave-NEW-CLASS-out', not a full leave-one-generator-out sweep. Pass "
                "--all_trained_classes (or an explicit --targets covering every trained "
                "class) for the stricter generalization number.", default_targets)
    targets = [t for t in targets if t in all_generators]
    logger.info("Generators present: %s; LOGO targets: %s", all_generators, targets)

    # Group-aware split (same-source-photo coupling fix, e.g. OpenForensics real+fake crop
    # pairs); see finetune_defake_head.py for details. No-op when no sidecar is found.
    group_map_paths = args.group_map if args.group_map else io_utils.default_group_map_paths(config)
    group_map = io_utils.load_group_map(group_map_paths, logger)

    real_generators = set(attr.get("real_generators", []))
    per_image_rows = []
    summary = {}
    for target in targets:
        train_mask = generator != target
        test_mask = generator == target
        if test_mask.sum() == 0:
            logger.warning("No samples for target %s; skipping", target)
            continue
        train_classes = sorted(set(generator[train_mask]))
        y_train = defake_head.encode_labels(generator[train_mask], train_classes)
        X_train = X[train_mask]

        # Carve a content-stable 10% micro-val from the LOGO training data so the head can select
        # its best checkpoint (early stopping), matching finetune_defake_head.py. Without this,
        # best_state is never set and the head keeps the last-epoch weights (fixed-epoch training).
        train_paths = paths[train_mask]
        train_groups = (io_utils.apply_group_map(train_paths, group_map, logger=logger)
                        if group_map else None)
        sub_tr, sub_va, _ = defake_head.stratified_split(
            y_train, test_size=0.0, val_size=0.1, seed=seed, keys=train_paths,
            groups=train_groups)
        cw = defake_head.compute_class_weights(y_train[sub_tr], len(train_classes))
        head = defake_head._MLPHead(in_dim=X.shape[1], num_classes=len(train_classes),
                                    device=args.device, seed=seed)
        head.fit(X_train[sub_tr], y_train[sub_tr], X_train[sub_va], y_train[sub_va],
                 epochs=args.epochs, lr=args.lr, class_weights=cw, logger=logger)

        proba = head.predict_proba(X[test_mask])
        pred_idx = proba.argmax(axis=1)
        pred_names = [train_classes[i] for i in pred_idx]
        conf = proba.max(axis=1)
        ent = metrics.predictive_entropy(proba)

        dist = Counter(pred_names)
        fkr = metrics.false_known_rate(conf, threshold=args.conf_threshold)
        summary[target] = {
            "n_held_out": int(test_mask.sum()),
            "is_real_class": target in real_generators,
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
                        help="Generators to hold out (default: trained fake set = "
                             "in_set_generators + finetune_new_classes = 'leave-new-class-out', "
                             "NOT a full LOGO sweep; see module docstring)")
    parser.add_argument("--all_trained_classes", action="store_true",
                        help="Proper LOGO: hold out EVERY class the head is normally trained "
                             "on in turn (real_generators + in_set_generators + "
                             "finetune_new_classes), overriding --targets.")
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
