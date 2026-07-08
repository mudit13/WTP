#!/usr/bin/env python3
"""
Leave-one-generator-out (LOGO): the strict out-of-set generalization test.

For each target generator, train the DE-FAKE head on ALL other classes (real + remaining
generators) and then test on the held-out generator's images. Because the held-out class is
absent from training, the closed-set head MUST force every held-out image into a known label.
We record how those forced labels distribute and how confident they are - the central
research question the supervisors emphasized.

Run with the DE-FAKE interpreter (venv_sd15 on the server):
  $WTP_PY_DEFAKE scripts/leave_one_generator_out.py --config configs/config.yaml \
      --index results/index_scaled.csv --out_dir results/logo_scaled/ \
      --features_cache results/clip_feats_scaled.npz \
      --targets "FLUX.1-schnell" "StyleGAN3-FFHQ"
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
    default_targets = list(dict.fromkeys(
        list(attr.get("in_set_generators", [])) + list(attr.get("finetune_new_classes", []))))
    targets = args.targets or default_targets
    targets = [t for t in targets if t in all_generators]
    logger.info("Generators present: %s; LOGO targets: %s", all_generators, targets)

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
        sub_tr, sub_va, _ = defake_head.stratified_split(
            y_train, test_size=0.0, val_size=0.1, seed=seed, keys=paths[train_mask])
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
                             "in_set_generators + finetune_new_classes)")
    parser.add_argument("--jpeg_aug", choices=["auto", "on", "off"], default="auto",
                        help="JPEG-augment features (auto = use config.augmentation.jpeg_train)")
    parser.add_argument("--conf_threshold", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--recompute_features", action="store_true")
    main(parser.parse_args())
