#!/usr/bin/env python3
"""
Seed sweep for the fine-tuned attribution head: re-split + re-train the MLP head over K seeds
on the SAME cached CLIP features, and report mean/std/95% CI of the in-set metrics.

The head is stochastic (random init + a content-stable split that depends on the seed), so a
single run's "in-set balanced accuracy 0.94 / StyleGAN3 recall 0.82" needs a variance estimate.
CLIP features are extracted ONCE (fixed feature seed); only the split + head init vary per seed,
so this is fast (no CLIP recompute) and isolates head/split variance.

Mirrors the class-space and evaluation logic of finetune_defake_head.py exactly.

Usage:
  $WTP_PY_DEFAKE scripts/seed_sweep.py --config configs/config.yaml \
      --index results/index_aspect.csv \
      --features_cache results/clip_feats_aspect_jpegaug.npz \
      --captions_csv /pitsec_sose26_topic8/dataset/defake_predictions_all.csv \
      --jpeg_aug on --n_seeds 10 --out results/ci/seed_sweep_aspect.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import io_utils, metrics, features_cache, defake_head  # noqa: E402

import numpy as np  # noqa: E402


def _agg(values):
    """mean/std/sem + normal-approx 95% CI over the per-seed values (kept raw for transparency)."""
    a = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if a.size == 0:
        return {"mean": None, "std": None, "values": []}
    mean = float(a.mean())
    std = float(a.std(ddof=1)) if a.size > 1 else 0.0
    sem = std / np.sqrt(a.size) if a.size > 1 else 0.0
    return {"mean": mean, "std": std, "sem": float(sem),
            "ci95_lo": float(mean - 1.96 * sem), "ci95_hi": float(mean + 1.96 * sem),
            "min": float(a.min()), "max": float(a.max()),
            "n_seeds": int(a.size), "values": [float(v) for v in a]}


def main(args):
    logger = io_utils.setup_logging("seed_sweep")
    io_utils.ensure_dir(os.path.dirname(os.path.abspath(args.out)))
    config = io_utils.load_config(args.config)
    feat_seed = int(config.get("seed", 42))

    aug_cfg = config.get("augmentation", {}) or {}
    if args.jpeg_aug == "on":
        jpeg_aug = True
    elif args.jpeg_aug == "off":
        jpeg_aug = False
    else:
        jpeg_aug = bool(aug_cfg.get("jpeg_train", False))
    qrange = tuple(aug_cfg.get("jpeg_quality_range", [30, 100]))

    # Extract/load features ONCE (fixed feature seed) so only split + head init vary.
    X, generator, label, paths = features_cache.build_features(
        args.index, args.features_cache, device=args.device,
        force=args.recompute_features, captions_csv=args.captions_csv,
        jpeg_aug=jpeg_aug, jpeg_quality_range=qrange, seed=feat_seed)

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
    Xi, gi, pi = X[in_mask], generator[in_mask], paths[in_mask]
    y = defake_head.encode_labels(gi, classes)
    logger.info("Class space (%d): %s | in-set rows=%d", len(classes), classes, len(y))

    # Group-aware split (same-source-photo coupling fix, e.g. OpenForensics real+fake crop
    # pairs); see finetune_defake_head.py for details. No-op when no sidecar is found. Loaded
    # once outside the seed loop (the sidecar does not depend on the split seed).
    group_map_paths = args.group_map if args.group_map else io_utils.default_group_map_paths(config)
    group_map = io_utils.load_group_map(group_map_paths, logger)
    groups = io_utils.apply_group_map(pi, group_map, logger=logger) if group_map else None

    seeds = [args.base_seed + i for i in range(args.n_seeds)]
    top1, macro_f1, bal = [], [], []
    per_class = {c: [] for c in classes}
    for s in seeds:
        tr, va, te = defake_head.stratified_split(
            y, test_size=config.get("test_size", 0.2),
            val_size=config.get("val_size", 0.1), seed=s, keys=pi, groups=groups)
        cw = defake_head.compute_class_weights(y[tr], len(classes))
        head = defake_head._MLPHead(in_dim=Xi.shape[1], num_classes=len(classes),
                                    device=args.device, seed=s)
        head.fit(Xi[tr], y[tr], Xi[va], y[va], epochs=args.epochs, lr=args.lr,
                 class_weights=cw, logger=None)
        pred = head.predict_proba(Xi[te]).argmax(axis=1)
        yt = [classes[i] for i in y[te]]
        yp = [classes[i] for i in pred]
        res = metrics.attribution_metrics(yt, yp, classes)
        top1.append(res["top1_accuracy"])
        macro_f1.append(res["macro_f1"])
        bal.append(res["balanced_accuracy"])
        for c in classes:
            pc = res["per_class"].get(c)
            per_class[c].append(pc["recall"] if pc and pc["support"] > 0 else None)
        logger.info("seed=%d balAcc=%.3f top1=%.3f", s, res["balanced_accuracy"],
                    res["top1_accuracy"])

    out = {
        "seeds": seeds, "classes": classes, "n_in_set": int(len(y)),
        "top1_accuracy": _agg(top1), "macro_f1": _agg(macro_f1), "balanced_accuracy": _agg(bal),
        "per_class_recall": {c: _agg(v) for c, v in per_class.items()},
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    logger.info("balAcc mean=%.3f std=%.3f over %d seeds -> %s",
                out["balanced_accuracy"]["mean"], out["balanced_accuracy"]["std"],
                len(seeds), args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed sweep for the fine-tuned head.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--features_cache", default=None)
    parser.add_argument("--captions_csv", default=None)
    parser.add_argument("--classes", nargs="*", default=None)
    parser.add_argument("--group_map", nargs="*", default=None,
                        help="Path(s) to full_path,source_image_id sidecar CSV(s) for "
                             "group-aware splitting. Default: auto-load "
                             "<dataset_root>/openforensics/openforensics_groups.csv if present.")
    parser.add_argument("--jpeg_aug", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--recompute_features", action="store_true")
    parser.add_argument("--out", required=True)
    main(parser.parse_args())
