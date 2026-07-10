#!/usr/bin/env python3
"""
Train the GAN Fingerprints CNN path (Yu2019-inspired, scripts/lib/ganfp_net.py).

Mirrors scripts/train_ganfp.py (Path A: features+_MLPHead) but trains the end-to-end CNN
(Path B) directly on luminance tensors. The CNN has a FIXED SRM high-pass front-end (faithful
family-level Fridrich-Kodovsky 2012 reconstruction) + 3 VGG conv blocks + GAP + linear (~82K
params at config channels [16,32,64]), designed to train in minutes on CPU over a few hundred
images/class with JPEG augmentation. Yu2019-inspired method (learned-fingerprint idea + an SRM
front-end), not a byte-faithful Yu2019 port.

Two data modes (identical to train_ganfp.py):
  --sample_dir <dir>   LOCAL PROTOTYPE. <dir>/<generator>/*.{png,jpg,jpeg}; folder name is
                       the generator; label = real if generator in real_generators else fake.
  --index <csv>        SERVER / FULL RUN. Reads master_metadata.csv (schema columns).

Reports multi-class attribution + secondary binary detection, a confusion matrix, and
per-image test predictions; saves the head as ganfp_cnn.pt ({state_dict, classes}), plus
ganfp_cnn_metrics.json, cm_ganfp_cnn_test.{png,csv}, ganfp_cnn_per_image.csv.

torch is imported lazily inside lib.ganfp_net; this entry point itself never imports torch
at module top so `python -m compileall -q scripts` is torch-free.

Local CPU prototype:
  python scripts/train_ganfp_cnn.py --config configs/config.yaml \
      --sample_dir ganfp_sample --out_dir results/ganfp_cnn_local --device cpu

Server (full run, DE-FAKE interpreter venv_sd15):
  $WTP_PY_DEFAKE scripts/train_ganfp_cnn.py --config configs/config.yaml \
      --index results/index_scaled.csv --out_dir results/ganfp_cnn_scaled
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import io_utils, metrics, defake_head, ganfp, ganfp_net, schema  # noqa: E402

import numpy as np  # noqa: E402


def main(args):
    logger = io_utils.setup_logging("train_ganfp_cnn")
    io_utils.ensure_dir(args.out_dir)
    config = io_utils.load_config(args.config)
    seed = int(config.get("seed", 42))

    gcfg = config.get("ganfp", {}) or {}
    ccfg = gcfg.get("cnn", {}) or {}
    common_size = int(args.common_size if args.common_size is not None
                      else ccfg.get("input_size", gcfg.get("common_size", 256)))
    epochs = args.epochs if args.epochs is not None else int(ccfg.get("epochs", 30))
    lr = args.lr if args.lr is not None else float(ccfg.get("lr", gcfg.get("lr", 1e-3)))
    weight_decay = float(args.weight_decay if args.weight_decay is not None
                         else ccfg.get("weight_decay", 1e-4))
    channels = list(args.channels) if args.channels is not None \
        else list(ccfg.get("channels", [16, 32, 64]))
    # NB: args.hflip is the STRING "true"/"false" (or None). bool("false") is True, so compare
    # explicitly; only fall back to the (already-boolean) config value when the flag is absent.
    if args.hflip is not None:
        hflip = (args.hflip == "true")
    else:
        hflip = bool(ccfg.get("hflip", True))
    batch_size = int(args.batch_size)
    logger.info("ganfp CNN: common_size=%d epochs=%d lr=%g wd=%g channels=%s hflip=%s",
                common_size, epochs, lr, weight_decay, channels, hflip)

    aug_cfg = config.get("augmentation", {}) or {}
    if args.jpeg_aug == "on":
        jpeg_aug = True
    elif args.jpeg_aug == "off":
        jpeg_aug = False
    else:
        jpeg_aug = bool(aug_cfg.get("jpeg_train", False))
    qrange = tuple(aug_cfg.get("jpeg_quality_range", [30, 100]))
    from lib import image_ops  # noqa: E402
    augment = image_ops.make_jpeg_augmenter(qrange, seed) if jpeg_aug else None
    logger.info("JPEG augmentation: %s (q %s)", jpeg_aug, list(qrange))

    real_set = set((config.get("attribution", {}) or {}).get("real_generators", []))

    # --- gather (paths, generators) the SAME way train_ganfp gathers features ---------
    if args.sample_dir:
        all_paths, generators = ganfp.scan_sample_dir(args.sample_dir)
        labels_bin = [schema.REAL if g in real_set else schema.FAKE for g in generators]
    elif args.index:
        import pandas as pd  # noqa: E402
        df = pd.read_csv(args.index)
        df[schema.PATH] = df[schema.PATH].astype(str)
        all_paths = df[schema.PATH].tolist()
        generators = df[schema.GENERATOR].astype(str).tolist()
        labels_bin = df[schema.LABEL].astype(str).tolist()
    else:
        raise SystemExit("Provide --sample_dir (local prototype) or --index (full run).")
    logger.info("Found %d images across %d classes", len(all_paths), len(set(generators)))
    if not all_paths:
        raise SystemExit("No images found; check --sample_dir/--index paths.")

    classes = list(args.classes) if args.classes else sorted(set(generators))
    keep = [g in set(classes) for g in generators]
    paths = [p for p, k in zip(all_paths, keep) if k]
    generators = [g for g, k in zip(generators, keep) if k]
    labels_bin = [l for l, k in zip(labels_bin, keep) if k]
    y = defake_head.encode_labels(np.array(generators, dtype=object), classes)
    logger.info("Training over %d classes: %s", len(classes), classes)

    # Group-aware split (same-source-photo coupling fix, e.g. OpenForensics real+fake crop
    # pairs); see finetune_defake_head.py for details. No-op when no sidecar is found.
    group_map_paths = args.group_map if args.group_map else io_utils.default_group_map_paths(config)
    group_map = io_utils.load_group_map(group_map_paths, logger)
    paths_arr = np.asarray(paths)
    # Lookup via source_path when --index is a variant/perturbed index (its full_path points at
    # a derived file the sidecar never knew about) - see io_utils.load_group_lookup_map.
    lookup_map = io_utils.load_group_lookup_map(args.index) if args.index else {}
    groups = (io_utils.apply_group_map_with_lookup(paths_arr, lookup_map, group_map, logger=logger)
             if group_map else None)

    # Content-stable split keyed on full_path (same scheme as finetune_defake_head.py) so the
    # GAN-fp and DE-FAKE test sets are the SAME images -> the benchmark comparison is valid.
    tr, va, te = defake_head.stratified_split(
        y, test_size=config.get("test_size", 0.2),
        val_size=config.get("val_size", 0.1), seed=seed, keys=paths_arr, groups=groups)
    logger.info("Split sizes: train=%d val=%d test=%d", len(tr), len(va), len(te))

    labels_int = y.tolist()
    train_paths, train_labels = ganfp_net.slice_paths_labels(paths, labels_int, tr)
    val_paths, val_labels = ganfp_net.slice_paths_labels(paths, labels_int, va)

    train_loader = ganfp_net.build_dataloaders(
        train_paths, train_labels, common_size=common_size, augment=augment,
        hflip=hflip, seed=seed, batch_size=batch_size, num_workers=args.num_workers,
        shuffle=True)
    val_loader = None
    if val_paths:
        val_loader = ganfp_net.build_dataloaders(
            val_paths, val_labels, common_size=common_size, augment=None,
            hflip=False, seed=seed, batch_size=batch_size, num_workers=args.num_workers,
            shuffle=False)

    cw = defake_head.compute_class_weights(y[tr], len(classes))
    clf = ganfp_net.GANFpClassifier(
        num_classes=len(classes), input_size=common_size, channels=channels,
        device=args.device, lr=lr, weight_decay=weight_decay, seed=seed)
    logger.info("CNN trainable params: %d", clf.param_count)
    clf.fit(train_loader, val_loader=val_loader, epochs=epochs,
            class_weights=cw, logger=logger)

    # --- test evaluation on the shared test split ------------------------------------
    test_paths, test_labels = ganfp_net.slice_paths_labels(paths, labels_int, te)
    proba = clf.predict_proba(test_paths, test_labels, common_size=common_size,
                              batch_size=batch_size, num_workers=args.num_workers)
    pred = proba.argmax(axis=1)
    y_true_names = [classes[i] for i in y[te]]
    y_pred_names = [classes[i] for i in pred]
    res = metrics.attribution_metrics(y_true_names, y_pred_names, classes)
    metrics.save_confusion_matrix(
        np.array(res["confusion_matrix"]), res["labels"],
        png_path=os.path.join(args.out_dir, "cm_ganfp_cnn_test.png"),
        csv_path=os.path.join(args.out_dir, "cm_ganfp_cnn_test.csv"),
        title="GAN-fp CNN attribution (test)", normalize=True)
    logger.info("GAN-fp CNN test attribution: top1=%.3f macroF1=%.3f balAcc=%.3f",
                res["top1_accuracy"], res["macro_f1"], res["balanced_accuracy"])

    # Secondary binary detection (real vs fake): fake = any non-real class.
    fake_idx = [i for i, c in enumerate(classes) if c not in real_set]
    y_true_bin = np.array([1 if t not in real_set else 0 for t in y_true_names])
    y_pred_bin = np.array([1 if p not in real_set else 0 for p in y_pred_names])
    y_score = proba[:, fake_idx].sum(axis=1) if fake_idx else None
    det = metrics.detection_metrics(y_true_bin, y_pred_bin, y_score=y_score)
    logger.info("GAN-fp CNN test detection: balAcc=%.3f auroc=%s",
                det["balanced_accuracy"], det.get("auroc"))

    with open(os.path.join(args.out_dir, "ganfp_cnn_metrics.json"), "w",
              encoding="utf-8") as fh:
        json.dump({
            "data_mode": "sample_dir" if args.sample_dir else "index",
            "path": "cnn",
            "classes": classes,
            "common_size": common_size,
            "channels": channels,
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "hflip": hflip,
            "jpeg_aug": jpeg_aug,
            "params": clf.param_count,
            "split": {"seed": seed, "train": len(tr), "val": len(va), "test": len(te)},
            "test_attribution": res,
            "test_detection": det,
        }, fh, indent=2)

    import pandas as pd  # noqa: E402
    ent = metrics.predictive_entropy(proba)
    pd.DataFrame({
        schema.PATH: test_paths,
        "true_generator": y_true_names,
        "pred_generator": y_pred_names,
        "confidence": proba.max(axis=1),
        "entropy": ent,
    }).to_csv(os.path.join(args.out_dir, "ganfp_cnn_per_image.csv"), index=False)

    clf.save(os.path.join(args.out_dir, "ganfp_cnn.pt"), classes)
    logger.info("Saved CNN head + metrics + per-image to %s", args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the GAN-fp CNN attribution head.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--index", default=None,
                        help="Index CSV (full_path,generator,label,...) - full run mode")
    parser.add_argument("--sample_dir", default=None,
                        help="Local prototype: <dir>/<generator>/* image folders")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--classes", nargs="*", default=None,
                        help="Restrict to these generator classes (default: all present)")
    parser.add_argument("--group_map", nargs="*", default=None,
                        help="Path(s) to full_path,source_image_id sidecar CSV(s) for "
                             "group-aware splitting. Default: auto-load "
                             "<dataset_root>/openforensics/openforensics_groups.csv if present.")
    parser.add_argument("--jpeg_aug", choices=["auto", "on", "off"], default="off",
                        help="JPEG-augment inputs (auto = use config.augmentation.jpeg_train)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override config.ganfp.cnn.epochs (default 30)")
    parser.add_argument("--lr", type=float, default=None, help="Override config.ganfp.cnn.lr")
    parser.add_argument("--common_size", type=int, default=None,
                        help="Override config.ganfp.cnn.input_size")
    parser.add_argument("--channels", type=int, nargs="*", default=None,
                        help="Override conv channel list (default [16,32,64])")
    parser.add_argument("--hflip", choices=["true", "false"], default=None,
                        help="Toggle random horizontal flip (default config.ganfp.cnn.hflip)")
    parser.add_argument("--weight_decay", type=float, default=None,
                        help="Override config.ganfp.cnn.weight_decay")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default="cpu",
                        help="torch device (cpu for local; cuda on the server)")
    main(parser.parse_args())
