#!/usr/bin/env python3
"""
Head-to-head GAN-fp attribution benchmark: Path A (features+MLP) vs Path B (CNN), on ONE
seeded stratified split (identical tr/va/te index arrays passed to both paths).

Both paths share:
  - the SAME split (defake_head.stratified_split over generator labels, config.seed);
  - the SAME per-image JPEG augmentation distribution (image_ops.make_jpeg_augmenter, same
    seed/qrange) so the comparison is apples-to-apples. KNOWN ASYMMETRY: Path A applies the
    augment once at feature-extraction time (frozen across epochs); Path B applies it per
    epoch at DataLoader time. Distributions match; within-method augmentation only.

Emits (under --out_dir):
  benchmark_metrics.json  ONE json: {split, classes, path_a:{attribution,detection,slices,
                          in_dim,pca_components,dct_fuse}, path_b:{attribution,detection,
                          slices,params,epochs}, extras:{defake?,dct?}, comparison:[...]};
    where each path's "slices" = {all, gan_only, diffusion_mismatch}: gan_only is the HEADLINE
    (GAN classes + reals only, diffusion excluded), diffusion_mismatch is the separate slice
    over [SD1.5,FLUX] (expected poor), all is every class. The comparison table carries the
    gan_only headline per method as <method>__gan_only rows.
  cm_path_a.{png,csv}, cm_path_b.{png,csv};
  per_image_path_a.csv, per_image_path_b.csv
      (full_path,true_generator,pred_generator,confidence,entropy [+ extra pred cols]);
  ganfp_pca_head.pt ({state_dict,classes}), ganfp_cnn.pt ({state_dict,classes}).

Optional cross-method ingest: --defake_csv / --dct_csv per-image CSVs (matched on
schema.PATH at TEST-row granularity) are appended as extra columns to the per-image CSVs and
as additional rows in the comparison table.

torch is imported lazily inside lib.ganfp_net; this entry point never imports torch at module
top so `python -m compileall -q scripts` is torch-free.

Local smoke run on the 200-sample:
  python scripts/benchmark_attribution.py --config configs/config.yaml \
      --sample_dir ganfp_sample_20260627_204748 --out_dir results/bench_local --device cpu

Server full run (venv_sd15 / $WTP_PY_DEFAKE):
  $WTP_PY_DEFAKE scripts/benchmark_attribution.py --config configs/config.yaml \
      --index results/index_scaled.csv --out_dir results/bench_scaled
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import io_utils, metrics, defake_head, ganfp, ganfp_net, schema  # noqa: E402

import numpy as np  # noqa: E402


def _extra_predictions(csv_path, test_paths):
    """Read an external per-image CSV (DE-FAKE / DCT) and return a path->pred_generator map
    restricted to the test rows. Returns (col_name, pred_list_aligned_to_test_paths) or
    (None, None) if no overlap."""
    import pandas as pd  # noqa: E402
    if not csv_path or not os.path.exists(csv_path):
        return None, None
    df = pd.read_csv(csv_path)
    if schema.PATH not in df.columns:
        return None, None
    # Prefer a 'pred_generator' column; fall back to defake_predict (binary).
    col = "pred_generator" if "pred_generator" in df.columns else None
    if col is None and "defake_predict" in df.columns:
        col = "defake_predict"
    if col is None:
        return None, None
    lookup = dict(zip(df[schema.PATH].astype(str), df[col]))
    out = []
    seen = False
    for p in test_paths:
        if p in lookup:
            out.append(lookup[p])
            seen = True
        else:
            out.append(None)
    if not seen:
        return None, None
    return col, out


def _binary_detection(classes, y_te_idx, pred_idx, proba, real_set):
    """Fold multi-class predictions into binary detection (fake=1) and score."""
    y_true_names = [classes[i] for i in y_te_idx]
    y_pred_names = [classes[i] for i in pred_idx]
    fake_idx = [i for i, c in enumerate(classes) if c not in real_set]
    y_true_bin = np.array([1 if t not in real_set else 0 for t in y_true_names])
    y_pred_bin = np.array([1 if p not in real_set else 0 for p in y_pred_names])
    y_score = proba[:, fake_idx].sum(axis=1) if fake_idx else None
    return metrics.detection_metrics(y_true_bin, y_pred_bin, y_score=y_score)


# GAN-only attribution slice. By design the closed-set GAN-fp model is trained to attribute
# GAN families + reals; diffusion generators (SD1.5, FLUX) are out-of-set and expected to
# mismatch. The GAN-only slice (GAN classes + reals, diffusion folded away) is the HEADLINE;
# the diffusion_mismatch slice quantifies how badly the model handles the held-out diffusion
# generators. The all-class number (every class, no folding) is kept alongside for context.
GAN_CLASSES = ["StyleGAN3-FFHQ", "PGGAN-v1", "PGGAN-v2", "StarGAN", "FaceApp"]
DIFFUSION_CLASSES = ["SD1.5", "FLUX", "FLUX.1-schnell", "SD1.5-img2img"]
REAL_CLASSES = ["London-DB", "FFHQ", "CelebA"]


def _ganonly_and_diffusion_slices(classes, y_te_idx, pred_idx, logger):
    """Compute the GAN-only headline slice + the diffusion_mismatch slice + the all-class
    number. Returns a dict keyed by slice name. Per-class reports are included so each class's
    recall is visible in every slice.

    GAN-only keep set = GAN classes present + reals present (diffusion EXCLUDED).
    diffusion_mismatch keep set = diffusion classes present (model expected to do poorly).
    all = every class present (no folding)."""
    y_true_names = [classes[i] for i in y_te_idx]
    y_pred_names = [classes[i] for i in pred_idx]
    present = set(classes)

    gan_keep = sorted((set(GAN_CLASSES) | set(REAL_CLASSES)) & present)
    diff_keep = sorted(set(DIFFUSION_CLASSES) & present)

    out = {"all": metrics.attribution_metrics(y_true_names, y_pred_names, classes)}
    if gan_keep:
        out["gan_only"] = metrics.attribution_slice(
            y_true_names, y_pred_names, classes, gan_keep, other_label="diffusion_mismatch")
        logger.info("GAN-only slice (exclude diffusion): top1=%.3f macroF1=%.3f balAcc=%.3f "
                    "over %d classes", out["gan_only"]["top1_accuracy"],
                    out["gan_only"]["macro_f1"], out["gan_only"]["balanced_accuracy"],
                    len(gan_keep))
    else:
        out["gan_only"] = None
    if diff_keep:
        out["diffusion_mismatch"] = metrics.attribution_metrics(
            [t for t in y_true_names if t in set(diff_keep)],
            [p for t, p in zip(y_true_names, y_pred_names) if t in set(diff_keep)],
            diff_keep)
        logger.info("Diffusion mismatch slice (%s): top1=%.3f macroF1=%.3f (expected poor)",
                    diff_keep, out["diffusion_mismatch"]["top1_accuracy"],
                    out["diffusion_mismatch"]["macro_f1"])
    else:
        out["diffusion_mismatch"] = None
    return out


def _write_per_image(out_dir, fname, test_paths, classes, y_te_idx, pred_idx, proba,
                     extras):
    import pandas as pd  # noqa: E402
    ent = metrics.predictive_entropy(proba)
    data = {
        schema.PATH: test_paths,
        "true_generator": [classes[i] for i in y_te_idx],
        "pred_generator": [classes[i] for i in pred_idx],
        "confidence": proba.max(axis=1),
        "entropy": ent,
    }
    for col, vals in extras.items():
        data[col] = vals
    pd.DataFrame(data).to_csv(os.path.join(out_dir, fname), index=False)


def run_path_a(X, generators, classes, y, tr, va, te, real_set,
               pca_components, dct_fuse, dct_components, dct_X,
               epochs, lr, device, seed, logger, out_dir):
    """Path A: FingerprintStandardizer (fit on TRAIN ONLY) + defake_head._MLPHead."""
    dct_train = dct_X[tr] if (dct_fuse and dct_X is not None) else None
    std, in_dim = ganfp.build_pca_pipeline(
        X[tr], pca_components=pca_components, dct_fuse=dct_fuse,
        dct_components=dct_components, dct_train=dct_train)
    dct_all = dct_X if (dct_fuse and dct_X is not None) else None
    Xtr_s = std.transform(X[tr], dct_train)
    Xva_s = std.transform(X[va], dct_X[va] if dct_all is not None else None)
    Xte_s = std.transform(X[te], dct_X[te] if dct_all is not None else None)
    logger.info("Path A: in_dim=%d pca_components=%d dct_fuse=%s",
                in_dim, pca_components, dct_fuse)

    cw = defake_head.compute_class_weights(y[tr], len(classes))
    head = defake_head._MLPHead(in_dim=in_dim, num_classes=len(classes),
                                device=device, seed=seed)
    head.fit(Xtr_s, y[tr], Xva_s, y[va], epochs=epochs, lr=lr,
             class_weights=cw, logger=logger)
    proba = head.predict_proba(Xte_s)
    pred = proba.argmax(axis=1)
    y_true_names = [classes[i] for i in y[te]]
    y_pred_names = [classes[i] for i in pred]
    res = metrics.attribution_metrics(y_true_names, y_pred_names, classes)
    metrics.save_confusion_matrix(
        np.array(res["confusion_matrix"]), res["labels"],
        png_path=os.path.join(out_dir, "cm_path_a.png"),
        csv_path=os.path.join(out_dir, "cm_path_a.csv"),
        title="Path A (feature+MLP) attribution (test)", normalize=True)
    det = _binary_detection(classes, y[te], pred, proba, real_set)
    slices = _ganonly_and_diffusion_slices(classes, y[te], pred, logger)
    head.save(os.path.join(out_dir, "ganfp_pca_head.pt"), classes)
    logger.info("Path A test: top1=%.3f macroF1=%.3f balAcc=%.3f",
                res["top1_accuracy"], res["macro_f1"], res["balanced_accuracy"])
    return {"proba": proba, "pred": pred, "res": res, "det": det, "slices": slices,
            "in_dim": in_dim, "std": std}


def run_path_b(paths, labels_int, classes, y, tr, va, te, real_set,
               common_size, channels, epochs, lr, weight_decay,
               batch_size, num_workers, augment, hflip, device, seed,
               logger, out_dir):
    """Path B: GANFpClassifier (CNN). Loads images lazily via GANFpDataset on the shared split."""
    train_paths, train_labels = ganfp_net.slice_paths_labels(paths, labels_int, tr)
    val_paths, val_labels = ganfp_net.slice_paths_labels(paths, labels_int, va)
    test_paths, test_labels = ganfp_net.slice_paths_labels(paths, labels_int, te)
    train_loader = ganfp_net.build_dataloaders(
        train_paths, train_labels, common_size=common_size, augment=augment,
        hflip=hflip, seed=seed, batch_size=batch_size, num_workers=num_workers,
        shuffle=True)
    val_loader = None
    if val_paths:
        val_loader = ganfp_net.build_dataloaders(
            val_paths, val_labels, common_size=common_size, augment=None, hflip=False,
            seed=seed, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    cw = defake_head.compute_class_weights(y[tr], len(classes))
    clf = ganfp_net.GANFpClassifier(
        num_classes=len(classes), input_size=common_size, channels=channels,
        device=device, lr=lr, weight_decay=weight_decay, seed=seed)
    logger.info("Path B: CNN params=%d", clf.param_count)
    clf.fit(train_loader, val_loader=val_loader, epochs=epochs,
            class_weights=cw, logger=logger)
    proba = clf.predict_proba(test_paths, test_labels, common_size=common_size,
                              batch_size=batch_size, num_workers=num_workers)
    pred = proba.argmax(axis=1)
    y_true_names = [classes[i] for i in y[te]]
    y_pred_names = [classes[i] for i in pred]
    res = metrics.attribution_metrics(y_true_names, y_pred_names, classes)
    metrics.save_confusion_matrix(
        np.array(res["confusion_matrix"]), res["labels"],
        png_path=os.path.join(out_dir, "cm_path_b.png"),
        csv_path=os.path.join(out_dir, "cm_path_b.csv"),
        title="Path B (CNN) attribution (test)", normalize=True)
    det = _binary_detection(classes, y[te], pred, proba, real_set)
    slices = _ganonly_and_diffusion_slices(classes, y[te], pred, logger)
    clf.save(os.path.join(out_dir, "ganfp_cnn.pt"), classes)
    logger.info("Path B test: top1=%.3f macroF1=%.3f balAcc=%.3f",
                res["top1_accuracy"], res["macro_f1"], res["balanced_accuracy"])
    return {"proba": proba, "pred": pred, "res": res, "det": det, "slices": slices,
            "params": clf.param_count, "test_paths": test_paths}


def main(args):
    logger = io_utils.setup_logging("benchmark_attribution")
    io_utils.ensure_dir(args.out_dir)
    config = io_utils.load_config(args.config)
    seed = int(config.get("seed", 42))

    gcfg = config.get("ganfp", {}) or {}
    common_size = int(gcfg.get("common_size", 256))
    feat_size = int(gcfg.get("feat_size", 32))
    mode = str(gcfg.get("mode", "both"))
    ccfg = gcfg.get("cnn", {}) or {}
    pcfg = gcfg.get("pca", {}) or {}
    pca_components = int(pcfg.get("components", 64))
    dct_fuse = bool(pcfg.get("dct_fuse", False))
    dct_components = int(pcfg.get("dct_components", 32))
    channels = list(ccfg.get("channels", [16, 32, 64]))
    cnn_epochs = args.cnn_epochs if args.cnn_epochs is not None else int(ccfg.get("epochs", 30))
    mlp_epochs = args.mlp_epochs if args.mlp_epochs is not None else int(gcfg.get("epochs", 40))
    lr = float(args.lr) if args.lr is not None else float(gcfg.get("lr", 1e-3))
    weight_decay = float(ccfg.get("weight_decay", 1e-4))
    hflip = bool(ccfg.get("hflip", True))

    aug_cfg = config.get("augmentation", {}) or {}
    qrange = tuple(aug_cfg.get("jpeg_quality_range", [30, 100]))
    from lib import image_ops  # noqa: E402
    augment = image_ops.make_jpeg_augmenter(qrange, seed) if args.jpeg_aug else None
    logger.info("Benchmark: pca=%d dct_fuse=%s cnn_channels=%s jpeg_aug=%s",
                pca_components, dct_fuse, channels, bool(augment))

    real_set = set((config.get("attribution", {}) or {}).get("real_generators", []))

    # --- gather paths + features ONCE (Path A consumes features, Path B consumes paths) -
    if args.sample_dir:
        paths_list, generators_list = ganfp.scan_sample_dir(args.sample_dir)
        labels_bin = [schema.REAL if g in real_set else schema.FAKE for g in generators_list]
        X, generator_arr, label_arr, path_arr = ganfp.features_from_samples(
            paths_list, generators_list, labels_bin, common_size, feat_size, mode, augment)
        paths_all = [str(p) for p in path_arr]
        generators_all = [str(g) for g in generator_arr]
    elif args.index:
        X, generator_arr, label_arr, path_arr = ganfp.build_features(
            args.index, args.features_cache, common_size, feat_size, mode,
            bool(augment), qrange, seed, force=args.recompute_features)
        paths_all = [str(p) for p in path_arr]
        generators_all = [str(g) for g in generator_arr]
    else:
        raise SystemExit("Provide --sample_dir (local prototype) or --index (full run).")
    logger.info("Gathered %d images; feature matrix %s", len(paths_all), X.shape)
    if len(paths_all) == 0:
        raise SystemExit("No features/images extracted; check --sample_dir/--index paths.")

    # Optional DCT feature channel. X is aligned to paths_all (the kept image order from
    # extract_fingerprints/features_from_samples); we MUST realign the DCT rows to the SAME
    # paths_all order by path key, NOT by positional truncation -- a separate extraction pass
    # can drop a different set of unreadable images and silently shift rows otherwise.
    dct_X = None
    if dct_fuse:
        _dct_raw, dct_kept = ganfp.extract_dct_features(paths_all, common_size=common_size,
                                                        augment=augment)
        # The kept-path set must agree with what X was built from. X is row-aligned to
        # paths_all (each row of X corresponds to paths_all[row] by construction in
        # features_from_samples / build_features, which realign to the kept order).
        if set(dct_kept) != set(paths_all):
            raise SystemExit(
                "DCT extraction dropped a different image set than the fingerprint "
                "extraction; cannot safely row-align the DCT channel. "
                f"fingerprint-kept={len(paths_all)} dct-kept={len(dct_kept)} "
                f"diff={sorted(set(paths_all) ^ set(dct_kept))[:5]}")
        # Realign dct rows to the paths_all order via the kept->row lookup.
        _dct_order = {p: i for i, p in enumerate(dct_kept)}
        dct_X = np.stack([_dct_raw[_dct_order[p]] for p in paths_all], axis=0).astype(
            np.float32)
        assert dct_X.shape[0] == X.shape[0], (
            f"DCT realignment size mismatch: dct={dct_X.shape[0]} X={X.shape[0]}")
        logger.info("DCT fusion channel (path-realigned): %s", dct_X.shape)

    classes = list(args.classes) if args.classes else sorted(set(generators_all))
    keep = np.array([g in set(classes) for g in generators_all])
    X = X[keep]
    if dct_X is not None:
        dct_X = dct_X[keep]
    paths_kept = [p for p, k in zip(paths_all, keep) if k]
    generators_kept = [g for g, k in zip(generators_all, keep) if k]
    y = defake_head.encode_labels(np.array(generators_kept, dtype=object), classes)
    labels_int = y.tolist()
    logger.info("Benchmarking over %d classes: %s", len(classes), classes)

    # ONE seeded stratified split, shared verbatim by both paths.
    tr, va, te = defake_head.stratified_split(
        y, test_size=config.get("test_size", 0.2),
        val_size=config.get("val_size", 0.1), seed=seed)
    logger.info("Shared split: train=%d val=%d test=%d", len(tr), len(va), len(te))

    # --- Path A --------------------------------------------------------------------
    a = run_path_a(X, generators_kept, classes, y, tr, va, te, real_set,
                   pca_components, dct_fuse, dct_components, dct_X,
                   mlp_epochs, lr, args.device, seed, logger, args.out_dir)

    # --- Path B --------------------------------------------------------------------
    b = run_path_b(paths_kept, labels_int, classes, y, tr, va, te, real_set,
                   common_size, channels, cnn_epochs, lr, weight_decay,
                   args.batch_size, args.num_workers, augment, hflip, args.device, seed,
                   logger, args.out_dir)

    # --- optional cross-method ingest (DE-FAKE / DCT per-image CSVs) ----------------
    test_paths = b["test_paths"]
    extras = {}
    defake_col, defake_preds = _extra_predictions(args.defake_csv, test_paths)
    if defake_col is not None:
        extras["defake_" + defake_col] = defake_preds
    dct_col, dct_preds = _extra_predictions(args.dct_csv, test_paths)
    if dct_col is not None:
        extras["dct_" + dct_col] = dct_preds

    _write_per_image(args.out_dir, "per_image_path_a.csv", test_paths, classes,
                     y[te], a["pred"], a["proba"], dict(extras))
    _write_per_image(args.out_dir, "per_image_path_b.csv", test_paths, classes,
                     y[te], b["pred"], b["proba"], dict(extras))

    comparison = [
        {"method": "ganfp_feature_mlp", "top1_accuracy": a["res"]["top1_accuracy"],
         "macro_f1": a["res"]["macro_f1"], "balanced_accuracy": a["res"]["balanced_accuracy"],
         "detection_balanced_accuracy": a["det"]["balanced_accuracy"]},
        {"method": "ganfp_cnn", "top1_accuracy": b["res"]["top1_accuracy"],
         "macro_f1": b["res"]["macro_f1"], "balanced_accuracy": b["res"]["balanced_accuracy"],
         "detection_balanced_accuracy": b["det"]["balanced_accuracy"]},
    ]
    # GAN-only headline (exclude diffusion) per method, where the slice was computable.
    for tag, src in (("ganfp_feature_mlp", a), ("ganfp_cnn", b)):
        go = src["slices"].get("gan_only") if src["slices"] else None
        if go is not None:
            comparison.append({
                "method": tag + "__gan_only", "top1_accuracy": go["top1_accuracy"],
                "macro_f1": go["macro_f1"], "balanced_accuracy": go["balanced_accuracy"],
                "detection_balanced_accuracy": None,
                "note": "GAN classes + reals only; diffusion folded to diffusion_mismatch"})
    if defake_col is not None:
        # Binary detection accuracy for the external detector on the test rows.
        import pandas as pd  # noqa: E402
        y_true_bin = np.array([1 if classes[i] not in real_set else 0 for i in y[te]])
        ext_bin = np.array([1 if str(v) not in real_set and v is not None else 0
                            for v in defake_preds])
        det_ext = metrics.detection_metrics(y_true_bin, ext_bin)
        comparison.append({"method": "defake", "top1_accuracy": det_ext["accuracy"],
                           "macro_f1": det_ext["macro_f1"],
                           "balanced_accuracy": det_ext["balanced_accuracy"],
                           "detection_balanced_accuracy": det_ext["balanced_accuracy"]})
    if dct_col is not None:
        y_true_names = [classes[i] for i in y[te]]
        dct_pred_names = [str(v) if v is not None else "" for v in dct_preds]
        res_dct = metrics.attribution_metrics(y_true_names, dct_pred_names, classes)
        comparison.append({"method": "dct", "top1_accuracy": res_dct["top1_accuracy"],
                           "macro_f1": res_dct["macro_f1"],
                           "balanced_accuracy": res_dct["balanced_accuracy"],
                           "detection_balanced_accuracy": None})

    out = {
        "split": {"seed": seed, "test_size": config.get("test_size", 0.2),
                  "val_size": config.get("val_size", 0.1),
                  "train": len(tr), "val": len(va), "test": len(te)},
        "classes": classes,
        "path_a": {"attribution": a["res"], "detection": a["det"],
                   "slices": a["slices"],
                   "in_dim": a["in_dim"], "pca_components": pca_components,
                   "dct_fuse": dct_fuse, "epochs": mlp_epochs},
        "path_b": {"attribution": b["res"], "detection": b["det"],
                   "slices": b["slices"],
                   "params": b["params"], "epochs": cnn_epochs,
                   "channels": channels, "common_size": common_size},
        "extras": {"defake_csv": args.defake_csv, "dct_csv": args.dct_csv},
        "comparison": comparison,
    }
    with open(os.path.join(args.out_dir, "benchmark_metrics.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    logger.info("Wrote benchmark_metrics.json + per-path CMs + per-image CSVs to %s",
                args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Head-to-head GAN-fp attribution benchmark (feature+MLP vs CNN).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--index", default=None,
                        help="Index CSV (full_path,generator,label,...) - full run mode")
    parser.add_argument("--sample_dir", default=None,
                        help="Local prototype: <dir>/<generator>/* image folders")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--features_cache", default=None, help="GAN-fp feature .npz cache path")
    parser.add_argument("--classes", nargs="*", default=None,
                        help="Restrict to these generator classes (default: all present)")
    parser.add_argument("--jpeg_aug", action="store_true",
                        help="Apply JPEG augmentation (use config.augmentation range+seed)")
    parser.add_argument("--recompute_features", action="store_true")
    parser.add_argument("--cnn_epochs", type=int, default=None,
                        help="Override config.ganfp.cnn.epochs for Path B")
    parser.add_argument("--mlp_epochs", type=int, default=None,
                        help="Override config.ganfp.epochs for Path A (MLP head)")
    parser.add_argument("--lr", type=float, default=None, help="Override config.ganfp.lr")
    parser.add_argument("--device", default="cpu",
                        help="torch device (cpu for local; cuda on the server)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--defake_csv", default=None,
                        help="Optional DE-FAKE per-image CSV (full_path,pred_generator|defake_predict)")
    parser.add_argument("--dct_csv", default=None,
                        help="Optional DCT per-image CSV (full_path,pred_generator)")
    main(parser.parse_args())
