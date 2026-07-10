#!/usr/bin/env python3
"""
Small CNN channel-width sweep for the GAN-fp attribution head (scripts/lib/ganfp_net.py).

Trains the end-to-end GANFpClassifier over a few conv-channel configs on the SAME seeded
stratified split, evaluates each on the held-out validation split, and reports val accuracy
per config (plus the trained-head test attribution for the BEST config). Picks the best by
val top-1 accuracy. The point is to choose the channel width empirically rather than hard-code
it; the default config is [32,64,128] and this sweep probes around it.

VRAM safety: trains on --device (default cpu; pass cuda). If a forward/backward step raises a
CUDA out-of-memory, the batch size is halved (32 -> 16 -> 8) and the failing config is retried;
if it still OOMs at batch 8 the config is reported as OOM and skipped. This keeps the sweep
safe on the ~6GB RTX 4050 while still using batch 32 where it fits.

Data modes (identical to train_ganfp_cnn.py):
  --sample_dir <dir>   LOCAL PROTOTYPE. <dir>/<generator>/*.{png,jpg,jpeg}.
  --index <csv>        SERVER / FULL RUN. Reads master_metadata.csv (schema columns).

torch is imported lazily inside lib.ganfp_net; this entry point never imports torch at module
top so `python -m compileall -q scripts` is torch-free.

Local CPU prototype:
  python scripts/ganfp_sweep.py --config configs/config.yaml \
      --sample_dir ganfp_sample --out_dir results/ganfp_sweep_local --device cpu

CUDA sweep (the venv with CUDA torch):
  python scripts/ganfp_sweep.py --config configs/config.yaml \
      --sample_dir ganfp_sample_20260627_215414 --out_dir results/ganfp_sweep \
      --device cuda --jpeg_aug on
"""
import argparse
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import io_utils, metrics, defake_head, ganfp, ganfp_net, schema  # noqa: E402

import numpy as np  # noqa: E402

# Default channel configs to probe (kept compact so the sweep is cheap). The middle entry
# matches config.ganfp.cnn.channels ([32,64,128]); the others bracket it in width.
DEFAULT_CHANNEL_CONFIGS = [[16, 32, 64], [32, 64, 128], [48, 96, 192]]


def _build_split(args, config, seed, common_size, augment, hflip, real_set):
    """Gather (paths, labels_int, classes, y, tr, va, te) the SAME way train_ganfp_cnn does."""
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
    if not all_paths:
        raise SystemExit("No images found; check --sample_dir/--index paths.")

    classes = list(args.classes) if args.classes else sorted(set(generators))
    keep = [g in set(classes) for g in generators]
    paths = [p for p, k in zip(all_paths, keep) if k]
    generators = [g for g, k in zip(generators, keep) if k]
    y = defake_head.encode_labels(np.array(generators, dtype=object), classes)

    # Group-aware split (same-source-photo coupling fix, e.g. OpenForensics real+fake crop
    # pairs); see finetune_defake_head.py for details. No-op when no sidecar is found.
    paths_arr = np.asarray(paths)
    group_map_paths = args.group_map if args.group_map else io_utils.default_group_map_paths(config)
    group_map = io_utils.load_group_map(group_map_paths)
    groups = io_utils.apply_group_map(paths_arr, group_map) if group_map else None

    tr, va, te = defake_head.stratified_split(
        y, test_size=config.get("test_size", 0.2),
        val_size=config.get("val_size", 0.1), seed=seed, keys=paths_arr, groups=groups)
    labels_int = y.tolist()
    return paths, labels_int, classes, y, tr, va, te


def _train_one_config(paths, labels_int, classes, y, tr, va, te,
                      channels, common_size, epochs, lr, weight_decay, augment, hflip,
                      device, seed, logger, max_batch):
    """Train ONE channel config, with OOM-driven batch halving from max_batch down to 8.

    Returns dict {channels, params, batch_size, val_top1, ...} or {channels, oom: True} if the
    config cannot fit even at batch 8. Val top-1 drives the sweep selection.
    """
    import torch  # noqa: E402  (gated: the sweep only runs where torch is importable)

    cw = defake_head.compute_class_weights(y[tr], len(classes))
    train_paths, train_labels = ganfp_net.slice_paths_labels(paths, labels_int, tr)
    val_paths, val_labels = ganfp_net.slice_paths_labels(paths, labels_int, va)

    batch = int(max_batch)
    while batch >= 8:
        train_loader = ganfp_net.build_dataloaders(
            train_paths, train_labels, common_size=common_size, augment=augment,
            hflip=hflip, seed=seed, batch_size=batch, num_workers=0, shuffle=True)
        val_loader = None
        if val_paths:
            val_loader = ganfp_net.build_dataloaders(
                val_paths, val_labels, common_size=common_size, augment=None, hflip=False,
                seed=seed, batch_size=batch, num_workers=0, shuffle=False)
        try:
            clf = ganfp_net.GANFpClassifier(
                num_classes=len(classes), input_size=common_size, channels=channels,
                device=device, lr=lr, weight_decay=weight_decay, seed=seed)
            logger.info("[sweep] channels=%s batch=%d params=%d -> training %d epochs",
                        channels, batch, clf.param_count, epochs)
            clf.fit(train_loader, val_loader=val_loader, epochs=epochs,
                    class_weights=cw, logger=None)
            val_top1 = (clf._eval_accuracy(val_loader) if val_loader is not None else None)
            return {"channels": list(channels), "params": int(clf.param_count),
                    "batch_size": batch, "val_top1": val_top1, "clf": clf}
        except RuntimeError as exc:
            # CUDA OOM: free cache, halve batch, retry. Only treat genuine OOM as retryable;
            # anything else re-raises so real bugs surface.
            if device != "cuda" or "out of memory" not in str(exc).lower():
                raise
            logger.info("[sweep] OOM at channels=%s batch=%d (%s); halving batch",
                        channels, batch, str(exc).strip().splitlines()[0])
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001  (cache clear is best-effort)
                pass
            batch //= 2
    logger.info("[sweep] channels=%s OOM even at batch 8; skipping", channels)
    return {"channels": list(channels), "oom": True}


def main(args):
    logger = io_utils.setup_logging("ganfp_sweep")
    io_utils.ensure_dir(args.out_dir)
    config = io_utils.load_config(args.config)
    seed = int(config.get("seed", 42))

    gcfg = config.get("ganfp", {}) or {}
    ccfg = gcfg.get("cnn", {}) or {}
    common_size = int(args.common_size if args.common_size is not None
                      else ccfg.get("input_size", gcfg.get("common_size", 256)))
    epochs = args.epochs if args.epochs is not None else int(ccfg.get("epochs", 60))
    lr = float(args.lr if args.lr is not None else ccfg.get("lr", gcfg.get("lr", 1e-3)))
    weight_decay = float(args.weight_decay if args.weight_decay is not None
                         else ccfg.get("weight_decay", 1e-4))
    hflip = (args.hflip.lower() == "true") if args.hflip is not None \
        else bool(ccfg.get("hflip", True))
    max_batch = int(args.batch_size)

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

    real_set = set((config.get("attribution", {}) or {}).get("real_generators", []))

    paths, labels_int, classes, y, tr, va, te = _build_split(
        args, config, seed, common_size, augment, hflip, real_set)
    logger.info("Sweep over %d classes: %s (train=%d val=%d test=%d)",
                len(classes), classes, len(tr), len(va), len(te))

    configs = [list(c) for c in (args.channel_configs or DEFAULT_CHANNEL_CONFIGS)]
    logger.info("Channel configs to sweep: %s (epochs=%d device=%s max_batch=%d)",
                configs, epochs, args.device, max_batch)

    results = []
    best = None
    for ch in configs:
        try:
            r = _train_one_config(
                paths, labels_int, classes, y, tr, va, te,
                ch, common_size, epochs, lr, weight_decay, augment, hflip,
                args.device, seed, logger, max_batch)
        except Exception as exc:  # noqa: BLE001  (one config failing must not abort the sweep)
            logger.info("[sweep] channels=%s FAILED: %s", ch, exc)
            traceback.print_exc()
            results.append({"channels": list(ch), "error": str(exc)})
            continue
        if r.get("oom"):
            results.append({"channels": r["channels"], "oom": True})
            continue
        v = r["val_top1"]
        logger.info("[sweep] channels=%s -> val_top1=%s (batch=%d params=%d)",
                    r["channels"], v, r["batch_size"], r["params"])
        results.append({"channels": r["channels"], "params": r["params"],
                        "batch_size": r["batch_size"], "val_top1": v})
        if v is not None and (best is None or v > best["val_top1"]):
            best = {"val_top1": v, "channels": r["channels"], "params": r["params"],
                    "batch_size": r["batch_size"], "clf": r["clf"]}

    # Evaluate the BEST config on the held-out TEST split + record a confusion matrix.
    best_report = None
    if best is not None:
        clf = best["clf"]
        test_paths, test_labels = ganfp_net.slice_paths_labels(paths, labels_int, te)
        proba = clf.predict_proba(test_paths, test_labels, common_size=common_size,
                                  batch_size=best["batch_size"], num_workers=0)
        pred = proba.argmax(axis=1)
        y_true_names = [classes[i] for i in y[te]]
        y_pred_names = [classes[i] for i in pred]
        res = metrics.attribution_metrics(y_true_names, y_pred_names, classes)
        metrics.save_confusion_matrix(
            np.array(res["confusion_matrix"]), res["labels"],
            png_path=os.path.join(args.out_dir, "cm_sweep_best.png"),
            csv_path=os.path.join(args.out_dir, "cm_sweep_best.csv"),
            title="GAN-fp CNN sweep BEST (test)", normalize=True)
        clf.save(os.path.join(args.out_dir, "ganfp_cnn_sweep_best.pt"), classes)
        best_report = {
            "channels": best["channels"], "params": best["params"],
            "batch_size": best["batch_size"], "val_top1": best["val_top1"],
            "test_attribution": res,
        }
        logger.info("[sweep] BEST channels=%s val_top1=%.3f -> test top1=%.3f macroF1=%.3f",
                    best["channels"], best["val_top1"],
                    res["top1_accuracy"], res["macro_f1"])
    else:
        logger.info("[sweep] no config produced a val score (all OOM/errored)")

    out = {
        "data_mode": "sample_dir" if args.sample_dir else "index",
        "seed": seed,
        "classes": classes,
        "common_size": common_size,
        "epochs": epochs,
        "device": args.device,
        "lr": lr,
        "weight_decay": weight_decay,
        "hflip": hflip,
        "jpeg_aug": jpeg_aug,
        "split": {"train": len(tr), "val": len(va), "test": len(te)},
        "configs": results,
        "best": best_report,
    }
    with open(os.path.join(args.out_dir, "ganfp_sweep_results.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    logger.info("Wrote ganfp_sweep_results.json (+ best CM + best head) to %s", args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sweep CNN channel widths for the GAN-fp attribution head.")
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
    parser.add_argument("--channel_configs", type=str, default=None,
                        help="Comma-separated channel lists, e.g. '16,32,64|32,64,128|48,96,192'")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override config.ganfp.cnn.epochs (default 60)")
    parser.add_argument("--lr", type=float, default=None, help="Override config.ganfp.cnn.lr")
    parser.add_argument("--common_size", type=int, default=None,
                        help="Override config.ganfp.cnn.input_size")
    parser.add_argument("--hflip", choices=["true", "false"], default=None,
                        help="Toggle random horizontal flip (default config.ganfp.cnn.hflip)")
    parser.add_argument("--weight_decay", type=float, default=None,
                        help="Override config.ganfp.cnn.weight_decay")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Starting batch size; halved on CUDA OOM down to 8 (VRAM-safe)")
    parser.add_argument("--device", default="cpu",
                        help="torch device (cpu for local; cuda on the server)")
    # Parse the --channel_configs shorthand into a list of int lists.
    ns = parser.parse_args()
    if ns.channel_configs:
        ns.channel_configs = [
            [int(x) for x in part.split(",")] for part in ns.channel_configs.split("|")]
    main(ns)
