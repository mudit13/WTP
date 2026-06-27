#!/usr/bin/env python3
"""
Build the ground-truth index (master_metadata.csv) for all datasets.

Single source of truth, config-driven superset of the team's original
build_master_index.py + update_master_index_dffd.py. Emits the EXACT same schema those
scripts use, so run_defake_batch.py / merge_predictions.py keep working unchanged:

    filename, full_path, label, generator, category, source_dataset, width, height

Runs INSIDE the container (paths in config are container paths and must exist). Skips
datasets whose directory is absent (e.g. openforensics/sd15_img2img before they are added).
Optionally reconciles against an existing predictions CSV to locate row discrepancies.

Usage:
  $WTP_PY_DEFAKE scripts/build_master_index.py --config configs/config.yaml \
      --out /pitsec_sose26_topic8/dataset/master_metadata.csv
  # reconcile against merged predictions:
  $WTP_PY_DEFAKE scripts/build_master_index.py --config configs/config.yaml \
      --out /pitsec_sose26_topic8/dataset/master_metadata.csv \
      --reconcile /pitsec_sose26_topic8/dataset/defake_predictions_all.csv
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import io_utils, schema  # noqa: E402

import pandas as pd  # noqa: E402


def _iter_files(dataset, seed, logger):
    """List image files in dataset['dir']; if sample_size is set, take a SEEDED RANDOM
    subset (reproducible, per-dataset) rather than the first-N sorted (which is biased)."""
    base = dataset["dir"]
    exts = [e.lower() for e in dataset.get("ext", [".png"])]
    files = sorted(f for f in os.listdir(base)
                   if os.path.splitext(f)[1].lower() in exts
                   and os.path.isfile(os.path.join(base, f)))
    total = len(files)
    sample = dataset.get("sample_size")
    if sample and total > int(sample):
        rng = random.Random("%s:%s" % (seed, dataset["name"]))
        files = sorted(rng.sample(files, int(sample)))
        logger.info("%-18s sampled %d/%d (seeded random, seed=%s)",
                    dataset["name"], len(files), total, seed)
    else:
        logger.info("%-18s %5d images from %s", dataset["name"], len(files), base)
    return [os.path.join(base, f) for f in files]


def build_rows(config, seed, logger):
    """Return (rows, skipped_missing, empty). skipped_missing = configured datasets whose
    dir is absent; empty = present dirs that yielded 0 matching files."""
    from PIL import Image
    rows, skipped, empty = [], [], []
    for dataset in config["datasets"]:
        base = dataset["dir"]
        if not os.path.isdir(base):
            logger.warning("[skip] %s: dir not found %s", dataset["name"], base)
            skipped.append(dataset["name"])
            continue
        paths = _iter_files(dataset, seed, logger)
        if not paths:
            empty.append(dataset["name"])
        for path in paths:
            width = height = -1
            try:
                with Image.open(path) as img:
                    width, height = img.size
            except Exception:  # noqa: BLE001
                logger.warning("Unreadable image (size -1): %s", path)
            rows.append({
                schema.FILENAME: os.path.basename(path),
                schema.PATH: path,
                schema.LABEL: dataset["label"],
                schema.GENERATOR: dataset["generator"],
                schema.CATEGORY: dataset.get("category", "unknown"),
                schema.DATASET: dataset["name"],
                schema.WIDTH: width,
                schema.HEIGHT: height,
            })
    return rows, skipped, empty


def reconcile(meta_df, predictions_csv, logger):
    if not os.path.exists(predictions_csv):
        logger.warning("Reconcile file not found: %s", predictions_csv)
        return
    pred = pd.read_csv(predictions_csv)
    if schema.PATH not in pred.columns:
        logger.warning("Predictions CSV has no %s column; cannot reconcile.", schema.PATH)
        return
    meta_paths = set(meta_df[schema.PATH])
    pred_paths = list(pred[schema.PATH])
    pred_set = set(pred_paths)

    logger.info("Reconcile: metadata=%d, predictions=%d (duplicate pred rows=%d)",
                len(meta_paths), len(pred_paths), len(pred_paths) - len(pred_set))
    extra = pred_set - meta_paths
    missing = meta_paths - pred_set
    logger.info("Reconcile: %d predicted paths absent from metadata", len(extra))
    for p in sorted(extra):
        logger.info("  EXTRA (likely a discrepancy row): %s", p)
    logger.info("Reconcile: %d metadata paths never predicted", len(missing))
    for p in sorted(missing):
        logger.info("  MISSING: %s", p)


def main(args):
    logger = io_utils.setup_logging("build_master_index")
    config = io_utils.load_config(args.config)
    seed = int(config.get("seed", 42))

    rows, skipped, empty = build_rows(config, seed, logger)
    df = pd.DataFrame(rows, columns=schema.MASTER_COLUMNS)
    before = len(df)
    df = df.drop_duplicates(subset=[schema.PATH]).reset_index(drop=True)
    if before != len(df):
        logger.info("Dropped %d duplicate %s rows", before - len(df), schema.PATH)

    io_utils.ensure_dir(os.path.dirname(os.path.abspath(args.out)))
    df.to_csv(args.out, index=False)
    logger.info("Wrote %s with %d rows", args.out, len(df))
    logger.info("Label counts:\n%s", df[schema.LABEL].value_counts().to_string())
    logger.info("Generator counts:\n%s", df[schema.GENERATOR].value_counts().to_string())

    # --- partial-dataset guard (make silent gaps loud) -----------------------
    n_real = int((df[schema.LABEL] == schema.REAL).sum())
    n_fake = int((df[schema.LABEL] == schema.FAKE).sum())
    if skipped:
        logger.warning("[summary] %d dataset dir(s) MISSING (skipped): %s",
                       len(skipped), skipped)
    if empty:
        logger.warning("[summary] %d dataset(s) present but yielded 0 files: %s",
                       len(empty), empty)
    if n_real == 0 or n_fake == 0:
        logger.error("[summary] one class is EMPTY (real=%d, fake=%d) - the index is "
                     "UNUSABLE for detection. Fix dataset paths before running experiments.",
                     n_real, n_fake)
    logger.info("[summary] total=%d real=%d fake=%d | datasets: present=%d skipped=%d empty=%d",
                len(df), n_real, n_fake,
                len(config["datasets"]) - len(skipped), len(skipped), len(empty))

    if args.reconcile:
        reconcile(df, args.reconcile, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build master_metadata.csv (real schema).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reconcile", default=None)
    main(parser.parse_args())
