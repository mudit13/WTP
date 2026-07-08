#!/usr/bin/env python3
"""
Evaluate multi-class generator attribution and split it into in-set vs out-of-set.

IMPORTANT: the DE-FAKE checkpoint provided on the server (clip_linear.pt) is a BINARY
real/fake head - there is NO pretrained multi-class attribution. Attribution in this
project therefore comes from our fine-tuned head (scripts/finetune_defake_head.py), whose
finetune_per_image.csv this script can score. It also works for any future attribution CSV
that has a predicted-generator column.

The script reads a predictions CSV with full_path + a predicted-generator column. True
generator labels come from that CSV if present (true_generator), otherwise by merging with
the master CSV on full_path. Generators are the human names from config (SD1.5,
FLUX.1-schnell, ...).

IN-SET vs OUT-OF-SET is decided by what the head ACTUALLY trained on, not a static config
list: it uses the per-image `in_set` column (written by finetune) if present, else the trained
`classes` list (finetune_metrics.json / --classes_json / --trained_classes), and only falls
back to config.in_set_generators with a warning. This prevents reporting held-out samples of a
TRAINED generator as if they were an out-of-set generalization result.

Outputs attribution metrics (top-1, macro-F1, balanced accuracy), confusion matrices
(PNG + CSV), and a normalized per-image export for the out-of-set analysis.

Usage:
  $WTP_PY_DEFAKE scripts/eval_defake_attribution.py --config configs/config.yaml \
      --predictions results/finetune/finetune_per_image.csv \
      --master /pitsec_sose26_topic8/dataset/master_metadata.csv \
      --out_dir results/attr_eval/ --pred_col pred_generator
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import io_utils, metrics, schema  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _resolve_truth(pred, master_csv, logger):
    """Return a df carrying at least full_path + true_generator (+ any prediction columns)."""
    if "true_generator" in pred.columns:
        return pred.copy()
    if not master_csv:
        raise SystemExit("Predictions lack true_generator and no --master given.")
    meta = pd.read_csv(master_csv)[[schema.PATH, schema.GENERATOR]]
    # Drop any pre-existing generator column so the merge does not create generator_x/_y
    # (which would make the rename a no-op and crash downstream on missing true_generator).
    df = pred.drop(columns=[schema.GENERATOR], errors="ignore").merge(
        meta, on=schema.PATH, how="inner")
    df = df.rename(columns={schema.GENERATOR: "true_generator"})
    logger.info("Merged %d rows with master truth", len(df))
    return df


def _load_trained_classes(args, logger):
    """The trained class list decides what is in-set. Priority: --trained_classes (explicit)
    > --classes_json > a finetune_metrics.json sitting next to the predictions CSV. Returns a
    set of class names, or None if nothing is available (caller then falls back to config)."""
    if args.trained_classes:
        return set(args.trained_classes)
    candidates = [args.classes_json,
                  os.path.join(os.path.dirname(os.path.abspath(args.predictions)),
                               "finetune_metrics.json")]
    for path in candidates:
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            cls = data.get("classes")
            if cls:
                logger.info("Trained classes from %s: %s", path, cls)
                return set(cls)
    return None


def main(args):
    logger = io_utils.setup_logging("eval_defake_attribution")
    io_utils.ensure_dir(args.out_dir)
    config = io_utils.load_config(args.config)
    in_set_cfg = set(config["attribution"]["in_set_generators"])
    real_gens = set(config["attribution"].get("real_generators", []))
    trained = _load_trained_classes(args, logger)

    pred = pd.read_csv(args.predictions)
    if args.pred_col not in pred.columns:
        raise SystemExit("Predictions missing '%s'. Columns: %s"
                         % (args.pred_col, list(pred.columns)))
    df = _resolve_truth(pred, args.master, logger)
    # Attribution is over fakes (exclude real-source rows).
    df = df[~df["true_generator"].isin(real_gens)].copy()
    logger.info("Attribution rows (fake only): %d", len(df))

    # Decide in-set membership from GROUND TRUTH about what was trained, not a static config
    # list. Priority: per-image `in_set` column (written by finetune) > trained-class list >
    # config in_set_generators (with a loud warning, since that can misclassify trained gens).
    if "in_set" in df.columns:
        is_in = df["in_set"].astype(bool)
        logger.info("in/out-of-set from per-image 'in_set' column")
    elif trained is not None:
        is_in = df["true_generator"].isin(trained - real_gens)
        logger.info("in/out-of-set from trained-class list")
    else:
        logger.warning("No trained-class info (no in_set column / --classes_json / "
                       "finetune_metrics.json); falling back to CONFIG in_set_generators - this "
                       "may misclassify trained generators as out-of-set.")
        is_in = df["true_generator"].isin(in_set_cfg)
    df["in_set"] = is_in.values

    def evaluate(subset, tag):
        if subset.empty:
            logger.warning("No rows for %s", tag)
            return None
        y_true = subset["true_generator"].astype(str).values
        y_pred = subset[args.pred_col].astype(str).values
        labels = sorted(set(list(y_true) + list(y_pred)))
        res = metrics.attribution_metrics(y_true, y_pred, labels)
        metrics.save_confusion_matrix(
            np.array(res["confusion_matrix"]), res["labels"],
            png_path=os.path.join(args.out_dir, "cm_%s.png" % tag),
            csv_path=os.path.join(args.out_dir, "cm_%s.csv" % tag),
            title="Attribution (%s)" % tag, normalize=True)
        logger.info("%s: top1=%.3f macroF1=%.3f balAcc=%.3f",
                    tag, res["top1_accuracy"], res["macro_f1"], res["balanced_accuracy"])
        return res

    results = {
        "all_fakes": evaluate(df, "all_fakes"),
        "in_set": evaluate(df[df["in_set"]], "in_set"),
        "out_of_set": evaluate(df[~df["in_set"]], "out_of_set"),
    }
    with open(os.path.join(args.out_dir, "attribution_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    export = df[[schema.PATH, "true_generator", args.pred_col, "in_set"]].copy()
    export = export.rename(columns={args.pred_col: "pred_generator"})
    if "confidence" in df.columns:
        export["confidence"] = df["confidence"].values
    export.to_csv(os.path.join(args.out_dir, "attribution_per_image.csv"), index=False)
    logger.info("Wrote attribution_metrics.json + attribution_per_image.csv to %s",
                args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate attribution (in/out-of-set).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--predictions", required=True,
                        help="CSV with full_path + predicted-generator col (e.g. finetune_per_image.csv)")
    parser.add_argument("--master", default=None,
                        help="master_metadata.csv (needed if predictions lack true_generator)")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--pred_col", default="pred_generator")
    parser.add_argument("--classes_json", default=None,
                        help="finetune_metrics.json with the trained 'classes' list (defines "
                             "in-set). Auto-detected next to --predictions if omitted.")
    parser.add_argument("--trained_classes", nargs="*", default=None,
                        help="Explicit trained class names (overrides --classes_json).")
    main(parser.parse_args())
