#!/usr/bin/env python3
"""
Score DE-FAKE binary detection (real vs fake) from a predictions CSV.

The actual inference is produced by the team's run_defake_batch.py / run_defake_dffd.py,
which write the schema:
    filename, full_path, label, generator, category, source_dataset, width, height,
    defake_predict (0=real,1=fake,-1=error), prob_real, prob_fake, blip_caption

This script consumes that CSV and reports detection metrics overall, per generator, and per
category (real / near_in_set / out_of_set). Run it on each preprocessing variant's
predictions to produce the scaling-vs-cropping comparison. Error rows (defake_predict==-1)
are excluded and counted.

Usage:
  $WTP_PY_DEFAKE scripts/score_defake_detection.py \
      --predictions /pitsec_sose26_topic8/dataset/defake_predictions_all.csv \
      --out_dir results/defake_detection/
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import io_utils, metrics, schema  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _detection(df):
    y_true = schema.is_fake_label(df[schema.LABEL]).astype(int).values
    y_pred = schema.is_fake_predict(df[schema.DEFAKE_PREDICT]).astype(int).values
    y_score = None
    if schema.PROB_FAKE in df.columns and df[schema.PROB_FAKE].notna().any():
        y_score = pd.to_numeric(df[schema.PROB_FAKE], errors="coerce").values
    # Threshold-free metrics need both classes present.
    use_score = y_score if (y_score is not None and len(np.unique(y_true)) == 2) else None
    return metrics.detection_metrics(y_true, y_pred, use_score)


def _group_summary(grp):
    """Per-group summary. Most groups (a single generator, or one category) contain ONLY one
    true class, where precision/recall/macro-F1 are undefined and misleading; for those we
    report a clean detection rate instead. Mixed groups fall back to full detection metrics."""
    y_true = schema.is_fake_label(grp[schema.LABEL]).astype(int).values
    y_pred = schema.is_fake_predict(grp[schema.DEFAKE_PREDICT]).astype(int).values
    classes = sorted(set(y_true.tolist()))
    if len(classes) == 2:
        return _detection(grp)
    n = int(len(grp))
    correct = int((y_true == y_pred).sum())
    rate = float(correct / n) if n else 0.0
    cls = "fake" if classes and classes[0] == 1 else "real"
    out = {"class": cls, "n": n, "correct": correct, "detection_rate": rate}
    # For reals, the error is a false positive (called fake); for fakes, it's a miss (called real).
    out["false_positive_rate" if cls == "real" else "miss_rate"] = float(1.0 - rate)
    return out


def main(args):
    logger = io_utils.setup_logging("score_defake_detection")
    io_utils.ensure_dir(args.out_dir)
    df = pd.read_csv(args.predictions)

    total = len(df)
    errors = int((pd.to_numeric(df[schema.DEFAKE_PREDICT], errors="coerce") == -1).sum())
    df = df[pd.to_numeric(df[schema.DEFAKE_PREDICT], errors="coerce") != -1].copy()
    logger.info("Rows: %d (excluded %d error rows)", total, errors)

    overall = _detection(df)
    logger.info("Overall detection: %s", json.dumps(overall))

    per_generator = {g: _group_summary(grp) for g, grp in df.groupby(schema.GENERATOR)}
    per_category = {}
    if schema.CATEGORY in df.columns:
        per_category = {c: _group_summary(grp) for c, grp in df.groupby(schema.CATEGORY)}

    out = {"n_total": total, "n_errors": errors, "overall": overall,
           "per_generator": per_generator, "per_category": per_category}
    with open(os.path.join(args.out_dir, "detection_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    logger.info("Wrote detection_metrics.json to %s", args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score DE-FAKE binary detection.")
    parser.add_argument("--predictions", required=True,
                        help="Predictions CSV from run_defake_batch.py (real schema)")
    parser.add_argument("--out_dir", required=True)
    main(parser.parse_args())
