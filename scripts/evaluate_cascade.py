#!/usr/bin/env python3
"""
Evaluate the requested two-stage system: DCT-SVM detection -> DE-FAKE attribution.

The attribution CSV must score the SAME fixed test index as dct_per_image.csv. For known fake
generators, a result is end-to-end correct only when DCT predicts fake AND the attribution head
predicts the true generator. OpenForensics-fake is reported separately as an unseen challenge.
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import attribution_taxonomy, io_utils, metrics, schema  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


NOT_DETECTED = "__not_detected__"


def _require_unique(df, path, name):
    if path not in df.columns:
        raise SystemExit("%s is missing %s" % (name, path))
    dup = int(df[path].duplicated().sum())
    if dup:
        raise SystemExit("%s has %d duplicate %s rows" % (name, dup, path))


def evaluate(dct, attr, config):
    _require_unique(dct, schema.PATH, "DCT predictions")
    _require_unique(attr, schema.PATH, "attribution predictions")
    required_dct = {"generator", "y_true", "pred"}
    required_attr = {"pred_generator"}
    if not required_dct.issubset(dct.columns):
        raise SystemExit("DCT predictions missing: %s"
                         % sorted(required_dct - set(dct.columns)))
    if not required_attr.issubset(attr.columns):
        raise SystemExit("Attribution predictions missing: %s"
                         % sorted(required_attr - set(attr.columns)))

    merged = dct.merge(
        attr.drop(columns=["true_generator"], errors="ignore"),
        on=schema.PATH, how="inner", validate="one_to_one")
    if len(merged) != len(dct) or len(merged) != len(attr):
        raise SystemExit(
            "Cascade inputs must cover the same fixed test rows: dct=%d attr=%d matched=%d"
            % (len(dct), len(attr), len(merged)))

    merged["y_true"] = pd.to_numeric(merged["y_true"], errors="raise").astype(int)
    merged["pred"] = pd.to_numeric(merged["pred"], errors="raise").astype(int)
    fake_classes = attribution_taxonomy.fake_generators(config)
    out_set = set(attribution_taxonomy.out_of_set_generators(config))
    known_fake = merged["generator"].isin(fake_classes)
    oos_fake = merged["generator"].isin(out_set)
    real = merged["y_true"] == 0
    passed = merged["pred"] == 1

    dct_score = (pd.to_numeric(merged["score"], errors="coerce").to_numpy()
                 if "score" in merged.columns else None)
    detection = metrics.detection_metrics(
        merged["y_true"].to_numpy(), merged["pred"].to_numpy(), dct_score)

    known = merged[known_fake].copy()
    known_passed = known[known["pred"] == 1]
    pipeline_pred = np.where(
        known["pred"].to_numpy() == 1,
        known["pred_generator"].astype(str).to_numpy(),
        NOT_DETECTED,
    )
    end_to_end = metrics.attribution_metrics(
        known["generator"].astype(str).to_numpy(), pipeline_pred,
        labels=fake_classes + [NOT_DETECTED])
    conditional = (
        metrics.attribution_metrics(
            known_passed["generator"].astype(str).to_numpy(),
            known_passed["pred_generator"].astype(str).to_numpy(),
            labels=fake_classes)
        if len(known_passed) else None
    )

    per_generator = {}
    for generator in fake_classes:
        grp = known[known["generator"] == generator]
        n = len(grp)
        detected = grp["pred"] == 1
        correct = detected & (grp["pred_generator"].astype(str) == generator)
        per_generator[generator] = {
            "n": int(n),
            "detection_recall": float(detected.mean()) if n else None,
            "conditional_attribution_accuracy": (
                float((grp.loc[detected, "pred_generator"].astype(str) == generator).mean())
                if int(detected.sum()) else None),
            "end_to_end_recall": float(correct.mean()) if n else None,
        }

    oos = merged[oos_fake & (merged["y_true"] == 1)]
    oos_passed = oos[oos["pred"] == 1]
    oos_result = {
        "n": int(len(oos)),
        "detection_recall": float((oos["pred"] == 1).mean()) if len(oos) else None,
        "forced_label_distribution_after_detection": dict(
            Counter(oos_passed["pred_generator"].astype(str))),
    }
    if len(oos_passed) and "confidence" in oos_passed.columns:
        oos_result["mean_attribution_confidence_after_detection"] = float(
            oos_passed["confidence"].mean())

    real_passed = merged[real & passed]
    result = {
        "pipeline": "DCT-SVM -> DE-FAKE eight-way attribution",
        "n_merged": int(len(merged)),
        "detection": detection,
        "known_fake": {
            "n": int(len(known)),
            "n_detected": int(len(known_passed)),
            "n_not_detected": int(len(known) - len(known_passed)),
            "conditional_attribution": conditional,
            "end_to_end_attribution": end_to_end,
            "per_generator": per_generator,
        },
        "openforensics_fake_challenge": oos_result,
        "real_false_positives": {
            "n_real": int(real.sum()),
            "n_predicted_fake": int(len(real_passed)),
            "rate": float(len(real_passed) / int(real.sum())) if int(real.sum()) else None,
            "forced_generator_distribution": dict(
                Counter(real_passed["pred_generator"].astype(str))),
        },
    }

    merged["pipeline_pred_generator"] = np.where(
        passed, merged["pred_generator"].astype(str), NOT_DETECTED)
    merged["known_fake"] = known_fake
    merged["out_of_set_fake"] = oos_fake
    merged["end_to_end_correct"] = (
        known_fake & passed
        & (merged["pred_generator"].astype(str) == merged["generator"].astype(str)))
    return result, merged


def main(args):
    logger = io_utils.setup_logging("evaluate_cascade")
    config = io_utils.load_config(args.config)
    dct = pd.read_csv(args.dct_predictions)
    attr = pd.read_csv(args.attribution_predictions)
    result, per_image = evaluate(dct, attr, config)
    io_utils.ensure_dir(args.out_dir)
    with open(os.path.join(args.out_dir, "cascade_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    per_image.to_csv(os.path.join(args.out_dir, "cascade_per_image.csv"), index=False)
    logger.info("Wrote cascade metrics/per-image rows to %s", args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DCT -> DE-FAKE attribution cascade.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dct_predictions", required=True,
                        help="dct_per_image.csv from the fixed shared test split")
    parser.add_argument("--attribution_predictions", required=True,
                        help="Eight-way head predictions over the same test index")
    parser.add_argument("--out_dir", required=True)
    main(parser.parse_args())
