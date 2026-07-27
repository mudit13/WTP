#!/usr/bin/env python3
"""
Compare source-specific attribution heads with and without FFHQ as a real class.

Both fine-tune runs use content-stable splits, so fake-generator test rows are identical.
The comparison is intentionally restricted to true fake rows and highlights StyleGAN3-FFHQ,
answering the professor's request without replacing the primary eight-way experiment.
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import attribution_taxonomy, io_utils, metrics, schema  # noqa: E402

import pandas as pd  # noqa: E402


def _load_fake_test(path, fake_classes):
    df = pd.read_csv(path)
    required = {schema.PATH, "true_generator", "pred_generator", "in_set"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit("%s missing column(s): %s" % (path, sorted(missing)))
    in_set = df["in_set"].astype(str).str.lower().isin({"true", "1"})
    df = df[in_set]
    df = df[df["true_generator"].astype(str).isin(fake_classes)].copy()
    if df[schema.PATH].duplicated().any():
        raise SystemExit("%s contains duplicate fake-test paths" % path)
    return df


def _summary(df, fake_classes, real_classes):
    overall = metrics.attribution_metrics(
        df["true_generator"].astype(str),
        df["pred_generator"].astype(str),
        labels=fake_classes + real_classes,
    )
    style = df[df["true_generator"].astype(str) == "StyleGAN3-FFHQ"]
    return {
        "overall_fake_only": overall,
        "stylegan3": {
            "n": int(len(style)),
            "recall": float(
                (style["pred_generator"].astype(str) == "StyleGAN3-FFHQ").mean())
            if len(style) else None,
            "prediction_distribution": dict(
                Counter(style["pred_generator"].astype(str))),
            "real_prediction_rate": float(
                style["pred_generator"].astype(str).isin(real_classes).mean())
            if len(style) else None,
        },
    }


def compare(with_ffhq, without_ffhq, config):
    fake_classes = attribution_taxonomy.fake_generators(config)
    real_classes = attribution_taxonomy.real_generators(config)
    a = _load_fake_test(with_ffhq, fake_classes)
    b = _load_fake_test(without_ffhq, fake_classes)
    paired = a.merge(
        b[[schema.PATH, "pred_generator", "confidence", "entropy"]],
        on=schema.PATH, how="inner", suffixes=("_with_ffhq", "_without_ffhq"),
        validate="one_to_one",
    )
    if len(paired) != len(a) or len(paired) != len(b):
        raise SystemExit(
            "FFHQ ablation test rows are not matched: with=%d without=%d paired=%d"
            % (len(a), len(b), len(paired)))

    with_summary = _summary(a, fake_classes, real_classes)
    without_reals = [r for r in real_classes if r != "FFHQ"]
    without_summary = _summary(b, fake_classes, without_reals)
    paired["correct_with_ffhq"] = (
        paired["pred_generator_with_ffhq"].astype(str)
        == paired["true_generator"].astype(str))
    paired["correct_without_ffhq"] = (
        paired["pred_generator_without_ffhq"].astype(str)
        == paired["true_generator"].astype(str))
    n_with_only = int(
        (paired["correct_with_ffhq"] & ~paired["correct_without_ffhq"]).sum())
    n_without_only = int(
        (~paired["correct_with_ffhq"] & paired["correct_without_ffhq"]).sum())

    result = {
        "purpose": "Professor-requested FFHQ removal diagnostic",
        "interpretation": (
            "A change after removing FFHQ shows sensitivity to the FFHQ class/training "
            "population; it does not by itself prove that FFHQ preprocessing caused the errors."
        ),
        "n_paired_fake_test": int(len(paired)),
        "with_ffhq": with_summary,
        "without_ffhq": without_summary,
        "delta_without_minus_with": {
            "top1_accuracy": (
                without_summary["overall_fake_only"]["top1_accuracy"]
                - with_summary["overall_fake_only"]["top1_accuracy"]),
            "balanced_accuracy": (
                without_summary["overall_fake_only"]["balanced_accuracy"]
                - with_summary["overall_fake_only"]["balanced_accuracy"]),
            "cohen_kappa": (
                without_summary["overall_fake_only"]["cohen_kappa"]
                - with_summary["overall_fake_only"]["cohen_kappa"]),
            "stylegan3_recall": (
                without_summary["stylegan3"]["recall"]
                - with_summary["stylegan3"]["recall"]),
            "stylegan3_real_prediction_rate": (
                without_summary["stylegan3"]["real_prediction_rate"]
                - with_summary["stylegan3"]["real_prediction_rate"]),
        },
        "paired_correctness": {
            "correct_only_with_ffhq": n_with_only,
            "correct_only_without_ffhq": n_without_only,
        },
    }
    return result, paired


def main(args):
    logger = io_utils.setup_logging("compare_ffhq_ablation")
    config = io_utils.load_config(args.config)
    result, paired = compare(args.with_ffhq, args.without_ffhq, config)
    io_utils.ensure_dir(args.out_dir)
    with open(os.path.join(args.out_dir, "ffhq_ablation.json"), "w",
              encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    paired.to_csv(os.path.join(args.out_dir, "ffhq_ablation_per_image.csv"), index=False)
    logger.info("Wrote paired FFHQ ablation over %d fake test rows to %s",
                len(paired), args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare attribution with/without FFHQ.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--with_ffhq", required=True)
    parser.add_argument("--without_ffhq", required=True)
    parser.add_argument("--out_dir", required=True)
    main(parser.parse_args())
