#!/usr/bin/env python3
"""
Out-of-set generalization analysis: how does the attribution model behave on generators it
was not trained on?

Consumes one or more per-image prediction CSVs and, for each, splits rows into in-set vs
out-of-set by the true generator. It then reports:
  - confidence distribution (in-set vs out-of-set)
  - predictive entropy (if an entropy column is present)
  - false-known rate at several confidence thresholds (confident-but-unseen = the failure)
and renders overlaid confidence histograms.

NOTE: there is no pretrained DE-FAKE attribution (the provided head is binary), so the
inputs here come from our fine-tuned head (finetune_per_image.csv, which now contains BOTH the
in-set test split and the force-scored unseen generators) and/or the attribution evaluator
(attribution_per_image.csv). Each per-image CSV needs columns: true_generator, pred_generator,
confidence[, entropy]. In/out-of-set uses the `in_set` column when present (ground truth from
the producing script), else the config out_of_set list.

Usage:
  $WTP_PY_DEFAKE scripts/out_of_set_analysis.py --config configs/config.yaml \
      --out_dir results/oos/ \
      --inputs finetuned=results/finetune_scaled/finetune_per_image.csv \
               attr_eval=results/attr_eval/attribution_per_image.csv
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import io_utils, metrics  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _parse_inputs(pairs):
    parsed = {}
    for item in pairs:
        if "=" not in item:
            raise SystemExit("Each --inputs entry must be name=path, got: %s" % item)
        name, path = item.split("=", 1)
        parsed[name] = path
    return parsed


def _analyze_one(name, df, out_set, thresholds, logger):
    df = df.copy()
    # Prefer the ground-truth `in_set` flag written by the producing script (finetune/eval);
    # only fall back to the config out-of-set list when that column is absent.
    if "in_set" in df.columns:
        df["is_out"] = ~df["in_set"].astype(bool)
    else:
        df["is_out"] = df["true_generator"].astype(str).isin(out_set)
    out_df = df[df["is_out"]]
    in_df = df[~df["is_out"]]

    def stats(sub):
        if sub.empty:
            return {"n": 0}
        s = {"n": int(len(sub)), "mean_confidence": float(sub["confidence"].mean())}
        if "entropy" in sub.columns:
            s["mean_entropy"] = float(sub["entropy"].mean())
        return s

    result = {"in_set": stats(in_df), "out_of_set": stats(out_df)}
    if not out_df.empty:
        result["false_known_rate"] = {
            "@%.2f" % t: metrics.false_known_rate(out_df["confidence"].values, t)
            for t in thresholds
        }
        # On out-of-set rows the true class is absent, so accuracy is meaningless; the
        # informative signal is how confidently the model assigns a (wrong) known label.
    logger.info("%s: in-set n=%d meanConf=%.3f | out-of-set n=%d meanConf=%.3f",
                name, result["in_set"].get("n", 0),
                result["in_set"].get("mean_confidence", float("nan")),
                result["out_of_set"].get("n", 0),
                result["out_of_set"].get("mean_confidence", float("nan")))
    return result, in_df, out_df


def _plot_confidence(name, in_df, out_df, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, 1, 21)
    if not in_df.empty:
        ax.hist(in_df["confidence"], bins=bins, alpha=0.6, label="in-set", density=True)
    if not out_df.empty:
        ax.hist(out_df["confidence"], bins=bins, alpha=0.6, label="out-of-set", density=True)
    ax.set_xlabel("max class confidence")
    ax.set_ylabel("density")
    ax.set_title("Confidence: in-set vs out-of-set (%s)" % name)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(args):
    logger = io_utils.setup_logging("out_of_set_analysis")
    io_utils.ensure_dir(args.out_dir)
    config = io_utils.load_config(args.config)
    out_set = set(config["attribution"]["out_of_set_generators"])
    thresholds = [0.5, 0.7, 0.9]

    inputs = _parse_inputs(args.inputs)
    summary = {}
    for name, path in inputs.items():
        if not os.path.exists(path):
            logger.warning("Input %s not found: %s", name, path)
            continue
        df = pd.read_csv(path)
        if "confidence" not in df.columns:
            logger.warning("%s has no confidence column; skipping", name)
            continue
        if "in_set" not in df.columns and "true_generator" not in df.columns:
            logger.warning("%s has neither 'in_set' nor 'true_generator'; cannot split "
                           "in/out-of-set, skipping", name)
            continue
        result, in_df, out_df = _analyze_one(name, df, out_set, thresholds, logger)
        summary[name] = result
        _plot_confidence(name, in_df, out_df,
                         os.path.join(args.out_dir, "confidence_%s.png" % name))

    with open(os.path.join(args.out_dir, "out_of_set_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Wrote out_of_set_summary.json + confidence plots to %s", args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Out-of-set generalization analysis.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="One or more name=per_image_csv entries")
    main(parser.parse_args())
