#!/usr/bin/env python3
"""
Robustness experiments (Phase F): apply controlled perturbations to HELD-OUT TEST images
only, then measure how detection/attribution degrades.

Perturbations (from configs/config.yaml robustness block):
  JPEG q30/50/70, Gaussian blur sigma=1/2, resize round-trip 0.5/0.75, sharpening.

Two modes:
  generate : write perturbed PNG copies of a test index and an index CSV per perturbation,
             ready to be fed to run_defake_detection.py / dct_extract_features.py.
  score    : given a clean predictions CSV and a perturbed predictions CSV (each with a
             prediction column and optional confidence), compute performance drop, mean
             confidence drop, and label-flip rate.

Usage:
  $WTP_PY_DEFAKE scripts/robustness_perturb.py --mode generate --config configs/config.yaml \
      --index results/test_index.csv --out_root /pitsec_sose26_topic8/dataset/robust \
      --index_dir results/robust/
  $WTP_PY_DEFAKE scripts/robustness_perturb.py --mode score \
      --clean results/clean_pred.csv --perturbed results/jpeg30_pred.csv \
      --out results/robust/jpeg30_drop.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import io_utils, image_ops, metrics, schema  # noqa: E402

import pandas as pd  # noqa: E402


def _perturbations(config):
    rob = config.get("robustness", {})
    ops = []
    for q in rob.get("jpeg_quality", []):
        ops.append(("jpeg%d" % q, lambda im, q=q: image_ops.jpeg_recompress(im, q)))
    for s in rob.get("gaussian_blur_sigma", []):
        ops.append(("blur%g" % s, lambda im, s=s: image_ops.gaussian_blur(im, s)))
    for f in rob.get("resize_factors", []):
        ops.append(("resize%g" % f, lambda im, f=f: image_ops.resize_roundtrip(im, f)))
    for a in rob.get("sharpen", []):
        ops.append(("sharpen%g" % a, lambda im, a=a: image_ops.sharpen(im, a)))
    return ops


def generate(args, logger):
    config = io_utils.load_config(args.config)
    df = pd.read_csv(args.index)
    ops = _perturbations(config)
    logger.info("Generating %d perturbations for %d images", len(ops), len(df))

    indices = {name: [] for name, _ in ops}
    for _, row in df.iterrows():
        src = row[schema.PATH]
        try:
            img = image_ops.load_rgb(src)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skip %s (%s)", src, exc)
            continue
        for name, op in ops:
            out_dir = os.path.join(args.out_root, name, str(row[schema.DATASET]))
            io_utils.ensure_dir(out_dir)
            stem = os.path.splitext(os.path.basename(src))[0]
            out_path = os.path.join(out_dir, stem + ".png")
            try:
                image_ops.save_png(op(img), out_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed %s on %s (%s)", name, src, exc)
                continue
            new_row = row.to_dict()
            new_row[schema.PATH] = out_path
            new_row["source_path"] = src
            new_row["perturbation"] = name
            indices[name].append(new_row)

    io_utils.ensure_dir(args.index_dir)
    for name, recs in indices.items():
        out_csv = os.path.join(args.index_dir, "index_%s.csv" % name)
        pd.DataFrame(recs).to_csv(out_csv, index=False)
        logger.info("Wrote %s (%d rows)", out_csv, len(recs))


def score(args, logger):
    clean = pd.read_csv(args.clean)
    pert = pd.read_csv(args.perturbed)
    # Perturbed rows carry source_path pointing back to the clean image's full_path.
    key = "source_path" if "source_path" in pert.columns else schema.PATH
    merged = clean.merge(pert, left_on=schema.PATH, right_on=key,
                         suffixes=("_clean", "_pert"))
    if merged.empty:
        raise SystemExit("No aligned rows; check that perturbed predictions carry source_path.")

    pc = args.pred_col + "_clean"
    pp = args.pred_col + "_pert"
    flip = metrics.label_flip_rate(merged[pc].astype(str), merged[pp].astype(str))
    out = {"n": int(len(merged)), "label_flip_rate": flip}

    if args.conf_col:
        cc, cp = args.conf_col + "_clean", args.conf_col + "_pert"
        if cc in merged.columns and cp in merged.columns:
            out["confidence_drop"] = metrics.confidence_drop(
                pd.to_numeric(merged[cc], errors="coerce"),
                pd.to_numeric(merged[cp], errors="coerce"))

    # If ground truth label is present, also report accuracy drop (fake=positive).
    label_col = schema.LABEL + "_clean" if (schema.LABEL + "_clean") in merged.columns else (
        schema.LABEL if schema.LABEL in merged.columns else None)
    if label_col:
        yt = schema.is_fake_label(merged[label_col]).astype(int)
        acc_clean = (schema.is_fake_predict(merged[pc]).astype(int) == yt).mean()
        acc_pert = (schema.is_fake_predict(merged[pp]).astype(int) == yt).mean()
        out["accuracy_clean"] = float(acc_clean)
        out["accuracy_perturbed"] = float(acc_pert)
        out["performance_drop"] = metrics.performance_drop(acc_clean, acc_pert)

    io_utils.ensure_dir(os.path.dirname(os.path.abspath(args.out)))
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    logger.info("Robustness score: %s", out)


def main(args):
    logger = io_utils.setup_logging("robustness_perturb")
    if args.mode == "generate":
        if not (args.config and args.index and args.out_root):
            raise SystemExit("generate needs --config --index --out_root")
        generate(args, logger)
    elif args.mode == "score":
        if not (args.clean and args.perturbed and args.out):
            raise SystemExit("score needs --clean --perturbed --out")
        score(args, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robustness perturbations + scoring.")
    parser.add_argument("--mode", choices=["generate", "score"], required=True)
    # generate
    parser.add_argument("--config")
    parser.add_argument("--index")
    parser.add_argument("--out_root")
    parser.add_argument("--index_dir", default="results/robust/")
    # score
    parser.add_argument("--clean")
    parser.add_argument("--perturbed")
    parser.add_argument("--pred_col", default="defake_predict")
    parser.add_argument("--conf_col", default="prob_fake")
    parser.add_argument("--out")
    main(parser.parse_args())
