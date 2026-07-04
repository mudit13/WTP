#!/usr/bin/env python3
"""
Preprocessing study (GOLD concern #2): emit PNG variants of every image.

  variant A "scaled"   : whole image resized to common_size x common_size (bicubic) - SQUASHES
                         (distorts) non-square images; kept as the "uncontrolled" reference.
  variant B "cropped"  : center crop of common_size x common_size (no interpolation)
  variant C "aspect"   : resize shortest side to common_size, then center-crop (aspect-
                         PRESERVING; no stretch). Recommended for the confound-controlled runs
                         because it removes both the format AND the aspect-ratio confound
                         (supervisor feedback: squashing risks replacing one confound with
                         another, since only non-square reals get stretched).

All are lossless PNG so we never stack JPEG artifacts. Running all downstream detection /
attribution on the variants and comparing deltas is how we test whether detectors react to
generator traces or to preprocessing.

Reads the master CSV (schema: filename, full_path, label, generator, category,
source_dataset, width, height) and writes one variant index CSV per variant with full_path
repointed to the derived PNG (plus source_path + variant columns).

Usage:
  $WTP_PY_DEFAKE scripts/prepare_variants.py --config configs/config.yaml \
      --master /pitsec_sose26_topic8/dataset/master_metadata.csv \
      --out_root /pitsec_sose26_topic8/dataset/variants --index_dir results/
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import io_utils, image_ops, schema  # noqa: E402

import pandas as pd  # noqa: E402


def _dest_path(out_root, variant, source_dataset, src_path):
    stem = os.path.splitext(os.path.basename(src_path))[0]
    dataset_dir = os.path.join(out_root, variant, str(source_dataset))
    io_utils.ensure_dir(dataset_dir)
    return os.path.join(dataset_dir, stem + ".png")


def main(args):
    logger = io_utils.setup_logging("prepare_variants")
    config = io_utils.load_config(args.config)
    size = int(config.get("common_size", 512))
    df = pd.read_csv(args.master)

    variants = {"scaled": image_ops.scale_to,
                "cropped": image_ops.center_crop,
                "aspect": image_ops.resize_shortest_center_crop}
    rows = {name: [] for name in variants}

    n_ok, n_fail = 0, 0
    for _, row in df.iterrows():
        src = row[schema.PATH]
        try:
            img = image_ops.load_rgb(src)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping unreadable %s (%s)", src, exc)
            n_fail += 1
            continue
        for variant, op in variants.items():
            out_path = _dest_path(args.out_root, variant, row[schema.DATASET], src)
            try:
                image_ops.save_png(op(img, size), out_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed %s variant for %s (%s)", variant, src, exc)
                continue
            new_row = row.to_dict()
            new_row[schema.PATH] = out_path
            new_row["source_path"] = src
            new_row["variant"] = variant
            new_row[schema.WIDTH] = size
            new_row[schema.HEIGHT] = size
            rows[variant].append(new_row)
        n_ok += 1
        if n_ok % 200 == 0:
            logger.info("Processed %d images", n_ok)

    io_utils.ensure_dir(args.index_dir)
    for variant, recs in rows.items():
        out_csv = os.path.join(args.index_dir, "index_%s.csv" % variant)
        pd.DataFrame(recs).to_csv(out_csv, index=False)
        logger.info("Wrote %s (%d rows)", out_csv, len(recs))
    logger.info("Done. ok=%d fail=%d size=%d", n_ok, n_fail, size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emit scaled + cropped PNG variants.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--master", required=True, help="master_metadata.csv")
    parser.add_argument("--out_root", required=True, help="Where variant PNGs are written")
    parser.add_argument("--index_dir", default="results/", help="Where index CSVs go")
    main(parser.parse_args())
