#!/usr/bin/env python3
"""
Auto-fill the measurable parts of the per-dataset datasheets from master_metadata.csv.

Emits results/datasheets.md with one section per dataset, pre-filling count, resolution
statistics, and on-disk format. Provenance/processing-history fields are left as TODO
markers to be completed manually (and confirmed with the supervisor), per docs/DATASHEET_TEMPLATE.md.

Usage:
  /usr/bin/python3.9 scripts/make_datasheets.py --metadata results/master_metadata.csv \
      --out results/datasheets.md
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import io_utils, schema  # noqa: E402

import pandas as pd  # noqa: E402


def _fmt_resolution(group):
    wh = (group[schema.WIDTH].astype(str) + "x" + group[schema.HEIGHT].astype(str))
    common = wh.value_counts()
    parts = ["%s (%d)" % (res, cnt) for res, cnt in common.head(3).items()]
    span = "min %dx%d / max %dx%d" % (
        group[schema.WIDTH].min(), group[schema.HEIGHT].min(),
        group[schema.WIDTH].max(), group[schema.HEIGHT].max(),
    )
    return "; ".join(parts) + " | " + span


def main(args):
    logger = io_utils.setup_logging("make_datasheets")
    df = pd.read_csv(args.metadata)

    lines = ["# Auto-generated dataset datasheets",
             "",
             "Measurable fields are auto-filled. Replace every TODO with confirmed",
             "provenance (see docs/DATASHEET_TEMPLATE.md). Confirm with the supervisor.",
             ""]
    for name, group in df.groupby(schema.DATASET):
        exts = group[schema.PATH].apply(lambda p: os.path.splitext(p)[1].lower())
        fmt_counts = exts.value_counts().to_dict()
        lines += [
            "## %s" % name,
            "",
            "- Role: %s" % group[schema.LABEL].iloc[0],
            "- Generator / source: %s" % group[schema.GENERATOR].iloc[0],
            "- Category: %s" % group[schema.CATEGORY].iloc[0],
            "- Count: %d" % len(group),
            "- Resolution: %s" % _fmt_resolution(group),
            "- On-disk format(s): %s" % fmt_counts,
            "",
            "### Provenance (manual)",
            "- Generation/sensor pipeline: TODO",
            "- Source resize/crop: TODO",
            "- Source compression (JPEG q?): TODO",
            "- Alignment / face-crop / watermark removal: TODO",
            "- License / access route: TODO",
            "",
        ]
    io_utils.ensure_dir(os.path.dirname(os.path.abspath(args.out)))
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    logger.info("Wrote %s (%d datasets)", args.out, df[schema.DATASET].nunique())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-fill dataset datasheets.")
    parser.add_argument("--metadata", required=True, help="master_metadata.csv path")
    parser.add_argument("--out", required=True, help="Output markdown path")
    main(parser.parse_args())
