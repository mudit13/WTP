#!/usr/bin/env python3
"""
Build the London-DB <-> SD1.5-img2img identity sidecar.

Input metadata is written by generate_sd15_img2img.py. Output follows the shared
`full_path,source_image_id` schema consumed by io_utils.load_group_map.
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import io_utils  # noqa: E402


def build_rows(metadata_csv):
    by_path = {}
    with open(metadata_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError("Generation metadata is empty: %s" % metadata_csv)

    required = {"output_path", "source_image", "source_identity"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError("Metadata missing required column(s): %s" % ", ".join(sorted(missing)))

    for row in rows:
        identity = "londondb:%s" % row["source_identity"]
        for path_col in ("source_image", "output_path"):
            path = row[path_col]
            if not path:
                raise ValueError("Empty %s in metadata row %r" % (path_col, row))
            prior = by_path.get(path)
            if prior is not None and prior != identity:
                raise ValueError("Path %s maps to conflicting identities %s and %s"
                                 % (path, prior, identity))
            by_path[path] = identity
    return [{"full_path": path, "source_image_id": by_path[path]}
            for path in sorted(by_path)]


def main(args):
    logger = io_utils.setup_logging("make_img2img_group_map")
    records = build_rows(args.metadata)
    io_utils.ensure_dir(os.path.dirname(os.path.abspath(args.out)))
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["full_path", "source_image_id"])
        writer.writeheader()
        writer.writerows(records)

    n_groups = len(set(row["source_image_id"] for row in records))
    n_generated = sum("/sd15_img2img/" in row["full_path"].replace("\\", "/")
                      for row in records)
    logger.info("Wrote %s: rows=%d identities=%d generated_paths=%d",
                args.out, len(records), n_groups, n_generated)


if __name__ == "__main__":
    root = os.environ.get("WTP_ROOT", "/pitsec_sose26_topic8")
    parser = argparse.ArgumentParser(description="Build London/img2img identity group sidecar.")
    parser.add_argument("--metadata",
                        default=os.path.join(root, "dataset", "sd15_img2img", "metadata.csv"))
    parser.add_argument("--out",
                        default=os.path.join(root, "dataset", "sd15_img2img",
                                             "londondb_img2img_groups.csv"))
    main(parser.parse_args())
