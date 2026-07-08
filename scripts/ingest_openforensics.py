#!/usr/bin/env python3
"""
Sort the flat OpenForensics face crops into real/ and fake/ subdirectories so the existing
config-driven build_master_index.py can pick them up with correct labels and CONTAINER paths.

Background: the extraction script writes every crop into ONE flat directory with the real/fake
label recorded only in openforensics_metadata.csv (and a host-side `full_path`). Our pipeline,
however, is directory-driven: configs/config.yaml has `openforensics_real` (dir .../openforensics/
real) and `openforensics_fake` (dir .../openforensics/fake) dataset entries, and
build_master_index.py scans those dirs, measures width/height, dedups, and reconciles. This
helper bridges the two: it reads the metadata CSV and places each crop under
<out_root>/{real,fake}/ by its label. After this, just run build_master_index.py - no bespoke
merge, no host-path leakage into master_metadata.csv.

Run wherever both the crops and the container-visible dataset root are reachable (typically the
host, then confirm the container sees <out_root>). Files are matched by BASENAME, so the CSV's
host `full_path` prefix does not matter; point --crops_dir at wherever the crops actually live.

Usage:
  python scripts/ingest_openforensics.py \
      --crops_csv ./openforensics_cropped/openforensics_metadata.csv \
      --crops_dir ./openforensics_cropped \
      --out_root /vol2/pitsec_sose26_topic8/sharedDockerDir/dataset/openforensics \
      --mode symlink
"""
import argparse
import csv
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import io_utils  # noqa: E402

VALID_LABELS = ("real", "fake")


def main(args):
    logger = io_utils.setup_logging("ingest_openforensics")
    counts = {"real": 0, "fake": 0}
    skipped = 0
    for label in VALID_LABELS:
        io_utils.ensure_dir(os.path.join(args.out_root, label))

    with open(args.crops_csv, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            label = str(row.get("label", "")).strip().lower()
            if label not in VALID_LABELS:
                skipped += 1
                continue
            fname = os.path.basename(str(row.get("full_path") or row.get("filename") or ""))
            if not fname:
                skipped += 1
                continue
            src = os.path.join(args.crops_dir, fname)
            if not os.path.isfile(src):
                logger.warning("Missing crop (skip): %s", src)
                skipped += 1
                continue
            dst = os.path.join(args.out_root, label, fname)
            if os.path.exists(dst) and not args.overwrite:
                counts[label] += 1
                continue
            try:
                if args.mode == "copy":
                    shutil.copy2(src, dst)
                elif args.mode == "move":
                    shutil.move(src, dst)
                else:  # symlink
                    if os.path.islink(dst) or os.path.exists(dst):
                        os.remove(dst)
                    os.symlink(os.path.abspath(src), dst)
            except OSError as exc:
                logger.warning("Failed placing %s -> %s (%s)", src, dst, exc)
                skipped += 1
                continue
            counts[label] += 1

    logger.info("Placed real=%d fake=%d (skipped=%d) under %s [%s]",
                counts["real"], counts["fake"], skipped, args.out_root, args.mode)
    logger.info("Next: run build_master_index.py (config already has openforensics_real + "
                "openforensics_fake), then metadata_confound_probe.py on the OpenForensics rows.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sort flat OpenForensics crops into real/ and fake/ for build_master_index.")
    parser.add_argument("--crops_csv", required=True,
                        help="openforensics_metadata.csv from the extraction script")
    parser.add_argument("--crops_dir", required=True,
                        help="Directory that actually contains the crop files (matched by basename)")
    parser.add_argument("--out_root", required=True,
                        help="Container-visible .../dataset/openforensics (real/ + fake/ created here)")
    parser.add_argument("--mode", choices=["symlink", "copy", "move"], default="symlink",
                        help="How to place files (symlink keeps one copy; move if disk-bound)")
    parser.add_argument("--overwrite", action="store_true")
    main(parser.parse_args())
