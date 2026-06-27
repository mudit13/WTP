#!/usr/bin/env python3
"""
Sample N images from a source directory into a destination directory by COPYING bytes
(no re-encode), preserving the original pixels and avoiding extra compression artifacts.

Used to build balanced, face-only real subsets (OpenForensics reals, DFFD reals) so the
"real" class is diverse but size-matched to the fake classes (GOLD concern #1).

Usage:
  /usr/bin/python3.9 scripts/sample_dataset.py \
      --src /share/DeepFake/DFFD_Images --glob "**/real/**/*.png" \
      --dst /pitsec_sose26_topic8/dffd_real --n 150 --seed 42
"""
import argparse
import glob
import os
import random
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import io_utils  # noqa: E402


def main(args):
    logger = io_utils.setup_logging("sample_dataset")
    matches = sorted(glob.glob(os.path.join(args.src, args.glob), recursive=True))
    matches = [m for m in matches if os.path.isfile(m)]
    if args.exclude:
        excluded = set(glob.glob(os.path.join(args.src, args.exclude), recursive=True))
        matches = [m for m in matches if m not in excluded]
    logger.info("Found %d candidate images under %s", len(matches), args.src)
    if not matches:
        raise SystemExit("No images matched. Check --src/--glob.")

    rng = random.Random(args.seed)
    if args.n > 0 and args.n < len(matches):
        chosen = rng.sample(matches, args.n)
    else:
        chosen = matches
        if args.n > len(matches):
            logger.warning("Requested %d but only %d available; copying all.",
                           args.n, len(matches))

    io_utils.ensure_dir(args.dst)
    copied = 0
    for src_path in chosen:
        # Flatten names but keep them unique by prefixing the parent dir.
        parent = os.path.basename(os.path.dirname(src_path))
        base = "%s__%s" % (parent, os.path.basename(src_path))
        shutil.copy2(src_path, os.path.join(args.dst, base))
        copied += 1
    logger.info("Copied %d images to %s", copied, args.dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample/copy images into a balanced subset.")
    parser.add_argument("--src", required=True, help="Source root directory")
    parser.add_argument("--glob", required=True, help="Glob relative to src (recursive ok)")
    parser.add_argument("--dst", required=True, help="Destination directory")
    parser.add_argument("--n", type=int, default=0, help="Number to sample (0 = all)")
    parser.add_argument("--exclude", default=None, help="Optional exclude glob")
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
