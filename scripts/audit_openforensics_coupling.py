#!/usr/bin/env python3
"""
OpenForensics same-source-photo coupling audit.

OpenForensics scene photos contain MULTIPLE face annotations - some genuine
(category_id 0 -> our "real"/OpenForensics), some manipulated (category_id 1 ->
"fake"/OpenForensics-fake). extract_openforensics.py crops each annotation independently and
names the crop by ANNOTATION id (openforensics_<split>_<ann_id>.jpg), dropping the source
IMAGE id. Our train/val/test split then keys on the crop's full_path, so a real crop and a fake
crop cropped from the EXACT SAME source photograph (same camera, lighting, background, JPEG
history) can land on opposite sides of the split. That is a same-source leak the dHash
near-duplicate audit (audit_split_leakage.py) will NOT catch, because a real face crop and a
fake face crop from one photo are different image REGIONS (different pixels, often different
subjects even) - they look nothing alike under perceptual hashing even though they share
acquisition statistics.

This script quantifies the coupling WITHOUT re-running extraction: it re-parses the ORIGINAL
OpenForensics polygon JSON(s) (metadata only, no image decoding) to recover annotation_id ->
image_id, matches that against the already-extracted crop filenames (the ann_id is embedded in
the filename), and cross-references with the current train/val/test split.

Reports:
  - how many source photos contributed more than one crop to our sample
  - how many of those photos contributed BOTH a real and a fake crop (the coupled population)
  - of THOSE, how many have the real crop and the fake crop on DIFFERENT sides of the split
    (the actual leak: e.g. real in train, fake in test) vs. the same side (statistically
    dependent but not a train/test leak)
  - a handful of concrete example groups for spot-checking

Usage (needs the ORIGINAL OpenForensics polygon JSON(s), e.g. on the host where
/vol1/share/DeepFake/OpenForensics is mounted, or a copy):
  python3 scripts/audit_openforensics_coupling.py \
      --polygon_json /vol1/share/DeepFake/OpenForensics/Val_poly.json \
      --config configs/config.yaml --index results/index_aspect.csv \
      --out results/of_coupling_audit.json

  # or against a fixed train/test split pair instead of reconstructing the finetune split:
  python3 scripts/audit_openforensics_coupling.py \
      --polygon_json /vol1/share/DeepFake/OpenForensics/Val_poly.json \
      --mode index_files --train_index results/train_index.csv \
      --test_index results/test_index.csv --out results/of_coupling_audit_binary.json
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import io_utils, schema  # noqa: E402
from audit_split_leakage import _finetune_splits, _index_file_splits  # noqa: E402

import pandas as pd  # noqa: E402

# Matches the filename convention written by extract_openforensics.py:
#   openforensics_<Split>_<ann_id>.jpg
_ANN_ID_RE = re.compile(r"openforensics_[^_]+_(\d+)\.")

OF_GENERATORS = {"OpenForensics", "OpenForensics-fake"}


def _load_ann_to_image(polygon_json_paths, logger):
    """annotation_id -> image_id, unioned across every given polygon JSON (metadata only)."""
    ann_to_image = {}
    for jp in polygon_json_paths:
        with open(jp, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        n = 0
        for ann in data.get("annotations", []):
            ann_to_image[int(ann["id"])] = int(ann["image_id"])
            n += 1
        logger.info("Parsed %s: %d annotations", jp, n)
    return ann_to_image


def _ann_id_from_path(path):
    m = _ANN_ID_RE.search(os.path.basename(str(path)))
    return int(m.group(1)) if m else None


def main(args):
    logger = io_utils.setup_logging("audit_openforensics_coupling")

    if args.mode == "index_files":
        if not (args.train_index and args.test_index):
            raise SystemExit("--train_index and --test_index required for index_files mode")
        df = _index_file_splits(args.train_index, args.test_index)
    else:
        if not (args.index and args.config):
            raise SystemExit("--index and --config required for finetune mode")
        df = _finetune_splits(args.index, io_utils.load_config(args.config))

    of_mask = df[schema.GENERATOR].astype(str).isin(OF_GENERATORS)
    of_df = df[of_mask].copy()
    logger.info("OpenForensics rows in this index/split: %d (of %d total)", len(of_df), len(df))
    if of_df.empty:
        raise SystemExit("No OpenForensics rows found in the given index/split.")

    ann_to_image = _load_ann_to_image(args.polygon_json, logger)

    of_df["ann_id"] = of_df[schema.PATH].apply(_ann_id_from_path)
    n_unparsed = int(of_df["ann_id"].isna().sum())
    if n_unparsed:
        logger.warning("%d/%d OF rows did not match the expected filename pattern "
                       "(openforensics_<split>_<ann_id>.jpg); dropped from the audit.",
                       n_unparsed, len(of_df))
    of_df = of_df.dropna(subset=["ann_id"]).copy()
    of_df["ann_id"] = of_df["ann_id"].astype(int)

    of_df["image_id"] = of_df["ann_id"].map(ann_to_image)
    n_unmatched = int(of_df["image_id"].isna().sum())
    if n_unmatched:
        logger.warning("%d/%d OF rows had an ann_id NOT found in --polygon_json (wrong/partial "
                       "JSON set for the split(s) actually used at extraction time?); dropped.",
                       n_unmatched, len(of_df))
    of_df = of_df.dropna(subset=["image_id"]).copy()
    of_df["image_id"] = of_df["image_id"].astype(int)
    logger.info("Matched %d/%d OF rows to a source image_id", len(of_df), int(of_mask.sum()))

    groups = defaultdict(list)
    for _, r in of_df.iterrows():
        groups[int(r["image_id"])].append({
            "path": str(r[schema.PATH]),
            "generator": str(r[schema.GENERATOR]),
            "label": str(r.get(schema.LABEL, "")),
            "split": str(r["split"]),
        })

    n_photos_total = len(groups)
    multi = {iid: rows for iid, rows in groups.items() if len(rows) > 1}
    both_classes = {}
    for iid, rows in multi.items():
        labels = {row["label"] for row in rows}
        if {"real", "fake"} <= labels or len({row["generator"] for row in rows}) > 1:
            both_classes[iid] = rows

    straddling = {}
    same_side = {}
    train_test_bridge = {}
    for iid, rows in both_classes.items():
        splits = {row["split"] for row in rows}
        real_splits = {row["split"] for row in rows if row["label"] == "real"}
        fake_splits = {row["split"] for row in rows if row["label"] == "fake"}
        if len(splits) > 1:
            straddling[iid] = rows
            # Direct train<->{val,test,unseen} bridge between the real and fake crop of ONE photo.
            if real_splits and fake_splits and real_splits != fake_splits:
                train_test_bridge[iid] = rows
        else:
            same_side[iid] = rows

    def _examples(d, cap):
        out = []
        for iid, rows in list(d.items())[:cap]:
            out.append({"image_id": iid, "crops": rows})
        return out

    result = {
        "n_of_rows_in_split": int(len(of_df)),
        "n_unparsed_filename": n_unparsed,
        "n_unmatched_ann_id": n_unmatched,
        "n_source_photos_total": n_photos_total,
        "n_source_photos_multi_crop": len(multi),
        "n_source_photos_real_and_fake": len(both_classes),
        "n_real_fake_pairs_straddling_splits": len(straddling),
        "n_real_fake_pairs_same_split": len(same_side),
        "n_real_fake_pairs_train_test_bridge": len(train_test_bridge),
        "fraction_of_coupled_photos_that_leak_across_splits": (
            len(straddling) / len(both_classes) if both_classes else None),
        "interpretation": (
            "n_source_photos_real_and_fake = source photos that contributed BOTH a real and a "
            "fake crop to our sample (the coupling the dHash audit cannot see, since the two "
            "crops are different face regions). n_real_fake_pairs_straddling_splits = of those, "
            "how many have the real crop and the fake crop on DIFFERENT split sides -- that "
            "count IS the direct real<->fake / train<->test leak magnitude. If this is 0 (or "
            "near it) relative to n_source_photos_real_and_fake, the coupling is present in "
            "principle but did not materially bridge our actual split; if it is a large "
            "fraction, headline OpenForensics numbers should be treated as optimistic and a "
            "group-aware re-extraction is warranted."),
        "example_straddling_groups": _examples(straddling, args.max_examples),
        "example_same_split_groups": _examples(same_side, min(5, args.max_examples)),
    }
    io_utils.ensure_dir(os.path.dirname(os.path.abspath(args.out)))
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    logger.info(
        "OF coupling: %d source photos have both real+fake crops; %d of those STRADDLE the "
        "split (leak); %d stay on one side -> %s",
        len(both_classes), len(straddling), len(same_side), args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quantify OpenForensics same-source-photo real/fake split coupling.")
    parser.add_argument("--polygon_json", nargs="+", required=True,
                        help="Path(s) to the ORIGINAL OpenForensics *_poly.json file(s) covering "
                             "every split extract_openforensics.py was run with (e.g. Val_poly.json "
                             "if --splits Val was used).")
    parser.add_argument("--mode", choices=["finetune", "index_files"], default="finetune")
    parser.add_argument("--config", default=None)
    parser.add_argument("--index", default=None, help="Index CSV for finetune-split reconstruction")
    parser.add_argument("--train_index", default=None)
    parser.add_argument("--test_index", default=None)
    parser.add_argument("--max_examples", type=int, default=20,
                        help="Cap on example groups written to the output JSON.")
    parser.add_argument("--out", required=True)
    main(parser.parse_args())
