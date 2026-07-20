#!/usr/bin/env python3
"""
Create stratified train/test index CSVs from an index (master_metadata.csv or a variant
index). Robustness perturbations and any from-scratch training must only touch the test
split so we never leak perturbed/augmented data into training.

Usage:
  /usr/bin/python3.9 scripts/make_split.py --config configs/config.yaml \
      --index results/index_scaled.csv \
      --train_out results/train_index.csv --test_out results/test_index.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import defake_head, io_utils, schema  # noqa: E402

import pandas as pd  # noqa: E402


def main(args):
    logger = io_utils.setup_logging("make_split")
    config = io_utils.load_config(args.config)
    df = pd.read_csv(args.index)
    seed = config.get("seed", 42)

    # Group-aware split (same-source-photo coupling fix, e.g. OpenForensics real+fake crop
    # pairs kept on the SAME side of train/test); see finetune_defake_head.py for details.
    # No-op when no sidecar is found, or falls back to sklearn's plain stratified split when a
    # class has <2 members (the content-stable hash split needs stratify-by-class like sklearn
    # does, but degenerate single-member classes are rare/edge-case only).
    group_map_paths = args.group_map if args.group_map else io_utils.default_group_map_paths(config)
    group_map = io_utils.load_group_map(group_map_paths, logger)
    paths = df[schema.PATH].astype(str).to_numpy()
    # Lookup via source_path when --index is a variant index (its full_path points at a derived
    # file the sidecar never knew about) - see io_utils.group_lookup_map_from_df.
    lookup_map = io_utils.group_lookup_map_from_df(df)
    groups = (io_utils.apply_group_map_with_lookup(paths, lookup_map, group_map, logger=logger)
             if group_map else None)

    if df[schema.GENERATOR].value_counts().min() >= 2:
        y = defake_head.encode_labels(df[schema.GENERATOR].astype(str).to_numpy(),
                                      sorted(df[schema.GENERATOR].unique()))
        tr_idx, _, te_idx = defake_head.stratified_split(
            y, test_size=config.get("test_size", 0.2), val_size=0.0, seed=seed,
            keys=paths, groups=groups)
        n_checked = defake_head.assert_no_group_straddle(
            groups, {"train": tr_idx, "test": te_idx}, keys=paths)
        logger.info("Post-split group assertion passed (%d explicit groups)", n_checked)
        train_df, test_df = df.iloc[tr_idx], df.iloc[te_idx]
    else:
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(
            df, test_size=config.get("test_size", 0.2), stratify=None, random_state=seed)
        if group_map:
            logger.warning("A generator class has <2 members; fell back to sklearn's "
                           "non-group-aware split (group_map was loaded but NOT applied).")

    for path, frame in [(args.train_out, train_df), (args.test_out, test_df)]:
        io_utils.ensure_dir(os.path.dirname(os.path.abspath(path)))
        frame.to_csv(path, index=False)
        logger.info("Wrote %s (%d rows)", path, len(frame))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stratified train/test split of an index.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--train_out", required=True)
    parser.add_argument("--test_out", required=True)
    parser.add_argument("--group_map", nargs="*", default=None,
                        help="Path(s) to full_path,source_image_id sidecar CSV(s) for "
                             "group-aware splitting. Default: auto-load "
                             "<dataset_root>/openforensics/openforensics_groups.csv if present.")
    main(parser.parse_args())
