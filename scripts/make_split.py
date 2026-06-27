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
from lib import io_utils, schema  # noqa: E402

import pandas as pd  # noqa: E402


def main(args):
    logger = io_utils.setup_logging("make_split")
    config = io_utils.load_config(args.config)
    df = pd.read_csv(args.index)

    from sklearn.model_selection import train_test_split
    strat = df[schema.GENERATOR] if df[schema.GENERATOR].value_counts().min() >= 2 else None
    train_df, test_df = train_test_split(
        df, test_size=config.get("test_size", 0.2),
        stratify=strat, random_state=config.get("seed", 42))

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
    main(parser.parse_args())
