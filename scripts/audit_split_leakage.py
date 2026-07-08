#!/usr/bin/env python3
"""
Split-leakage audit: detect train/val/test contamination and report per-split balance.

Near-duplicates across splits inflate every accuracy number. Our biggest risk is the SD/FLUX
generators (multiple seeds per prompt -> visually similar siblings) landing on both sides of the
split. This is a DIAGNOSTIC: we do NOT switch to identity-group splitting (no identity labels;
reals are drawn from large pools so identity collisions are improbable, and per-source grouping
is degenerate because source == generator == class). Dependency-free (PIL + numpy only).

Two ways to define the splits:
  - finetune (default): reconstruct the in-set train/val/test split exactly as
    finetune_defake_head.py does (config class space + content-stable hash split on full_path).
    Out-of-set generators present in the index are tagged `unseen`.
  - index_files: pass --train_index / --test_index (e.g. results/{train,test}_index.csv).

Reports: exact-duplicate (SHA-256) pairs across splits, near-duplicate (dHash Hamming <= thr)
pairs across splits, and per-generator / per-source counts per split.

Usage:
  $WTP_PY_DEFAKE scripts/audit_split_leakage.py --config configs/config.yaml \
      --index results/index_aspect.csv --out results/leakage_audit.json
  $WTP_PY_DEFAKE scripts/audit_split_leakage.py --mode index_files \
      --train_index results/train_index.csv --test_index results/test_index.csv \
      --out results/leakage_audit_binary.json
"""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import io_utils, defake_head, schema  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _dhash(path, size=8):
    """64-bit difference hash: row-wise brightness gradients, robust to resize/JPEG (aHash-style)."""
    from PIL import Image
    img = Image.open(path).convert("L").resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(img, dtype=np.int16)
    diff = a[:, 1:] > a[:, :-1]
    bits = 0
    for v in diff.flatten():
        bits = (bits << 1) | int(v)
    return bits


def _hamming(a, b):
    return bin(a ^ b).count("1")


def _finetune_splits(index_csv, config):
    """Reconstruct the in-set train/val/test split as finetune_defake_head.py does; out-of-set
    generators are tagged `unseen`. Returns a DataFrame with a `split` column."""
    df = pd.read_csv(index_csv)
    generator = df[schema.GENERATOR].astype(str).to_numpy()
    paths = df[schema.PATH].astype(str).to_numpy()
    attr = config.get("attribution", {}) or {}
    real_generators = set(attr.get("real_generators", []))
    trained_fakes = list(dict.fromkeys(
        list(attr.get("in_set_generators", [])) + list(attr.get("finetune_new_classes", []))))
    allowed = real_generators | set(trained_fakes)
    present = set(generator)
    classes = sorted(g for g in present if g in allowed)
    class_set = set(classes)
    in_mask = np.array([g in class_set for g in generator])

    split = np.array(["unseen"] * len(df), dtype=object)
    gi, pi = generator[in_mask], paths[in_mask]
    y = defake_head.encode_labels(gi, classes)
    tr, va, te = defake_head.stratified_split(
        y, test_size=config.get("test_size", 0.2),
        val_size=config.get("val_size", 0.1), seed=int(config.get("seed", 42)), keys=pi)
    in_positions = np.where(in_mask)[0]
    for local, name in [(tr, "train"), (va, "val"), (te, "test")]:
        for li in local:
            split[in_positions[li]] = name
    df = df.copy()
    df["split"] = split
    return df


def _index_file_splits(train_csv, test_csv):
    tr = pd.read_csv(train_csv)
    te = pd.read_csv(test_csv)
    tr["split"] = "train"
    te["split"] = "test"
    return pd.concat([tr, te], ignore_index=True)


def main(args):
    logger = io_utils.setup_logging("audit_split_leakage")
    if args.mode == "index_files":
        if not (args.train_index and args.test_index):
            raise SystemExit("--train_index and --test_index required for index_files mode")
        df = _index_file_splits(args.train_index, args.test_index)
    else:
        if not (args.index and args.config):
            raise SystemExit("--index and --config required for finetune mode")
        df = _finetune_splits(args.index, io_utils.load_config(args.config))

    rows, skipped = [], 0
    for _, r in df.iterrows():
        path = str(r[schema.PATH])
        try:
            sha = _sha256(path)
            dh = _dhash(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping %s (%s)", path, exc)
            skipped += 1
            continue
        rows.append({
            "path": path, "split": r["split"],
            "generator": str(r.get(schema.GENERATOR, "")),
            "source": str(r.get(schema.DATASET, "")),
            "sha": sha, "dhash": dh,
        })
    logger.info("Hashed %d images (%d skipped)", len(rows), skipped)

    # Exact duplicates that span more than one split.
    by_sha = defaultdict(list)
    for i, row in enumerate(rows):
        by_sha[row["sha"]].append(i)
    exact_cross = []
    for sha, idxs in by_sha.items():
        splits = {rows[i]["split"] for i in idxs}
        if len(idxs) > 1 and len(splits) > 1:
            exact_cross.append({"sha": sha,
                                "paths": [rows[i]["path"] for i in idxs],
                                "splits": sorted(splits)})

    # Near-duplicates across DIFFERENT splits (Hamming <= threshold on dHash).
    # Include "unseen" so near-duplicates between TRAIN and out-of-set generators are detected -
    # that train<->unseen bridge is exactly the population the out-of-set claims depend on.
    eval_splits = {"val", "test", "unseen"}
    near_cross = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if rows[i]["split"] == rows[j]["split"]:
                continue
            # Only care about pairs bridging an eval split and another split.
            if not (rows[i]["split"] in eval_splits or rows[j]["split"] in eval_splits):
                continue
            d = _hamming(rows[i]["dhash"], rows[j]["dhash"])
            if d <= args.hamming:
                near_cross.append({
                    "hamming": d,
                    "a": {"path": rows[i]["path"], "split": rows[i]["split"],
                          "generator": rows[i]["generator"]},
                    "b": {"path": rows[j]["path"], "split": rows[j]["split"],
                          "generator": rows[j]["generator"]},
                })
    near_cross.sort(key=lambda x: x["hamming"])

    # Per-split balance counts.
    per_split_gen = defaultdict(Counter)
    per_split_src = defaultdict(Counter)
    split_totals = Counter()
    for row in rows:
        per_split_gen[row["split"]][row["generator"]] += 1
        per_split_src[row["split"]][row["source"]] += 1
        split_totals[row["split"]] += 1

    out = {
        "mode": args.mode,
        "n_images": len(rows), "n_skipped": skipped,
        "hamming_threshold": args.hamming,
        "split_totals": dict(split_totals),
        "exact_cross_split_duplicates": {"count": len(exact_cross), "groups": exact_cross},
        "near_cross_split_duplicates": {
            "count": len(near_cross), "pairs": near_cross[:args.max_pairs]},
        "per_split_generator_counts": {s: dict(c) for s, c in per_split_gen.items()},
        "per_split_source_counts": {s: dict(c) for s, c in per_split_src.items()},
    }
    io_utils.ensure_dir(os.path.dirname(os.path.abspath(args.out)))
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    logger.info("LEAKAGE: exact cross-split=%d, near cross-split(<=%d)=%d -> %s",
                len(exact_cross), args.hamming, len(near_cross), args.out)
    if exact_cross:
        logger.warning("Exact cross-split duplicates found (%d groups) - investigate.",
                       len(exact_cross))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit train/val/test split for leakage.")
    parser.add_argument("--mode", choices=["finetune", "index_files"], default="finetune")
    parser.add_argument("--config", default=None)
    parser.add_argument("--index", default=None, help="Index CSV for finetune-split reconstruction")
    parser.add_argument("--train_index", default=None)
    parser.add_argument("--test_index", default=None)
    parser.add_argument("--hamming", type=int, default=6,
                        help="Max dHash Hamming distance to flag a near-duplicate (0-64)")
    parser.add_argument("--max_pairs", type=int, default=200,
                        help="Cap on near-duplicate pairs written to JSON")
    parser.add_argument("--out", required=True)
    main(parser.parse_args())
