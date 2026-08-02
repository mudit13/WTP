#!/usr/bin/env python3
"""
Generate a byte-level manifest of the EXACT mounted files an authoritative run depends on, so a
supervisor can verify their own server's dataset mount (and checkpoints) match this run WITHOUT
this repository ever redistributing the underlying licensed images or weights.

Two manifest kinds:
  --mode data         One row per image in --index: absolute path (as recorded at manifest time),
                       path normalized relative to the dataset root, label, generator,
                       source_dataset, group_id (when a group sidecar covers it), file size, and
                       SHA-256.
  --mode checkpoints   One row per required weight file (default: the two paths
                       configs/paths.env declares as WTP_DEFAKE_CLIP_LINEAR and
                       WTP_DEFAKE_FINETUNE_CLIP; override with --paths).

Store the resulting manifest CSV + summary JSON in the run's evidence directory, NOT the
underlying files. `scripts/verify_handover.py` re-hashes the mounted files against this manifest
to give the supervisor byte-level verification using their own existing server access.

Usage:
  $PY scripts/create_data_manifest.py --mode data --config configs/config.yaml \
      --index results/2026-08-01_eightway_v1/index_aspect.csv \
      --out results/2026-08-01_eightway_v1/data_manifest.csv

  $PY scripts/create_data_manifest.py --mode checkpoints \
      --out results/2026-08-01_eightway_v1/checkpoint_manifest.csv
"""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import io_utils, schema  # noqa: E402

import pandas as pd  # noqa: E402


def _sha256(path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize(path, root):
    """Path relative to `root` with forward slashes, when `path` is under `root`; otherwise the
    original path unchanged. Purely for human-readable/portable display - verification itself
    re-opens the recorded absolute `path`, which only resolves on the SAME server/mount."""
    if not root:
        return path
    try:
        rel = os.path.relpath(path, root)
    except ValueError:
        return path
    if rel.startswith(".."):
        return path
    return rel.replace(os.sep, "/")


def _hash_row(path, logger):
    if not os.path.isfile(path):
        logger.warning("Missing file (recorded as missing in manifest): %s", path)
        return None, True
    try:
        return _sha256(path), False
    except OSError as exc:
        logger.warning("Could not read %s (%s); recording as missing.", path, exc)
        return None, True


def build_data_manifest(index_csv, config, group_map_paths, logger):
    df = pd.read_csv(index_csv)
    dataset_root = config.get("dataset_root")

    gm_paths = group_map_paths if group_map_paths else io_utils.default_group_map_paths(config)
    group_map = io_utils.load_group_map(gm_paths, logger)
    lookup_map = io_utils.group_lookup_map_from_df(df)
    paths = df[schema.PATH].astype(str).to_numpy()
    groups = (io_utils.apply_group_map_with_lookup(paths, lookup_map, group_map, logger=logger)
             if group_map else [""] * len(df))

    rows = []
    n_missing = 0
    for i, record in enumerate(df.to_dict("records")):
        path = str(record[schema.PATH])
        sha256, missing = _hash_row(path, logger)
        n_missing += int(missing)
        size = os.path.getsize(path) if not missing else None
        group_id = groups[i] if group_map else ""
        # A group id equal to the row's own path just means "no coupling recorded" (see
        # io_utils.apply_group_map_with_lookup); report that as an empty group, not a fake id.
        if group_id == path:
            group_id = ""
        rows.append({
            "path": path,
            "relative_path": _normalize(path, dataset_root),
            "label": record.get(schema.LABEL),
            "generator": record.get(schema.GENERATOR),
            "source_dataset": record.get(schema.DATASET),
            "group_id": group_id,
            "size_bytes": size,
            "sha256": sha256,
            "missing": missing,
        })
    manifest = pd.DataFrame(rows)
    summary = {
        "kind": "data",
        "created_at": datetime.now().isoformat(),
        "index_csv": os.path.abspath(index_csv),
        "n_rows": len(manifest),
        "n_missing": n_missing,
        "total_bytes": int(manifest["size_bytes"].fillna(0).sum()),
        "counts_by_label": Counter(manifest["label"].fillna("<unknown>")).most_common(),
        "counts_by_generator": Counter(manifest["generator"].fillna("<unknown>")).most_common(),
    }
    return manifest, summary


def build_checkpoint_manifest(paths, logger):
    rows = []
    n_missing = 0
    for path in paths:
        sha256, missing = _hash_row(path, logger)
        n_missing += int(missing)
        size = os.path.getsize(path) if not missing else None
        rows.append({"path": path, "size_bytes": size, "sha256": sha256, "missing": missing})
    manifest = pd.DataFrame(rows)
    summary = {
        "kind": "checkpoints",
        "created_at": datetime.now().isoformat(),
        "n_rows": len(manifest),
        "n_missing": n_missing,
        "total_bytes": int(manifest["size_bytes"].fillna(0).sum()),
    }
    return manifest, summary


def _default_checkpoint_paths():
    root = os.environ.get("WTP_ROOT", "/pitsec_sose26_topic8")
    return [
        os.environ.get("WTP_DEFAKE_CLIP_LINEAR", os.path.join(root, "models", "clip_linear.pt")),
        os.environ.get("WTP_DEFAKE_FINETUNE_CLIP", os.path.join(root, "models", "finetune_clip.pt")),
    ]


def main(args):
    logger = io_utils.setup_logging("create_data_manifest")

    if args.mode == "data":
        if not args.index:
            raise SystemExit("--mode data requires --index")
        config = io_utils.load_config(args.config)
        manifest, summary = build_data_manifest(args.index, config, args.group_map, logger)
    else:
        paths = args.paths if args.paths else _default_checkpoint_paths()
        manifest, summary = build_checkpoint_manifest(paths, logger)

    io_utils.ensure_dir(os.path.dirname(os.path.abspath(args.out)) or ".")
    manifest.to_csv(args.out, index=False)
    summary_path = os.path.splitext(args.out)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Wrote %s (%d rows, %d missing) + %s", args.out, summary["n_rows"],
               summary["n_missing"], summary_path)
    if summary["n_missing"]:
        logger.warning("%d file(s) were missing at manifest-creation time; see 'missing' column.",
                       summary["n_missing"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a byte-level data/checkpoint manifest for handover verification.")
    parser.add_argument("--mode", choices=["data", "checkpoints"], default="data")
    parser.add_argument("--config", default="configs/config.yaml", help="Required for --mode data")
    parser.add_argument("--index", default=None,
                        help="Index CSV to manifest (--mode data), e.g. results/<run>/index_aspect.csv")
    parser.add_argument("--group_map", nargs="*", default=None,
                        help="Override group-map sidecar path(s) (--mode data); default: "
                             "io_utils.default_group_map_paths(config)")
    parser.add_argument("--paths", nargs="*", default=None,
                        help="Checkpoint file path(s) (--mode checkpoints); default: "
                             "WTP_DEFAKE_CLIP_LINEAR + WTP_DEFAKE_FINETUNE_CLIP")
    parser.add_argument("--out", required=True,
                        help="Output manifest CSV path; a sibling *_summary.json is also written")
    main(parser.parse_args())
