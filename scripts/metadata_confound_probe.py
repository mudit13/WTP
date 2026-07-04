#!/usr/bin/env python3
"""
Metadata-only confound probe: how separable is real vs fake from IMAGE METADATA ALONE
(width, height, aspect ratio, on-disk format), with NO pixels?

This is the direct, measured answer to the supervisor's question ("did the data actually turn
out separable by format/resolution, or does the model only partially use it?"). It converts the
confound from an assertion into a number:

  - Run it on the RAW master (original width/height + original .png/.jpg extension). A HIGH
    balanced accuracy here = the label leaks strongly from metadata alone, i.e. a detector COULD
    cheat on format/resolution instead of content. This bounds the confound.
  - Run it on a NORMALIZED variant index (every image 256x256 PNG). Accuracy should collapse to
    ~chance, which is the evidence that the preprocessing pipeline REMOVES the metadata leak.

The gap between the two is the measurement. No CLIP/torch needed - sklearn only.

Usage (server, venv_sd15):
  $WTP_PY_DEFAKE scripts/metadata_confound_probe.py --config configs/config.yaml \
      --metadata /pitsec_sose26_topic8/dataset/master_metadata.csv \
      --out_dir results/confound_probe_raw/
  # control (should be ~chance):
  $WTP_PY_DEFAKE scripts/metadata_confound_probe.py --config configs/config.yaml \
      --metadata results/index_aspect.csv --out_dir results/confound_probe_aspect/
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import io_utils, schema  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _build_features(df, logger):
    """Return (X, feature_names, y, note). Features are metadata-only; missing width/height are
    measured from the file if present, else the row is dropped with a warning."""
    df = df.copy()

    # On-disk format from the extension (fakes are all PNG in this project; JPEG only in reals).
    ext = df[schema.PATH].astype(str).str.lower().str.rsplit(".", n=1).str[-1]
    df["is_png"] = (ext == "png").astype(int)
    df["is_jpeg"] = ext.isin(["jpg", "jpeg"]).astype(int)

    # Width/height: use the columns if present, else measure from the file (PIL) when available.
    if schema.WIDTH not in df.columns or schema.HEIGHT not in df.columns:
        logger.warning("No width/height columns; measuring from files (slower).")
        from PIL import Image
        ws, hs = [], []
        for p in df[schema.PATH].astype(str):
            try:
                with Image.open(p) as im:
                    ws.append(im.width)
                    hs.append(im.height)
            except Exception:  # noqa: BLE001
                ws.append(np.nan)
                hs.append(np.nan)
        df[schema.WIDTH] = ws
        df[schema.HEIGHT] = hs

    df = df.dropna(subset=[schema.WIDTH, schema.HEIGHT]).copy()
    df[schema.WIDTH] = df[schema.WIDTH].astype(float)
    df[schema.HEIGHT] = df[schema.HEIGHT].astype(float)
    df["aspect"] = df[schema.WIDTH] / df[schema.HEIGHT].replace(0, np.nan)
    df["log_area"] = np.log(np.clip(df[schema.WIDTH] * df[schema.HEIGHT], 1, None))

    feat_names = [schema.WIDTH, schema.HEIGHT, "aspect", "log_area", "is_png", "is_jpeg"]
    X = df[feat_names].to_numpy(dtype=float)
    y = schema.is_fake_label(df[schema.LABEL]).astype(int).to_numpy()
    return X, feat_names, y, df


def main(args):
    logger = io_utils.setup_logging("metadata_confound_probe")
    io_utils.ensure_dir(args.out_dir)
    config = io_utils.load_config(args.config)
    seed = int(config.get("seed", 42))

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(args.metadata)
    if schema.LABEL not in df.columns:
        raise SystemExit("Metadata CSV needs a '%s' column." % schema.LABEL)
    X, feat_names, y, df = _build_features(df, logger)
    n_fake = int(y.sum())
    n_real = int((y == 0).sum())
    logger.info("Rows: %d (real=%d, fake=%d); features: %s", len(y), n_real, n_fake, feat_names)
    if n_fake == 0 or n_real == 0:
        raise SystemExit("Need both classes present to probe separability.")

    # Split on INDICES so we can map test rows back to df for the per-source breakdown.
    idx = np.arange(len(y))
    tr_idx, te_idx = train_test_split(
        idx, test_size=config.get("test_size", 0.2), stratify=y, random_state=seed)
    clf = RandomForestClassifier(n_estimators=300, random_state=seed, class_weight="balanced")
    clf.fit(X[tr_idx], y[tr_idx])
    proba = clf.predict_proba(X[te_idx])[:, 1]
    pred = (proba >= 0.5).astype(int)
    yte = y[te_idx]

    bal_acc = float(balanced_accuracy_score(yte, pred))
    auroc = float(roc_auc_score(yte, proba)) if len(np.unique(yte)) == 2 else float("nan")
    importances = dict(sorted(zip(feat_names, [float(v) for v in clf.feature_importances_]),
                              key=lambda kv: kv[1], reverse=True))

    # Per-source leakage: fraction of each generator's TEST rows the metadata-only model calls
    # fake. A real source with a high rate is being separated by metadata (leak), not content.
    per_source = {}
    if schema.GENERATOR in df.columns:
        te_df = df.iloc[te_idx].copy()
        te_df["_pred_fake"] = pred
        for gen, grp in te_df.groupby(schema.GENERATOR):
            per_source[str(gen)] = {
                "n": int(len(grp)),
                "true_label": str(grp[schema.LABEL].iloc[0]),
                "predicted_fake_rate": float(grp["_pred_fake"].mean()),
            }

    result = {
        "n": int(len(y)), "n_real": n_real, "n_fake": n_fake,
        "test_n": int(len(yte)),
        "balanced_accuracy": bal_acc,
        "auroc": auroc,
        "chance_balanced_accuracy": 0.5,
        "feature_importances": importances,
        "per_source_predicted_fake_rate": per_source,
        "interpretation": (
            "High balanced accuracy / AUROC = real vs fake is separable from METADATA ALONE "
            "(format/resolution confound is real and exploitable). ~0.5 = no metadata leak "
            "(e.g. after normalization to a common size + format)."),
    }
    with open(os.path.join(args.out_dir, "metadata_confound.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    logger.info("Metadata-only real/fake: balanced_acc=%.3f AUROC=%.3f (chance=0.5)",
                bal_acc, auroc)
    logger.info("Top metadata predictors: %s", importances)
    logger.info("Wrote metadata_confound.json to %s", args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Metadata-only real/fake separability probe.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--metadata", required=True,
                        help="master_metadata.csv (raw) or a variant index CSV (normalized).")
    parser.add_argument("--out_dir", required=True)
    main(parser.parse_args())
