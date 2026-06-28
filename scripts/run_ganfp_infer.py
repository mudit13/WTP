#!/usr/bin/env python3
"""
Run a trained GAN-fp head (ganfp_head.pt from train_ganfp.py) over a set of images and emit a
per-image CSV consumable by the existing eval_defake_attribution.py
(`--pred_col pred_generator`). This lets GAN-fp slot into the standard attribution-eval
pipeline (in_set / out_of_set slices); diffusion sources are expected to land low in the
out_of_set slice - the documented category mismatch.

Two data modes (same as train_ganfp.py):
  --sample_dir <dir>   local prototype folders
  --index <csv>        master_metadata.csv (full run)

Local:
  python scripts/run_ganfp_infer.py --config configs/config.yaml \
      --head results/ganfp_local/ganfp_head.pt --sample_dir ganfp_sample \
      --out results/ganfp_local/ganfp_infer_per_image.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import io_utils, metrics, defake_head, ganfp, schema  # noqa: E402

import numpy as np  # noqa: E402


def _load_head(head_path, device):
    """Rebuild _MLPHead and load the saved state_dict + classes. infers in_dim/num_classes
    from the saved Linear weights (Sequential keys '0.weight' / '3.weight')."""
    import torch
    ckpt = torch.load(head_path, map_location=device, weights_only=False)
    classes = list(ckpt["classes"])
    sd = ckpt["state_dict"]
    in_dim = int(sd["0.weight"].shape[1])
    num_classes = int(sd["3.weight"].shape[0])
    head = defake_head._MLPHead(in_dim=in_dim, num_classes=num_classes, device=device)
    head.model.load_state_dict(sd)
    return head, classes


def main(args):
    logger = io_utils.setup_logging("run_ganfp_infer")
    config = io_utils.load_config(args.config)
    gcfg = config.get("ganfp", {}) or {}
    common_size = int(gcfg.get("common_size", config.get("common_size", 256)))
    feat_size = int(gcfg.get("feat_size", 32))
    mode = str(gcfg.get("mode", "both"))
    real_set = set((config.get("attribution", {}) or {}).get("real_generators", []))

    head, classes = _load_head(args.head, args.device)
    logger.info("Loaded head: %d classes %s (in_dim resolved from weights)", len(classes), classes)

    if args.sample_dir:
        paths, generators = ganfp.scan_sample_dir(args.sample_dir)
        labels = [schema.REAL if g in real_set else schema.FAKE for g in generators]
        X, generator, label, path_arr = ganfp.features_from_samples(
            paths, generators, labels, common_size, feat_size, mode)
    elif args.index:
        X, generator, label, path_arr = ganfp.build_features(
            args.index, None, common_size, feat_size, mode)
    else:
        raise SystemExit("Provide --sample_dir or --index.")
    logger.info("Inference on %d images", len(X))

    proba = head.predict_proba(X)
    pred = proba.argmax(axis=1)
    y_pred_names = [classes[i] for i in pred]
    ent = metrics.predictive_entropy(proba)

    import pandas as pd  # noqa: E402
    pd.DataFrame({
        schema.PATH: path_arr,
        "true_generator": generator,
        "pred_generator": y_pred_names,
        "confidence": proba.max(axis=1),
        "entropy": ent,
    }).to_csv(args.out, index=False)
    logger.info("Wrote per-image predictions to %s", args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAN-fp inference -> per-image CSV.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--head", required=True, help="Path to ganfp_head.pt")
    parser.add_argument("--sample_dir", default=None)
    parser.add_argument("--index", default=None)
    parser.add_argument("--out", required=True, help="Output per-image CSV path")
    parser.add_argument("--device", default="cpu")
    main(parser.parse_args())
