#!/usr/bin/env python3
"""
Inference-only for a fine-tuned DE-FAKE attribution head (defake_head.pt from
finetune_defake_head.py) over an arbitrary index, emitting a per-image CSV
(full_path,true_generator,pred_generator,confidence,entropy).

Purpose: attribution ROBUSTNESS. finetune_defake_head.py trains + evaluates in one call and has
no predict-on-external-index path, so this loads the saved head and scores a (possibly perturbed)
index. Feed the resulting CSV to robustness_perturb.py --mode score --pred_col pred_generator to
measure how perturbations change WHICH generator is predicted (separate from detection robustness).

Features mirror training: 1024-dim CLIP image+text when --captions_csv is given (same as the
fine-tune). JPEG augmentation is OFF here on purpose - for perturbed inputs the perturbation is
already baked into the image; we must not add a second random re-compression on top.

Run with the DE-FAKE interpreter (venv_sd15: CLIP + torch):
  $WTP_PY_DEFAKE scripts/predict_defake_head.py --config configs/config.yaml \
      --head results/finetune_aspect_jpegaug/defake_head.pt \
      --index results/robust/index_jpeg30.csv \
      --captions_csv /pitsec_sose26_topic8/dataset/defake_predictions_all.csv \
      --out results/robust/attr_jpeg30.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import io_utils, metrics, features_cache, defake_head, schema  # noqa: E402


def _load_head(head_path, device):
    """Rebuild _MLPHead + classes from the saved checkpoint. in_dim/num_classes are inferred
    from the Sequential Linear weights ('0.weight' input, '3.weight' output), same as the
    GAN-fp inference helper so the two stay consistent."""
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
    logger = io_utils.setup_logging("predict_defake_head")
    config = io_utils.load_config(args.config)
    seed = int(config.get("seed", 42))

    head, classes = _load_head(args.head, args.device)
    logger.info("Loaded head: %d classes %s", len(classes), classes)

    # jpeg_aug=False: never re-augment at inference (esp. for already-perturbed robustness inputs).
    X, generator, label, paths = features_cache.build_features(
        args.index, args.features_cache, device=args.device, force=args.recompute_features,
        captions_csv=args.captions_csv, jpeg_aug=False, seed=seed)
    logger.info("Features: %s over %d images", X.shape, len(X))

    proba = head.predict_proba(X)
    pred = proba.argmax(axis=1)
    y_pred_names = [classes[i] for i in pred]
    ent = metrics.predictive_entropy(proba)

    import pandas as pd
    io_utils.ensure_dir(os.path.dirname(os.path.abspath(args.out)))
    pd.DataFrame({
        schema.PATH: paths,
        "true_generator": generator,
        "pred_generator": y_pred_names,
        "confidence": proba.max(axis=1),
        "entropy": ent,
    }).to_csv(args.out, index=False)
    logger.info("Wrote per-image predictions to %s", args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuned DE-FAKE head inference -> per-image CSV.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--head", required=True, help="Path to defake_head.pt")
    parser.add_argument("--index", required=True, help="Index CSV to score")
    parser.add_argument("--captions_csv", default=None,
                        help="Predictions CSV (full_path,blip_caption) for 1024-dim image+text "
                             "features (match the fine-tune).")
    parser.add_argument("--features_cache", default=None, help="Optional CLIP feature .npz cache")
    parser.add_argument("--recompute_features", action="store_true")
    parser.add_argument("--out", required=True, help="Output per-image CSV path")
    parser.add_argument("--device", default="cuda")
    main(parser.parse_args())
