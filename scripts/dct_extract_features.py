#!/usr/bin/env python3
"""
DCT feature extraction (Frank2020, "Leveraging Frequency Analysis for Deep Fake Image
Recognition").

For each image: convert to grayscale luminance, resize to dct_size x dct_size, take the
2D DCT-II (orthonormal), and log-scale the magnitude. GAN upsampling leaves a regular
spectral grid that is visible in this representation. Output is a flat feature vector per
image, ready for the linear SVM in dct_svm.py.

Consumes an index CSV (master_metadata.csv or a variant index) so labels/generators stay
aligned with the rows.

Usage:
  /usr/bin/python3.9 scripts/dct_extract_features.py \
      --index results/index_scaled.csv --out results/dct_features_scaled.npz --dct_size 128
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import io_utils, schema, image_ops  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def dct2_logmag(gray, eps=1e-8):
    """2D orthonormal DCT-II magnitude, log-scaled. `gray` is a float32 HxW array."""
    from scipy.fft import dctn
    coeffs = dctn(gray, type=2, norm="ortho")
    return np.log(np.abs(coeffs) + eps).astype(np.float32)


def main(args):
    logger = io_utils.setup_logging("dct_extract_features")
    from PIL import Image

    df = pd.read_csv(args.index)
    feats, paths, labels, generators, datasets = [], [], [], [], []

    augment = None
    if args.jpeg_aug:
        augment = image_ops.make_jpeg_augmenter((args.jpeg_qmin, args.jpeg_qmax), args.seed)
        logger.info("JPEG augmentation ON (q %d-%d, seed %d) - confound control",
                    args.jpeg_qmin, args.jpeg_qmax, args.seed)

    for _, row in df.iterrows():
        path = row[schema.PATH]
        try:
            img_rgb = Image.open(path).convert("RGB")
            if augment is not None:
                img_rgb = augment(img_rgb, path)
            img = img_rgb.convert("L").resize(
                (args.dct_size, args.dct_size), Image.BICUBIC)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping %s (%s)", path, exc)
            continue
        gray = np.asarray(img, dtype=np.float32) / 255.0
        feats.append(dct2_logmag(gray).ravel())
        paths.append(path)
        labels.append(row[schema.LABEL])
        generators.append(row[schema.GENERATOR])
        datasets.append(row[schema.DATASET])
        if len(paths) % 200 == 0:
            logger.info("Extracted %d", len(paths))

    if not feats:
        raise SystemExit("No features extracted; check --index paths.")

    X = np.stack(feats).astype(np.float32)
    io_utils.ensure_dir(os.path.dirname(os.path.abspath(args.out)))
    np.savez_compressed(
        args.out,
        X=X,
        paths=np.array(paths),
        label=np.array(labels),
        generator=np.array(generators),
        dataset=np.array(datasets),
        dct_size=np.array([args.dct_size]),
        jpeg_aug=np.array([bool(args.jpeg_aug)]),
    )
    logger.info("Wrote %s with X shape %s", args.out, X.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract log-DCT features (Frank2020).")
    parser.add_argument("--index", required=True, help="Index CSV with image_path,label,...")
    parser.add_argument("--out", required=True, help="Output .npz path")
    parser.add_argument("--dct_size", type=int, default=128,
                        help="DCT grid size (image resized to this before DCT)")
    parser.add_argument("--jpeg_aug", action="store_true",
                        help="Apply random JPEG compression per image (format/compression "
                             "confound control). Use for the controlled run; omit for raw.")
    parser.add_argument("--jpeg_qmin", type=int, default=30)
    parser.add_argument("--jpeg_qmax", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
