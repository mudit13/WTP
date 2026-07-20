"""
Build and cache CLIP feature matrices from an index CSV (real schema).

Extracting CLIP features is the expensive step; caching to .npz lets the fine-tune,
leave-one-generator-out, and out-of-set scripts reuse the same embeddings. The class label
for attribution is the `generator` column (human names; real datasets carry their real
source name, e.g. London-DB / FFHQ).

If a captions CSV (full_path, blip_caption) is provided, features are the faithful DE-FAKE
1024-dim image+text embedding; otherwise the 512-dim image embedding.

DE-FAKE interpreter only (venv_sd15 on the server). ASCII-only; Python 3.9.
"""
import hashlib
import json
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from . import schema


def training_aug_cache_path(cache_path: Optional[str]) -> Optional[str]:
    """Companion cache for training-only JPEG features; clean eval keeps `cache_path`."""
    if not cache_path:
        return None
    root, ext = os.path.splitext(cache_path)
    return root + "_train_jpegaug" + (ext or ".npz")


def _file_hash(path: Optional[str]) -> str:
    """SHA-256 of a file's bytes (or a sentinel if missing). Used so the feature cache
    invalidates when the index/captions content changes, not just their path."""
    if not path or not os.path.exists(path):
        return "<none>"
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _signature(index_csv, captions_csv, model_name, jpeg_aug, jpeg_quality_range, seed) -> str:
    """Fingerprint of everything that affects the features. A cache whose signature differs
    (different index content, captions, CLIP model, JPEG params, or seed) is NOT reused."""
    meta = {
        "index": _file_hash(index_csv),
        "captions": _file_hash(captions_csv),
        "model": str(model_name),
        "jpeg_aug": bool(jpeg_aug),
        "qr": [int(jpeg_quality_range[0]), int(jpeg_quality_range[1])],
        "seed": int(seed),
    }
    return hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()


def _load_captions(captions_csv: str) -> dict:
    cap = pd.read_csv(captions_csv)
    col = schema.BLIP_CAPTION if schema.BLIP_CAPTION in cap.columns else None
    if col is None or schema.PATH not in cap.columns:
        return {}
    cap[col] = cap[col].fillna("").astype(str)
    return dict(zip(cap[schema.PATH], cap[col]))


def build_features(index_csv: str,
                   cache_path: Optional[str],
                   model_name: str = "ViT-B/32",
                   device: str = "cuda",
                   batch_size: int = 64,
                   force: bool = False,
                   captions_csv: Optional[str] = None,
                   jpeg_aug: bool = False,
                   jpeg_quality_range=(30, 100),
                   seed: int = 42
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, generator, label, paths) for all rows in index_csv.

    Uses cache_path when present (and not force). Generator/label come from the index's
    schema columns. When captions_csv is given, X is 1024-dim image+text.

    When jpeg_aug is True, each image is pushed through a random JPEG quality (per-path
    deterministic) before CLIP - the format/compression confound control. A cache is reused
    ONLY when its full signature (index + captions content, model, jpeg params, seed) matches;
    otherwise it is recomputed, so stale features can never silently back a new experiment.
    """
    sig = _signature(index_csv, captions_csv, model_name, jpeg_aug, jpeg_quality_range, seed)
    if cache_path and os.path.exists(cache_path) and not force:
        data = np.load(cache_path, allow_pickle=True)
        cached_sig = str(np.asarray(data["signature"]).ravel()[0]) if "signature" in data else ""
        if cached_sig == sig:
            return (data["X"].astype(np.float32), data["generator"].astype(str),
                    data["label"].astype(str), data["paths"].astype(str))

    df = pd.read_csv(index_csv)
    paths = df[schema.PATH].tolist()

    captions = None
    if captions_csv:
        cap_map = _load_captions(captions_csv)
        captions = [cap_map.get(p, "") for p in paths]

    augment = None
    if jpeg_aug:
        from . import image_ops
        augment = image_ops.make_jpeg_augmenter(jpeg_quality_range, seed)

    from . import clip_features
    model, preprocess, dev = clip_features.get_clip(model_name, device)
    X, kept = clip_features.extract_features(paths, model, preprocess, dev,
                                             batch_size, captions=captions, augment=augment)

    order = {p: i for i, p in enumerate(kept)}
    df = df[df[schema.PATH].isin(set(kept))].copy()
    df = df.iloc[df[schema.PATH].map(order).argsort()].reset_index(drop=True)

    generator = df[schema.GENERATOR].astype(str).values
    label = df[schema.LABEL].astype(str).values
    paths_arr = df[schema.PATH].astype(str).values

    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        np.savez_compressed(cache_path, X=X, generator=generator,
                            label=label, paths=paths_arr,
                            jpeg_aug=np.array([bool(jpeg_aug)]),
                            signature=np.array([sig]))
    return X, generator, label, paths_arr
