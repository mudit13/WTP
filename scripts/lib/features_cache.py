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
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from . import schema


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
    deterministic) before CLIP - the format/compression confound control. A cache built with a
    different jpeg_aug setting is ignored (not silently reused).
    """
    if cache_path and os.path.exists(cache_path) and not force:
        data = np.load(cache_path, allow_pickle=True)
        cached_aug = bool(np.asarray(data["jpeg_aug"]).ravel()[0]) if "jpeg_aug" in data else False
        if cached_aug == bool(jpeg_aug):
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
                            jpeg_aug=np.array([bool(jpeg_aug)]))
    return X, generator, label, paths_arr
