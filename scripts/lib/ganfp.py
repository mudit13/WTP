"""
GAN Fingerprints (Yu2019) reproduction: residual/spectrum fingerprint features.

DE-FAKE (CLIP/semantic) cannot attribute GAN images - it misses model-specific generation
traces. GAN fingerprints are those traces. Per image we compute, on the luminance channel:
  - a noise residual (image minus a Gaussian-blurred estimate -> high-frequency,
    model-specific structure), and/or
  - the FFT log-magnitude spectrum (regular spectral grid left by GAN upsampling),
each downsampled to a fixed grid and L2-normalized. A small learned classifier
(defake_head._MLPHead) then attributes real + each GAN source. Diffusion sources are
expected to mismatch (out-of-set) - documented behavior, not a failure.

No pretrained GAN-fp weights exist (models/ holds DE-FAKE + generators only) and the legacy
/workspace/GANFingerprints repo is Chainer/cupy (dead), so this reproduces the method in
PyTorch over our generators (GOLD_ALIGNMENT.md GAN-Fingerprints note; PROJECT_LOG section 5).

This module is numpy/scipy/Pillow only - torch is NOT imported here (the classifier lives in
defake_head). Safe to import under any interpreter, including CI (no torch). ASCII; Python 3.9.
"""
import hashlib
import json
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import schema

IMG_EXTS = (".png", ".jpg", ".jpeg")


# --- feature dimensionality --------------------------------------------------
def _feature_dim(common_size: int, feat_size: int, mode: str) -> int:
    """Length of one fingerprint vector: (#blocks) * feat_size^2. (common_size is the
    pre-downsample image size and does not itself set the vector length.)"""
    n_blocks = 2 if mode == "both" else 1
    return n_blocks * int(feat_size) * int(feat_size)


# --- per-image fingerprint ---------------------------------------------------
def _luminance(img_rgb, common_size: int) -> np.ndarray:
    """Scale to common_size x common_size and return the luminance as float32 in [0, 1]."""
    from . import image_ops
    small = image_ops.scale_to(img_rgb, common_size)
    arr = np.asarray(small, dtype=np.float32) / 255.0
    if arr.ndim == 3:  # RGB -> Rec.601 luminance
        arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return arr.astype(np.float32)


def _resize_block(block: np.ndarray, feat_size: int) -> np.ndarray:
    """Downsample a 2D float block to feat_size x feat_size and flatten (BILINEAR)."""
    from PIL import Image
    pil = Image.fromarray(block.astype(np.float32))  # mode "F" for a 2D float array
    pil = pil.resize((int(feat_size), int(feat_size)), Image.BILINEAR)
    return np.asarray(pil, dtype=np.float32).ravel()


def _residual_block(gray: np.ndarray, feat_size: int) -> np.ndarray:
    """High-frequency residual (image - Gaussian blur), downsampled. Captures local
    model-specific structure that a global spectrum average would smear out."""
    from scipy.ndimage import gaussian_filter
    blurred = gaussian_filter(gray, sigma=1.0)
    return _resize_block(gray - blurred, feat_size)


def _spectrum_block(gray: np.ndarray, feat_size: int, eps: float = 1e-8) -> np.ndarray:
    """FFT log-magnitude spectrum (DC-centered), downsampled. GAN upsampling leaves a
    regular spectral grid visible here (cf. Frank2020 / dct_extract_features.py)."""
    mag = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    return _resize_block(np.log(mag + eps), feat_size)


def _fingerprint_vector(img_rgb, common_size: int, feat_size: int,
                        mode: str) -> np.ndarray:
    """Build one L2-normalized fingerprint vector from an RGB PIL image."""
    gray = _luminance(img_rgb, common_size)
    blocks = []
    if mode in ("residual", "both"):
        blocks.append(_residual_block(gray, feat_size))
    if mode in ("spectrum", "both"):
        blocks.append(_spectrum_block(gray, feat_size))
    vec = np.concatenate(blocks).astype(np.float32)
    norm = float(np.linalg.norm(vec)) + 1e-8
    return (vec / norm).astype(np.float32)


def extract_fingerprints(image_paths: Sequence[str],
                         common_size: int = 256,
                         feat_size: int = 32,
                         mode: str = "both",
                         augment=None,
                         batch_size: int = 64
                         ) -> Tuple[np.ndarray, List[str]]:
    """Return (features [N, D] float32, kept_paths) for the given paths.

    `augment`, if given, is a callable(img, path) -> img applied to each loaded RGB image
    before fingerprinting (training-time JPEG augmentation). Unreadable images are skipped
    and excluded from kept_paths so the caller keeps labels aligned. `batch_size` is accepted
    for signature parity with clip_features.extract_features (no GPU batching here).
    """
    from . import image_ops

    feats: List[np.ndarray] = []
    kept: List[str] = []
    for path in image_paths:
        try:
            img = image_ops.load_rgb(path)
        except Exception:  # noqa: BLE001 - skip unreadable files, keep the run going
            continue
        if augment is not None:
            img = augment(img, path)
        feats.append(_fingerprint_vector(img, common_size, feat_size, mode))
        kept.append(path)

    dim = _feature_dim(common_size, feat_size, mode)
    if not feats:
        return np.zeros((0, dim), dtype=np.float32), kept
    return np.stack(feats).astype(np.float32), kept


# --- sample-dir scanning (local prototype) -----------------------------------
def scan_sample_dir(sample_dir: str) -> Tuple[List[str], List[str]]:
    """Return (paths, generators) for <sample_dir>/<generator>/* image files. The folder
    name IS the generator string (must match schema.GENERATOR values, e.g. 'StyleGAN3-FFHQ')."""
    paths: List[str] = []
    generators: List[str] = []
    for gen in sorted(os.listdir(sample_dir)):
        gdir = os.path.join(sample_dir, gen)
        if not os.path.isdir(gdir):
            continue
        for fn in sorted(os.listdir(gdir)):
            if fn.lower().endswith(IMG_EXTS):
                paths.append(os.path.join(gdir, fn))
                generators.append(gen)
    return paths, generators


# --- cache signature (mirrors features_cache._signature) ---------------------
def _file_hash(path: Optional[str]) -> str:
    if not path or not os.path.exists(path):
        return "<none>"
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _signature(index_csv, common_size, feat_size, mode, jpeg_aug,
               jpeg_quality_range, seed) -> str:
    """Fingerprint of everything that affects the features. A cache whose signature differs
    is NOT reused."""
    meta = {
        "index": _file_hash(index_csv),
        "common_size": int(common_size),
        "feat_size": int(feat_size),
        "mode": str(mode),
        "jpeg_aug": bool(jpeg_aug),
        "qr": [int(jpeg_quality_range[0]), int(jpeg_quality_range[1])],
        "seed": int(seed),
    }
    return hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()


def _realignment_order(kept: Sequence[str]) -> dict:
    """path -> row index in the extracted feature matrix (kept order)."""
    return {p: i for i, p in enumerate(kept)}


def features_from_samples(paths: Sequence[str], generators: Sequence[str],
                          labels: Sequence[str], common_size: int = 256,
                          feat_size: int = 32, mode: str = "both",
                          augment=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract fingerprints for parallel (paths, generators, labels) lists and realign to
    the readable images. Used by the --sample_dir local prototype path. No caching.

    Returns (X, generator, label, paths) as aligned arrays.
    """
    X, kept = extract_fingerprints(paths, common_size, feat_size, mode, augment)
    order = _realignment_order(kept)
    g_out, l_out, p_out, idx = [], [], [], []
    for p, g, l in zip(paths, generators, labels):
        if p in order:
            idx.append(order[p])
            g_out.append(g)
            l_out.append(l)
            p_out.append(p)
    if idx:
        Xa = X[idx]
    else:
        Xa = np.zeros((0, _feature_dim(common_size, feat_size, mode)), dtype=np.float32)
    return (Xa.astype(np.float32), np.array(g_out, dtype=object).astype(str),
            np.array(l_out, dtype=object).astype(str), np.array(p_out, dtype=object).astype(str))


def build_features(index_csv: str, cache_path: Optional[str], common_size: int = 256,
                   feat_size: int = 32, mode: str = "both", jpeg_aug: bool = False,
                   jpeg_quality_range=(30, 100), seed: int = 42,
                   force: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, generator, label, paths) for all rows in index_csv, with .npz caching.

    Reads the index's schema.PATH/GENERATOR/LABEL. When jpeg_aug is True, each image is pushed
    through a per-path-deterministic random JPEG quality before fingerprinting. A cache is
    reused ONLY when its full signature (index content, common_size, feat_size, mode, jpeg
    params, seed) matches; otherwise it is recomputed.
    """
    import pandas as pd

    sig = _signature(index_csv, common_size, feat_size, mode, jpeg_aug,
                     jpeg_quality_range, seed)
    if cache_path and os.path.exists(cache_path) and not force:
        data = np.load(cache_path, allow_pickle=True)
        cached_sig = str(np.asarray(data["signature"]).ravel()[0]) if "signature" in data else ""
        if cached_sig == sig:
            return (data["X"].astype(np.float32), data["generator"].astype(str),
                    data["label"].astype(str), data["paths"].astype(str))

    df = pd.read_csv(index_csv)
    df[schema.PATH] = df[schema.PATH].astype(str)
    paths = df[schema.PATH].tolist()

    augment = None
    if jpeg_aug:
        from . import image_ops
        augment = image_ops.make_jpeg_augmenter(jpeg_quality_range, seed)

    X, kept = extract_fingerprints(paths, common_size, feat_size, mode, augment)

    order = _realignment_order(kept)
    df = df[df[schema.PATH].isin(set(kept))].copy()
    df = df.iloc[df[schema.PATH].map(order).argsort()].reset_index(drop=True)

    generator = df[schema.GENERATOR].astype(str).values
    label = df[schema.LABEL].astype(str).values
    paths_arr = df[schema.PATH].astype(str).values

    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        np.savez_compressed(cache_path, X=X, generator=generator, label=label,
                            paths=paths_arr, jpeg_aug=np.array([bool(jpeg_aug)]),
                            signature=np.array([sig]))
    return X, generator, label, paths_arr


# --- PCA / standardization pipeline (train-only fit, NO leakage) -------------
class FingerprintStandardizer:
    """StandardScaler + PCA, fit on TRAIN indices ONLY.

    Wraps sklearn.preprocessing.StandardScaler and sklearn.decomposition.PCA so the
    GAN-fp feature path (residual+spectrum vectors from extract_fingerprints) gets a
    lower-dimensional, decorrelated input for defake_head._MLPHead while keeping the
    leakage guard explicit: fit() sees only X_train, transform() is applied to val/test.

    Contract (asserted by tests/test_ganfp.py):
      - fit(X_train) learns mean_/scale_ and the PCA rotation from TRAIN ONLY;
      - transform(X_train) is approximately zero-mean, unit-variance;
      - transform(X_val) is NOT standardized (val keeps train's mean/scale) -> the
        dedicated leakage-guard test asserts this.

    mean_/scale_/PCA components are stored as float32 numpy arrays so the whole
    pipeline can be serialized into the metrics JSON / a .npz sidecar alongside the head.

    Optional DCT fusion: when dct_features is provided to fit()/transform(), a SECOND
    scaler+PCA is fit independently on the DCT channel and concatenated after the
    residual/spectrum PCA vector (additive, not replacing). This isolates whether the
    8x8 block-DCT adds discriminative signal beyond the residual+spectrum fingerprint
    (controlled by config.ganfp.pca.dct_fuse).

    sklearn is imported INSIDE the methods so importing ganfp never pulls sklearn at
    module load (CI stays sklearn-import-clean until a FingerprintStandardizer is
    actually constructed and fitted). ASCII; Python 3.9.
    """

    def __init__(self, pca_components: int = 64,
                 dct_components: Optional[int] = None):
        self.pca_components = int(pca_components)
        self.dct_components = (int(dct_components)
                               if dct_components is not None else None)
        # learned state (set in fit)
        self.scaler_ = None
        self.pca_ = None
        self.dct_scaler_ = None
        self.dct_pca_ = None
        self.in_dim_ = None

    # -- internal: clamp n_components to a rank-safe value ------------------
    @staticmethod
    def _clamped_components(want: int, n_samples: int) -> int:
        """PCA n_components cannot exceed n_samples-1 (and n_features)."""
        return max(1, min(int(want), int(n_samples) - 1))

    def fit(self, X_train: np.ndarray,
            dct_features: Optional[np.ndarray] = None) -> "FingerprintStandardizer":
        """Fit scaler+PCA on TRAIN ONLY. Returns self."""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        X_train = np.asarray(X_train, dtype=np.float32)
        if X_train.ndim != 2:
            raise ValueError("X_train must be 2D [n_samples, n_features]")
        n = X_train.shape[0]
        if n < 2:
            raise ValueError("Need at least 2 train samples to fit scaler+PCA.")

        comp = self._clamped_components(self.pca_components, n)
        self.scaler_ = StandardScaler().fit(X_train)
        self.pca_ = PCA(n_components=comp, random_state=0).fit(
            self.scaler_.transform(X_train))

        in_dim = comp
        if dct_features is not None and self.dct_components is not None:
            D = np.asarray(dct_features, dtype=np.float32)
            if D.shape[0] != n:
                raise ValueError("dct_features row count must match X_train.")
            dcomp = self._clamped_components(self.dct_components, n)
            self.dct_scaler_ = StandardScaler().fit(D)
            self.dct_pca_ = PCA(n_components=dcomp, random_state=0).fit(
                self.dct_scaler_.transform(D))
            in_dim += dcomp
        self.in_dim_ = int(in_dim)
        return self

    def fit_transform(self, X_train: np.ndarray,
                      dct_features: Optional[np.ndarray] = None) -> np.ndarray:
        """Convenience: fit on X_train then transform it (train-only)."""
        return self.fit(X_train, dct_features).transform(X_train, dct_features)

    def transform(self, X: np.ndarray,
                  dct_features: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply the train-fit scaler+PCA. dct_features, if given, is appended as a
        second channel. NO refit happens here (leakage guard)."""
        if self.scaler_ is None or self.pca_ is None:
            raise RuntimeError("FingerprintStandardizer.transform called before fit.")
        X = np.asarray(X, dtype=np.float32)
        out = self.pca_.transform(self.scaler_.transform(X)).astype(np.float32)
        if self.dct_pca_ is not None:
            if dct_features is None:
                raise ValueError(
                    "Pipeline was fit with DCT fusion; dct_features required at transform.")
            D = np.asarray(dct_features, dtype=np.float32)
            dout = self.dct_pca_.transform(self.dct_scaler_.transform(D)).astype(
                np.float32)
            out = np.concatenate([out, dout], axis=1)
        return out

    def to_dict(self) -> dict:
        """Serialize the learned pipeline state (float32 arrays) for the metrics JSON /
        a .npz sidecar so the exact scaler/PCA applied at train time can be audited
        alongside the saved head."""
        return {
            "pca_components": self.pca_components,
            "dct_components": self.dct_components,
            "in_dim": self.in_dim_,
            "scaler_mean": (None if self.scaler_ is None
                            else np.asarray(self.scaler_.mean_, dtype=np.float32).tolist()),
            "scaler_scale": (None if self.scaler_ is None
                             else np.asarray(self.scaler_.scale_, dtype=np.float32).tolist()),
            "pca_components_": (None if self.pca_ is None
                                else np.asarray(self.pca_.components_, dtype=np.float32).tolist()),
            "dct_scaler_mean": (None if self.dct_scaler_ is None
                                else np.asarray(self.dct_scaler_.mean_, dtype=np.float32).tolist()),
            "dct_scaler_scale": (None if self.dct_scaler_ is None
                                 else np.asarray(self.dct_scaler_.scale_, dtype=np.float32).tolist()),
            "dct_pca_components_": (None if self.dct_pca_ is None
                                    else np.asarray(self.dct_pca_.components_, dtype=np.float32).tolist()),
        }


def _block_dct_features(img_rgb, common_size: int = 256,
                        block: int = 8) -> np.ndarray:
    """8x8 block-DCT of the luminance, mean-pooled per block -> 64-dim vector.

    Uses scipy.fftpack.dct (lazy import) type-II DCT per 8x8 block, takes the |.| of the
    first 8 AC coefficients per block averaged, flattened to block*block = 64 dims. This
    is the SECOND feature channel for optional DCT fusion (config.ganfp.pca.dct_fuse)."""
    from scipy.fftpack import dct as _dct

    gray = _luminance(img_rgb, common_size)
    h, w = gray.shape
    bh, bw = h // block, w // block
    if bh == 0 or bw == 0:
        return np.zeros(block * block, dtype=np.float32)
    feat = np.zeros((block, block), dtype=np.float32)
    for i in range(block):
        for j in range(block):
            tile = gray[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw]
            c = _dct(_dct(tile, axis=0, type=2, norm="ortho"),
                     axis=1, type=2, norm="ortho")
            # mean |AC| energy of the block (skip the DC term at [0,0])
            ac = np.abs(c).sum() - abs(c[0, 0])
            feat[i, j] = float(ac) / float(c.size)
    return feat.ravel().astype(np.float32)


def extract_dct_features(image_paths: Sequence[str], common_size: int = 256,
                         block: int = 8, augment=None
                         ) -> Tuple[np.ndarray, List[str]]:
    """Return (DCT features [N, block*block], kept_paths) parallel to extract_fingerprints.

    `augment`, if given, is a callable(img, path) -> img applied before the DCT (training-
    time JPEG augmentation, same augmenter object the residual/spectrum path uses)."""
    from . import image_ops

    feats: List[np.ndarray] = []
    kept: List[str] = []
    for path in image_paths:
        try:
            img = image_ops.load_rgb(path)
        except Exception:  # noqa: BLE001
            continue
        if augment is not None:
            img = augment(img, path)
        feats.append(_block_dct_features(img, common_size, block))
        kept.append(path)
    if not feats:
        return np.zeros((0, block * block), dtype=np.float32), kept
    return np.stack(feats).astype(np.float32), kept


def build_pca_pipeline(X_train: np.ndarray, pca_components: int = 64,
                       dct_fuse: bool = False,
                       dct_components: int = 32,
                       dct_train: Optional[np.ndarray] = None
                       ) -> Tuple["FingerprintStandardizer", int]:
    """Fit a FingerprintStandardizer on X_train and return (standardizer, in_dim).

    When dct_fuse is True, dct_train (the per-image DCT feature matrix for the TRAIN
    rows, aligned to X_train) is required; the pipeline concatenates a PCA'd DCT channel
    after the residual/spectrum PCA channel, so in_dim = pca_components[+dct_components].

    Contract: X_train (and dct_train) MUST be the TRAIN-split rows only. The helper
    exposes fit/transform separately (via FingerprintStandardizer) precisely so a caller
    cannot accidentally fit_transform the whole matrix (leakage guard). sklearn is
    imported inside FingerprintStandardizer.fit, so importing ganfp never pulls sklearn
    at module load.
    """
    dcomp = dct_components if dct_fuse else None
    std = FingerprintStandardizer(pca_components=pca_components, dct_components=dcomp)
    darg = dct_train if dct_fuse else None
    std.fit(X_train, darg)
    return std, int(std.in_dim_)
