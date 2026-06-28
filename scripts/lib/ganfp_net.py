"""
GAN Fingerprints CNN path (Yu2019-inspired): a compact VGG-style CNN over single-channel
luminance, with a FIXED (non-trainable) SRM high-pass front-end.

Attribution note (honest scope): Yu et al. 2019 ("Attributing Fake Images to GANs") learn the
fingerprint with a plain CNN on RGB images. We keep that core idea -- the conv filters BECOME
the learned per-generator fingerprints -- but prepend a fixed SRM steganalysis front-end
(Fridrich & Kodovsky 2012) to bias the model toward forensic high-frequency traces. So this is
"Yu2019-inspired with an SRM front-end", NOT a byte-faithful reimplementation of Yu2019.

Path B of the GAN-fp reproduction. Path A (residual/spectrum features + defake_head._MLPHead)
lives in ganfp.py; this module is the end-to-end CNN alternative. Both share the SAME seeded
stratified split and the SAME per-image JPEG augmentation (image_ops.make_jpeg_augmenter) so
the head-to-head benchmark (scripts/benchmark_attribution.py) is apples-to-apples.

Architecture (trainable params scale with config.ganfp.cnn.channels; default [32,64,128]
~330K params, trains in minutes on CUDA / modest time on CPU):
  - Frozen SRM high-pass front-end: Conv2d(1,30,5,padding=2,bias=False) loaded with the
    Spatial Rich Model filter bank (Fridrich & Kodovsky 2012, "Rich Models for Steganalysis").
    The canonical SRM = 30 high-pass filters (3x3 SPAM1d3x3, 5x5 SPAM14, ~24 'minmax' 5x5,
    and 'KB' 5x5). Every filter sums to ~0 -> DC-suppressed. requires_grad=False and never
    updated. This emphasizes the forensic high-frequency traces GAN-fingerprint attribution
    relies on and makes the first-layer 'fingerprint' interpretable. NO mean/std z-scoring:
    the SRM high-pass + BatchNorm combo replaces z-scoring (constant image -> ~0 residual).
  - 3 VGG-style conv blocks (Conv-BN-ReLU x2 + MaxPool), channels default [32,64,128].
  - AdaptiveAvgPool2d(1) -> Flatten -> Linear(channels[-1],128)+ReLU+Dropout(0.3) ->
    Linear(128,C).

torch is imported INSIDE the class bodies / forward / __init__ so importing ganfp_net NEVER
pulls torch at module top (CI runs with no torch; the torch tests use pytest.importorskip).
Module top-level = stdlib + numpy only. ASCII-only; Python 3.9.

Interpreter:
  - CPU/local prototype: system Python (torch optional; the CNN path is exercised under the
    DE-FAKE interpreter or wherever torch is available).
  - CNN full run / benchmark full run: the venv with CUDA torch ($WTP_PY_DEFAKE / .venv).
"""
import os
import sys
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import image_ops, schema

IMG_EXTS = (".png", ".jpg", ".jpeg")

# ---------------------------------------------------------------------------
# SRM (Spatial Rich Model) high-pass filter bank.
# Fridrich & Kodovsky 2012, "Rich Models for Steganalysis". The canonical SRM is a set of
# 30 linear high-pass filters (sizes 3x3 and 5x5) used as a fixed first layer in steganalysis.
# Each filter is normalized so its weights sum to ~0 (DC-suppressed): a constant image yields
# a ~zero residual on every filter, which is the defining property of a high-pass / residual
# operator and the property the test_highpass_bank_* tests assert.
#
# Composition of this 30-filter bank (matches the SRM families in the canonical reference):
#   * 1x  3x3  spam1d3x3 (the 1-D SPAM residual in 3x3 form, the 'LoG-like' KB edge kernel)
#   * 4x  3x3  SPAM1d horizontal/vertical, plus second-order (square) 3x3 minmax filters
#   * 1x  3x3  Laplacian-4 center-surround
#   * 1x  5x5  SPAM14 (the 4th-order 2-D SPAM residual, the workhorse of the bank)
#   * 3x  5x5  horizontal/vertical/diagonal 1-D SPAM-derivative variants
#   * 2x  5x5  Laplacian-8 / square-Laplacian (the 'KV'/cubic kernels)
#   * ~18x 5x5 'minmax' third-order residuals (the 3rd/4th-order 2-D SPAM minmax family that
#             dominates the count of the canonical 30-filter bank).
# Where the exact published minmax coefficients are non-unique / sign-ambiguous, a distinct
# DC-suppressed high-pass kernel that is a member of the same SPAM/minmax family is used, so
# the bank is exactly 30 DISTINCT DC-suppressed filters spanning the SRM families (3x3 SPAM,
# 5x5 SPAM, Laplacian, minmax). Honest scope: this is an SRM-FAMILY high-pass bank, not a
# coefficient-for-coefficient copy of the published SRM; the defining DC-suppressed/high-pass
# property holds for every filter (asserted by the tests), which is what the CNN needs.
# ---------------------------------------------------------------------------

def _norm0(k: np.ndarray) -> np.ndarray:
    """Normalize a kernel so its weights sum to EXACTLY 0 (DC-suppressed) while PRESERVING
    its sparsity pattern and high-pass character.

    Two SRM-faithful steps:
      1. Zero-mean by adjusting the CENTER pixel only (the canonical SRM convention): the
         center absorbs the residual so the sum is exactly 0 and the off-center coefficients
         keep their exact published values. This is distinct from subtracting the global mean,
         which would smear a constant offset into every cell (including a zero halo) and make
         otherwise-distinct sparse filters collinear.
      2. Rescale to unit L2 norm so every filter has comparable dynamic range.
    Preserves zeros (sparsity) and the relative weight pattern, so structurally different
    high-pass operators remain pairwise non-collinear.
    """
    k = np.asarray(k, dtype=np.float32).copy()
    s = float(k.sum())
    if abs(s) > 1e-12:
        c = k.shape[0] // 2
        k[c, c] = k[c, c] - s
    nrm = float(np.linalg.norm(k))
    if nrm > 1e-12:
        k = k / nrm
    return k.astype(np.float32)


def _build_srm_bank() -> np.ndarray:
    """Build the (30, 1, 5, 5) SRM high-pass filter bank as float32.

    All kernels are embedded in a 5x5 frame (3x3 kernels sit in the center with a zero halo)
    so the returned ndarray is uniformly shaped for a single Conv2d(1,30,5,padding=2). Every
    kernel sums to ~0 after _norm0. Returns float32 ndarray shape (30,1,5,5).
    """
    filters = []

    def add3(k3):
        """Embed a 3x3 kernel in the center of a 5x5 zero frame, then DC-suppress."""
        f = np.zeros((5, 5), dtype=np.float32)
        f[1:4, 1:4] = np.asarray(k3, dtype=np.float32)
        filters.append(_norm0(f))

    def add5(k5):
        filters.append(_norm0(np.asarray(k5, dtype=np.float32)))

    # ---- 3x3 family -------------------------------------------------------
    # (1) spam1d3x3 -- the canonical 3x3 SPAM second-order residual (center - avg of 4-neigh).
    add3([[0.0, 1.0, 0.0],
          [1.0, -4.0, 1.0],
          [0.0, 1.0, 0.0]])
    # (2) SPAM1d horizontal second-order.
    add3([[0.0, 0.0, 0.0],
          [1.0, -2.0, 1.0],
          [0.0, 0.0, 0.0]])
    # (3) SPAM1d vertical second-order.
    add3([[0.0, 1.0, 0.0],
          [0.0, -2.0, 0.0],
          [0.0, 1.0, 0.0]])
    # (4) SPAM1d diagonal (NE-SW) second-order.
    add3([[1.0, 0.0, 0.0],
          [0.0, -2.0, 0.0],
          [0.0, 0.0, 1.0]])
    # (5) SPAM1d diagonal (NW-SE) second-order.
    add3([[0.0, 0.0, 1.0],
          [0.0, -2.0, 0.0],
          [1.0, 0.0, 0.0]])
    # (6) Laplacian-4 (explicit, distinct from (1)'s center-surround normalization).
    add3([[-1.0, -1.0, -1.0],
          [-1.0, 8.0, -1.0],
          [-1.0, -1.0, -1.0]])
    # (7) 3x3 minmax square (second-order 'square' residual).
    add3([[1.0, -2.0, 1.0],
          [-2.0, 4.0, -2.0],
          [1.0, -2.0, 1.0]])
    # (8) 3x3 asymmetric minmax (third-order 'L'-shaped edge residual; NOT a sign/rotation
    #     of the square kernel above, so it carries distinct directional information).
    add3([[1.0, 1.0, 0.0],
          [1.0, -3.0, -1.0],
          [0.0, -1.0, 0.0]])

    # ---- 5x5 family -------------------------------------------------------
    # (9) SPAM14 -- the canonical 4th-order 2-D SPAM residual (the SRM workhorse).
    spam14 = np.array([
        [-1,  2, -2,  2, -1],
        [ 2, -6,  8, -6,  2],
        [-2,  8, -12, 8, -2],
        [ 2, -6,  8, -6,  2],
        [-1,  2, -2,  2, -1]], dtype=np.float32)
    add5(spam14)
    # (10) Laplacian-of-Gaussian (LoG) 5x5 -- the 'KB' edge kernel.
    log5 = np.array([
        [0,  0, -1,  0,  0],
        [0, -1, -2, -1,  0],
        [-1, -2, 16, -2, -1],
        [0, -1, -2, -1,  0],
        [0,  0, -1,  0,  0]], dtype=np.float32)
    add5(log5)
    # (11) Laplacian-8 5x5 (full 8-neighborhood second-order).
    lap8 = np.array([
        [-1, -1, -1, -1, -1],
        [-1,  1,  1,  1, -1],
        [-1,  1, -8,  1, -1],
        [-1,  1,  1,  1, -1],
        [-1, -1, -1, -1, -1]], dtype=np.float32)
    add5(lap8)
    # (12) SPAM1d horizontal fourth-order (5-tap second-derivative).
    add5(np.array([
        [0, 0, 0, 0, 0],
        [1, -4, 6, -4, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]], dtype=np.float32))
    # (13) SPAM1d vertical fourth-order.
    add5(np.array([
        [0, 1, 0, 0, 0],
        [0, -4, 0, 0, 0],
        [0, 6, 0, 0, 0],
        [0, -4, 0, 0, 0],
        [0, 1, 0, 0, 0]], dtype=np.float32))
    # (14) SPAM1d diagonal fourth-order (NW-SE).
    add5(np.array([
        [1, 0, 0, 0, 0],
        [0, -4, 0, 0, 0],
        [0, 0, 6, 0, 0],
        [0, 0, 0, -4, 0],
        [0, 0, 0, 0, 1]], dtype=np.float32))

    # ---- minmax third/fourth-order family (15..30) ------------------------
    # The minmax residuals are the dominant family in the canonical 30-filter SRM. Each is a
    # distinct DC-suppressed high-pass operator: directional (min/max of SPAM derivatives),
    # square, and 3x3/5x5 cross combinations. These fill out the 30-filter bank to the
    # declared count while every filter remains distinct and DC-suppressed.

    # (15) 5x5 minmax horizontal 3rd-order.
    add5(np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 2, -6, 2, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]], dtype=np.float32))
    # (16) 5x5 minmax vertical 3rd-order.
    add5(np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 2, 0, 0],
        [0, 0, -6, 0, 0],
        [0, 0, 2, 0, 0],
        [0, 0, 1, 0, 0]], dtype=np.float32))
    # (17) 5x5 square (LoG^2 sign-pattern) minmax.
    add5(np.array([
        [1, -2, 0, -2, 1],
        [-2, 4, 0, 4, -2],
        [0, 0, 0, 0, 0],
        [-2, 4, 0, 4, -2],
        [1, -2, 0, -2, 1]], dtype=np.float32))
    # (18) 5x5 diagonal minmax NE-SW.
    add5(np.array([
        [0, 0, 0, 0, 1],
        [0, 0, 0, 2, 0],
        [0, 0, -6, 0, 0],
        [0, 2, 0, 0, 0],
        [1, 0, 0, 0, 0]], dtype=np.float32))
    # (19) 5x5 diagonal minmax NW-SE.
    add5(np.array([
        [1, 0, 0, 0, 0],
        [0, 2, 0, 0, 0],
        [0, 0, -6, 0, 0],
        [0, 0, 0, 2, 0],
        [0, 0, 0, 0, 1]], dtype=np.float32))
    # (20) 5x5 cross (third-order center surround).
    add5(np.array([
        [0, 0, 1, 0, 0],
        [0, 0, -2, 0, 0],
        [1, -2, 2, -2, 1],
        [0, 0, -2, 0, 0],
        [0, 0, 1, 0, 0]], dtype=np.float32))
    # (21) 5x5 SPAM horizontal 3rd-order (left-pointing).
    add5(np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [-1, 3, -3, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]], dtype=np.float32))
    # (22) 5x5 SPAM horizontal 3rd-order (right-pointing).
    add5(np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, -3, 3, -1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]], dtype=np.float32))
    # (23) 5x5 SPAM vertical 3rd-order (up-pointing).
    add5(np.array([
        [0, 0, -1, 0, 0],
        [0, 0, 3, 0, 0],
        [0, 0, -3, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0]], dtype=np.float32))
    # (24) 5x5 SPAM vertical 3rd-order (down-pointing).
    add5(np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, -3, 0, 0],
        [0, 0, 3, 0, 0],
        [0, 0, -1, 0, 0]], dtype=np.float32))
    # (25) 5x5 minmax 'edge' (gradient magnitude proxy, x then y combined).
    add5(np.array([
        [0, 0, 0, 0, 0],
        [1, 1, -2, -1, -1],
        [1, 1, -2, -1, -1],
        [-1, -1, 2, 1, 1],
        [-1, -1, 2, 1, 1]], dtype=np.float32))
    # (26) 5x5 square-Laplacian (cubic residual).
    add5(np.array([
        [0, 0, 1, 0, 0],
        [0, 2, -4, 2, 0],
        [1, -4, 2, -4, 1],
        [0, 2, -4, 2, 0],
        [0, 0, 1, 0, 0]], dtype=np.float32))
    # (27) 5x5 minmax 4th-order horizontal (symmetric).
    add5(np.array([
        [0, 0, 0, 0, 0],
        [1, -3, 4, -3, 1],
        [0, 0, 0, 0, 0],
        [-1, 3, -4, 3, -1],
        [0, 0, 0, 0, 0]], dtype=np.float32))
    # (28) 5x5 one-sided 3rd-order SPAM (odd derivative, off-center; the asymmetric placement
    #      makes it genuinely non-collinear with the symmetric center-surround kernels above).
    add5(np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, -2, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]], dtype=np.float32))
    # (29) 5x5 minmax 'KV'-style cubic residual, placed OFF-CENTER (rows 1-2 only) so it is
    #      NOT a sign/rotation of the centered square Laplacian (filter 7) -- the asymmetric
    #      vertical placement carries distinct row-band information.
    add5(np.array([
        [0, 0, 0, 0, 0],
        [1, -2, 1, 0, 0],
        [-1, 2, -1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]], dtype=np.float32))
    # (30) 5x5 omnidirectional minmax (sum of the two diagonal Laplacians).
    add5(np.array([
        [-1, 0, 0, 0, -1],
        [0, 2, -1, 2, 0],
        [0, -1, 0, -1, 0],
        [0, 2, -1, 2, 0],
        [-1, 0, 0, 0, -1]], dtype=np.float32))

    bank = np.stack(filters, axis=0)[:, None, :, :].astype(np.float32)  # (N,1,5,5)
    assert bank.shape == (30, 1, 5, 5), "SRM bank must be (30,1,5,5), got %s" % (bank.shape,)
    return bank


# (30, 1, 5, 5) SRM filter bank. Built once at import (numpy-only). DC-suppressed per filter.
_SRM_BANK = _build_srm_bank()
# Number of filters in the SRM front-end (exposed for tests + the CNN to size its Conv2d).
SRM_FILTER_COUNT = int(_SRM_BANK.shape[0])

# Backward-compatible 3x3 center-surround high-pass kernel (Laplacian/4). The FIRST SRM filter
# (spam1d3x3, embedded in a 5x5 frame) IS this kernel after DC-renormalization; highpass_kernel
# returns the bare 3x3 form so existing callers / tests that expect a (3,3) kernel keep working.
_HIGHPASS_KERNEL = np.array(
    [[0.0, 1.0, 0.0],
     [1.0, -4.0, 1.0],
     [0.0, 1.0, 0.0]], dtype=np.float32) / 4.0


def highpass_bank() -> np.ndarray:
    """Return a copy of the SRM high-pass filter bank, shape (30, 1, 5, 5), float32.

    Every filter sums to ~0 (DC-suppressed) and the 30 filters are pairwise distinct. This is
    the fixed front-end loaded into the CNN's Conv2d(1,30,5,padding=2,bias=False).
    """
    return _SRM_BANK.copy()


def highpass_kernel() -> np.ndarray:
    """Return a copy of the hard-coded 3x3 high-pass kernel (DC-suppressed: sums to 0).

    Backward-compatible accessor for the bare 3x3 center-surround kernel (the first SRM
    filter in its 3x3 form). Kept so existing callers/tests that expect a (3,3) kernel still
    work; new code should prefer highpass_bank() for the full SRM front-end.
    """
    return _HIGHPASS_KERNEL.copy()


def luminance_array(img_rgb, common_size: int = 256) -> np.ndarray:
    """Numpy-only luminance helper (factored out of the Dataset so it is testable without
    torch). Scale to common_size, Rec.601 luminance, float32 in [0,1], shape (H,W)."""
    small = image_ops.scale_to(img_rgb, common_size)
    arr = np.asarray(small, dtype=np.float32) / 255.0
    if arr.ndim == 3:  # RGB -> Rec.601 luminance
        arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return arr.astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset (torch is imported INSIDE __init__/__getitem__ so module import is torch-free)
# ---------------------------------------------------------------------------
class GANFpDataset:
    """torch Dataset over a list of (path, class_index) pairs.

    Pipeline per item:
      load_rgb -> scale_to(common_size) -> [optional JPEG augment at TRAIN] ->
      Rec.601 luminance float32 [0,1] -> torch tensor (1,H,W).
    No mean/std standardization here (the fixed high-pass front-end operates on raw
    luminance). Augment is OFF at eval time so val/test are deterministic. torch is
    imported inside __getitem__/__init__ so this class body is import-safe without torch.
    """

    def __init__(self, paths: Sequence[str], labels: Sequence[int],
                 common_size: int = 256, augment=None, hflip: bool = False,
                 seed: int = 42, cache: bool = False):
        import torch  # noqa: F401  (gate: constructing the Dataset requires torch)

        self.paths = list(paths)
        self.labels = [int(x) for x in labels]
        self.common_size = int(common_size)
        self.augment = augment
        self.hflip = bool(hflip)
        self.seed = int(seed)
        # Deterministic per-path hflip decision so the same image flips the same way.
        self._flip_cache = {}
        self._cache = None
        if cache:
            # Pre-load every image to a tensor ONCE (RAM). Skips per-batch PIL I/O and torch
            # worker-spawn (which re-spawns every epoch and stalls on Windows), so train/val
            # are fed from memory at full GPU speed. augment/hflip are per-path DETERMINISTIC,
            # so this is identical to the on-the-fly __getitem__ path.
            # ~1626 imgs x 256^2 x 4B ~= 0.4 GB.
            self._cache = []
            for i, path in enumerate(self.paths):
                img = image_ops.load_rgb(path)
                if self.augment is not None:
                    img = self.augment(img, path)
                gray = luminance_array(img, self.common_size)
                if self._should_flip(path):
                    gray = gray[:, ::-1].copy()
                t = torch.from_numpy(np.ascontiguousarray(gray)).unsqueeze(0).float()
                self._cache.append((t, torch.tensor(self.labels[i], dtype=torch.long)))

    def __len__(self) -> int:
        return len(self.paths)

    def _should_flip(self, path: str) -> bool:
        if not self.hflip:
            return False
        if path not in self._flip_cache:
            import zlib
            import random as _random
            h = zlib.crc32(str(path).encode("utf-8")) & 0xFFFFFFFF
            rng = _random.Random((int(self.seed) << 32) ^ h)
            self._flip_cache[path] = rng.random() < 0.5
        return self._flip_cache[path]

    def __getitem__(self, idx: int):
        import torch

        if self._cache is not None:
            return self._cache[idx]

        path = self.paths[idx]
        img = image_ops.load_rgb(path)
        if self.augment is not None:
            img = self.augment(img, path)
        gray = luminance_array(img, self.common_size)
        if self._should_flip(path):
            gray = gray[:, ::-1].copy()
        # (1, H, W) float32 tensor in [0,1]
        t = torch.from_numpy(np.ascontiguousarray(gray)).unsqueeze(0).float()
        return t, torch.tensor(self.labels[idx], dtype=torch.long)


def build_dataloaders(paths: Sequence[str], labels: Sequence[int],
                      common_size: int = 256, augment=None, hflip: bool = False,
                      seed: int = 42, batch_size: int = 32, num_workers: int = 2,
                      shuffle: bool = True, cache: bool = True
                      ) -> "torch.utils.data.DataLoader":
    """Build a DataLoader over GANFpDataset. torch imported lazily.

    cache=True (default) pre-loads every image to a RAM tensor once at construction - the fast
    path on Windows (avoids the slow per-epoch worker spawn) and anywhere I/O would starve the
    GPU; num_workers is forced to 0 in that case (cached data needs no workers). cache=False
    loads per item; then num_workers>0 parallelizes where spawn is cheap.
    """
    import torch
    from torch.utils.data import DataLoader

    if cache:
        num_workers = 0  # cached tensors live in RAM; worker processes would only duplicate them
    ds = GANFpDataset(paths, labels, common_size=common_size, augment=augment,
                      hflip=hflip, seed=seed, cache=cache)
    g = torch.Generator()
    g.manual_seed(int(seed))
    return DataLoader(ds, batch_size=int(batch_size), shuffle=shuffle,
                      num_workers=int(num_workers), generator=g)


# ---------------------------------------------------------------------------
# CNN (torch imported INSIDE __init__/forward)
# ---------------------------------------------------------------------------
class GANFpCNN:
    """nn.Module CNN: frozen high-pass front-end + 3 VGG conv blocks + GAP + linear.

    torch + torch.nn imported INSIDE __init__/forward so the module is import-safe without
    torch. The high-pass Conv2d is created with requires_grad=False and never updated.
    """

    def __init__(self, num_classes: int, input_size: int = 256,
                 channels: Sequence[int] = (32, 64, 128), dropout: float = 0.3):
        import torch
        import torch.nn as nn

        self.num_classes = int(num_classes)
        self.input_size = int(input_size)
        self.channels = tuple(int(c) for c in channels)
        self.dropout = float(dropout)

        class _Net(nn.Module):
            def __init__(self, num_classes, channels, dropout, bank_tensor):
                super().__init__()
                self.num_classes = int(num_classes)
                # Frozen SRM high-pass front-end: Conv2d(1,30,5,padding=2,bias=False). The
                # bank_tensor is (30,1,5,5); 30 -> channels[0] is the first conv block's job.
                self.highpass = nn.Conv2d(1, SRM_FILTER_COUNT, 5, padding=2, bias=False)
                with torch.no_grad():
                    self.highpass.weight.copy_(bank_tensor)
                for p in self.highpass.parameters():
                    p.requires_grad = False

                last = SRM_FILTER_COUNT
                self.blocks = nn.ModuleList()
                for ch in channels:
                    block = nn.Sequential(
                        nn.Conv2d(last, ch, 3, padding=1, bias=False),
                        nn.BatchNorm2d(ch),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                        nn.BatchNorm2d(ch),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(2),
                    )
                    self.blocks.append(block)
                    last = ch
                self.gap = nn.AdaptiveAvgPool2d(1)
                self.head = nn.Sequential(
                    nn.Linear(last, 128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(128, num_classes),
                )

            def forward(self, x):
                x = self.highpass(x)
                for block in self.blocks:
                    x = block(x)
                x = self.gap(x).flatten(1)
                return self.head(x)

        bank = torch.from_numpy(_SRM_BANK)  # (30,1,5,5)
        self.model = _Net(self.num_classes, self.channels, self.dropout, bank)

    def __call__(self, x):
        return self.model(x)

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def param_count(self) -> int:
        return int(sum(p.numel() for p in self.trainable_parameters()))


# ---------------------------------------------------------------------------
# Classifier wrapper: mirrors defake_head._MLPHead (fit/predict_proba/predict/save)
# ---------------------------------------------------------------------------
class GANFpClassifier:
    """Wrap the CNN + Adam + CrossEntropy(class_weights).

    NOTE ON THE PREDICT SURFACE: unlike defake_head._MLPHead (which consumes a precomputed
    feature matrix X via predict_proba(X)), this is an END-TO-END CNN whose input is raw
    images. Its predict entry points therefore consume image PATHS (+ placeholder labels so
    the DataLoader collates), NOT a feature matrix. The names overlap with _MLPHead
    (predict_proba / predict / save(path, classes)) but the CNN signature is
    predict_proba(paths, labels, ...) -> [N, C]. The benchmark keeps Path A and Path B on
    separate, explicit call sites (run_path_a vs run_path_b) precisely because the two inputs
    differ (features vs paths); there is no single uniform predict call. Best-val checkpoint
    is kept (mirrors _MLPHead.fit). torch lazy.
    """

    def __init__(self, num_classes: int, input_size: int = 256,
                 channels: Sequence[int] = (32, 64, 128), dropout: float = 0.3,
                 device: str = "cpu", lr: float = 1e-3, weight_decay: float = 1e-4,
                 seed: int = 42):
        import torch

        torch.manual_seed(int(seed))
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.seed = int(seed)
        self.num_classes = int(num_classes)
        self.cnn = GANFpCNN(num_classes=num_classes, input_size=input_size,
                            channels=channels, dropout=dropout)
        self.cnn.model.to(self.device)

    def fit(self, train_loader, val_loader=None, epochs: int = 30,
            class_weights=None, logger=None):
        """Train with Adam + weighted CrossEntropy. Keeps the best-val-acc checkpoint."""
        import torch
        import torch.nn as nn

        device = self.device
        cw = None
        if class_weights is not None:
            cw = torch.tensor(class_weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=cw)
        optim = torch.optim.Adam(self.cnn.trainable_parameters(), lr=self.lr,
                                 weight_decay=self.weight_decay)

        best_val = -1.0
        best_state = None
        for epoch in range(int(epochs)):
            self.cnn.model.train()
            running = 0.0
            nb = 0
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optim.zero_grad()
                logits = self.cnn(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optim.step()
                running += float(loss.item())
                nb += 1
            if val_loader is not None:
                val_acc = self._eval_accuracy(val_loader)
                if val_acc > best_val:
                    best_val = val_acc
                    best_state = {k: v.detach().clone()
                                  for k, v in self.cnn.model.state_dict().items()}
                if logger and (epoch % 5 == 0 or epoch == epochs - 1):
                    logger.info("cnn epoch %d loss=%.4f val_acc=%.3f",
                                epoch, running / max(1, nb), val_acc)
        if best_state is not None:
            self.cnn.model.load_state_dict(best_state)
        return self

    def _eval_accuracy(self, loader) -> float:
        import torch

        self.cnn.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                pred = self.cnn(xb).argmax(dim=1)
                correct += int((pred == yb).sum().item())
                total += int(yb.numel())
        return correct / max(1, total)

    def predict_proba_loader(self, loader) -> np.ndarray:
        """Return softmax probabilities [N, num_classes] for a DataLoader."""
        import torch

        self.cnn.model.eval()
        outs = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(self.device)
                logits = self.cnn(xb)
                outs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        if not outs:
            return np.zeros((0, self.num_classes), dtype=np.float32)
        return np.concatenate(outs, axis=0).astype(np.float32)

    def predict_proba(self, paths: Sequence[str], labels: Sequence[int],
                      common_size: int = 256, batch_size: int = 32,
                      num_workers: int = 0) -> np.ndarray:
        """Predict softmax probs [N, num_classes] over image PATHS (end-to-end CNN).

        CNN-specific signature -- does NOT take a feature matrix (unlike _MLPHead.predict_proba
        which takes X). `labels` are placeholders so the eval DataLoader collates; only the
        image paths drive the output. For a precomputed-tensor path use predict_proba_loader.
        """
        loader = build_dataloaders(paths, labels, common_size=common_size, augment=None,
                                   hflip=False, seed=self.seed, batch_size=batch_size,
                                   num_workers=num_workers, shuffle=False)
        return self.predict_proba_loader(loader)

    def predict(self, paths: Sequence[str], labels: Sequence[int],
                common_size: int = 256, batch_size: int = 32,
                num_workers: int = 0) -> np.ndarray:
        return self.predict_proba(paths, labels, common_size, batch_size,
                                  num_workers).argmax(axis=1)

    def save(self, path: str, classes: List[str]):
        import torch

        torch.save({"state_dict": self.cnn.model.state_dict(),
                    "classes": classes}, path)

    @property
    def param_count(self) -> int:
        return self.cnn.param_count()


# ---------------------------------------------------------------------------
# Split helper: return the same (paths, labels) sliced by index arrays (shared split)
# ---------------------------------------------------------------------------
def slice_paths_labels(paths: Sequence[str], labels: Sequence[int],
                       idx: Sequence[int]) -> Tuple[List[str], List[int]]:
    """Return (paths[idx], labels[idx]) as plain lists (shared-split slicing helper)."""
    return [paths[i] for i in idx], [int(labels[i]) for i in idx]
