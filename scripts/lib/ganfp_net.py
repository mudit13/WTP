"""
GAN Fingerprints CNN path (Yu2019 faithful): a compact VGG-style CNN over single-channel
luminance, with a FIXED (non-trainable) SRM high-pass front-end.

Path B of the GAN-fp reproduction. Path A (residual/spectrum features + defake_head._MLPHead)
lives in ganfp.py; this module is the end-to-end CNN alternative. Both share the SAME seeded
stratified split and the SAME per-image JPEG augmentation (image_ops.make_jpeg_augmenter) so
the head-to-head benchmark (scripts/benchmark_attribution.py) is apples-to-apples.

Architecture (trainable params scale with config.ganfp.cnn.channels; default [32,64,128]
~330K params, trains in minutes on CUDA / modest time on CPU):
  - Frozen SRM high-pass front-end, faithful to Fridrich & Kodovsky 2012, "Rich Models for
    Steganalysis" (the AUTHORITATIVE SRM source). The paper defines RESIDUAL/PREDICTOR
    FAMILIES (SPAM, SQUARE, EDGE), not a fixed 30-kernel bank; the "30 SRM filters" is a
    downstream CNN-steganalysis convention. This front-end reconstructs the paper's families:
      * LINEAR branch: a Conv2d(1, SRM_FILTER_COUNT, 5, padding=2, bias=False) loaded with
        genuine paper-family kernels:
          - SPAM 1st-order [1,-1] directional (H/V/2 diagonals),
          - SPAM 2nd-order [1,-2,1] (H/V/2 diagonals),
          - SPAM 3rd/4th-order ([1,-3,3,-1], [1,-4,6,-4,1]) and spam14 (the 4th-order 2-D
            workhorse),
          - SQUARE S3a (3x3 L2-optimal shift-invariant) and S5a (5x5 circular-symmetric),
          - EDGE3x3 (E3a-E3d) / EDGE5x5 (E5a-E5d) edge predictors derived from S3a/S5a.
      * NONLINEAR MINMAX branch (the paper's MOST discriminative signal -- minmax24 was the
        single best submodel, 33 of 106 submodels): the pointwise torch.minimum /
        torch.maximum over the directional 1st/2nd-order SPAM residual maps. Minmax is a
        pixel-wise operator over >=2 linear residuals and CANNOT be a single conv kernel. It
        is fixed (non-trainable) and differentiable. These channels are concatenated onto the
        linear bank so the VGG blocks receive (linear + minmax) channels.
    Every LINEAR kernel is DC-suppressed structurally (residual = X(center) - predictor, so
    the kernel sums to ~0); _norm0 enforces an exact-zero sum via the center cell (Eq.1).
    requires_grad=False on the whole front-end; it is never updated.
    NO per-kernel L2 normalization is applied (the paper does not prescribe it; magnitude
    differences are absorbed by the downstream BatchNorm). NO mean/std z-scoring: the
    high-pass front-end + BatchNorm combo replaces it (a constant image -> ~0 residual).
  - 3 VGG-style conv blocks (Conv-BN-ReLU x2 + MaxPool), channels default [32,64,128]. The
    FIRST conv's input channels = SRM_FILTER_COUNT + SRM_MINMAX_CHANNEL_COUNT.
  - AdaptiveAvgPool2d(1) -> Flatten -> Linear(channels[-1],128)+ReLU+Dropout(0.3) ->
    Linear(128,C).

This is a faithful FAMILY-LEVEL reconstruction of the SRM, NOT an "exact SRM" (the paper does
not define a single fixed kernel bank). Where the paper leaves a kernel non-unique / sign /
order-ambiguous, a distinct DC-suppressed kernel from the same family is used.

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
# SRM (Spatial Rich Model) high-pass filter bank -- faithful family reconstruction.
# Fridrich & Kodovsky 2012, "Rich Models for Steganalysis" (AUTHORITATIVE). The paper defines
# RESIDUAL/PREDICTOR FAMILIES, not a fixed 30-kernel bank. This bank reconstructs the genuine
# linear families: SPAM (1st/2nd/3rd/4th-order directional predictors + the 2-D spam14
# workhorse), SQUARE (S3a 3x3 L2-optimal, S5a 5x5 circular-symmetric), and EDGE (E3a-E3d 3x3,
# E5a-E5d 5x5 -- edge predictors derived from S3a/S5a). One honest Laplacian-4 center-surround
# kernel is kept (it is a legitimate linear residual, NOT SPAM; it is labelled as such).
#
# Each filter is DC-suppressed structurally: the residual is r = X(center) - predictor, so the
# kernel weights sum to ~0. _norm0 makes that sum EXACTLY 0 (a constant image yields ~0 on
# every filter). This is the defining high-pass / residual property the test_highpass_bank_*
# tests assert.
#
# The paper's MOST discriminative family -- MINMAX (minmax24 was the single best submodel) --
# is NOT a conv kernel: it is the pointwise minimum / maximum of >=2 linear residual maps,
# a nonlinear, pixel-wise operator. It is realized as a separate NONLINEAR branch in the CNN
# forward (torch.minimum / torch.maximum over directional SPAM residuals), NOT as part of this
# linear bank. The linear directional channels the MINMAX branch consumes are recorded in
# SRM_MINMAX_SOURCE_PAIRS below.
#
# Rejected (NOT in this bank): the denoising-style filters the paper rejects in Sec II-E --
# Laplacian-8 (large central weight 8), Laplacian-of-Gaussian / LoG (central weight 16), and
# the square-Laplacian cubic residual. These bias the predictor and suppress the signal; they
# are POST-paper (Kang 2013 / Xu-Net / Ye-Net "KB/LoG/KV") and are excluded here.
#
# No per-kernel L2 normalization is applied -- the paper does not prescribe it (normalization
# is at residual level q/T and histogram level n). Downstream BatchNorm absorbs the linear
# kernels' magnitude differences.
# ---------------------------------------------------------------------------

def _norm0(k: np.ndarray) -> np.ndarray:
    """Zero-mean a kernel via its CENTER cell so the weights sum to EXACTLY 0 (DC-suppressed),
    preserving the sparsity pattern and high-pass character (Eq.1 of the SRM paper).

    The center cell absorbs the residual so the sum is exactly 0 and every off-center
    coefficient keeps its published value. This is distinct from subtracting the global mean,
    which would smear a constant offset into every cell (including the zero halo) and make
    otherwise-distinct sparse filters collinear. NO per-kernel L2 normalization is applied
    (the paper does not prescribe it; downstream BatchNorm absorbs magnitude differences).
    """
    k = np.asarray(k, dtype=np.float32).copy()
    s = float(k.sum())
    if abs(s) > 1e-12:
        c = k.shape[0] // 2
        k[c, c] = k[c, c] - s
    return k.astype(np.float32)


def _build_srm_bank():
    """Build the LINEAR SRM high-pass filter bank as float32 ndarray (N, 1, 5, 5), plus the
    index metadata that drives the nonlinear MINMAX branch.

    All kernels are embedded in a 5x5 frame (3x3 kernels sit in the center with a zero halo)
    so the returned ndarray is uniformly shaped for a single Conv2d(1,N,5,padding=2). Every
    kernel sums to ~0 after _norm0.

    This is a faithful FAMILY-LEVEL reconstruction of Fridrich-Kodovsky 2012 -- SPAM
    (1st/2nd/3rd/4th-order directional predictors + spam14), SQUARE (S3a 3x3, S5a 5x5), EDGE
    (E3a-d 3x3, E5a-d 5x5), and one honest Laplacian-4 center-surround kernel. The rejected
    denoising-style kernels (Laplacian-8, LoG, square-Laplacian cubic) are EXCLUDED (Sec II-E).

    Returns
    -------
    bank : np.ndarray, shape (N, 1, 5, 5), float32
        The linear kernel bank (N = SRM_FILTER_COUNT).
    families : list of str, len N
        The paper-family label of each linear channel ('spam1','spam2','spam3','spam4',
        'spam14','square','edge','lap4').
    minmax_pairs : list of (int, int)
        Index pairs into the linear bank whose residual maps the MINMAX branch reduces with
        torch.minimum / torch.maximum (1st/2nd-order directional SPAM pairs, per the paper).
    """
    filters = []
    families = []  # paper-family label per linear channel

    def add3(k3, fam):
        """Embed a 3x3 kernel in the center of a 5x5 zero frame, then DC-suppress."""
        f = np.zeros((5, 5), dtype=np.float32)
        f[1:4, 1:4] = np.asarray(k3, dtype=np.float32)
        filters.append(_norm0(f))
        families.append(fam)

    def add5(k5, fam):
        filters.append(_norm0(np.asarray(k5, dtype=np.float32)))
        families.append(fam)

    # ---- SPAM 1st-order: genuine 2-tap [1,-1] directional predictors ------------
    # Fridrich-Kodovsky: 1st-order SPAM = the 2-tap difference in each of 8 directions.
    # 4 axis-aligned/diagonal directions embedded in 3x3 (the other 4 are sign flips; the
    # nonlinear MINMAX branch captures directionality, so one of each axis is enough here).
    add3([[0.0, 0.0, 0.0],
          [1.0, -1.0, 0.0],
          [0.0, 0.0, 0.0]], "spam1")          # horizontal (W->E)
    add3([[0.0, 1.0, 0.0],
          [0.0, -1.0, 0.0],
          [0.0, 0.0, 0.0]], "spam1")          # vertical (N->S)
    add3([[1.0, 0.0, 0.0],
          [0.0, -1.0, 0.0],
          [0.0, 0.0, 0.0]], "spam1")          # diagonal NW->SE
    add3([[0.0, 0.0, 1.0],
          [0.0, -1.0, 0.0],
          [0.0, 0.0, 0.0]], "spam1")          # anti-diagonal NE->SW

    # ---- SPAM 2nd-order: [1,-2,1] in 4 directions (H/V/2 diagonals) ------------
    add3([[0.0, 0.0, 0.0],
          [1.0, -2.0, 1.0],
          [0.0, 0.0, 0.0]], "spam2")          # horizontal 2nd derivative
    add3([[0.0, 1.0, 0.0],
          [0.0, -2.0, 0.0],
          [0.0, 1.0, 0.0]], "spam2")          # vertical 2nd derivative
    add3([[1.0, 0.0, 0.0],
          [0.0, -2.0, 0.0],
          [0.0, 0.0, 1.0]], "spam2")          # diagonal (NW-SE) 2nd derivative
    add3([[0.0, 0.0, 1.0],
          [0.0, -2.0, 0.0],
          [1.0, 0.0, 0.0]], "spam2")          # anti-diagonal (NE-SW) 2nd derivative

    # ---- SQUARE S3a: 3x3 L2-optimal shift-invariant predictor -----------------
    # S3a: center minus the average of its 8 neighbours (the 3x3 L2-optimal edge predictor
    # underlying the SQUARE family). DC-suppressed by construction.
    add3([[-1.0, -1.0, -1.0],
          [-1.0,  8.0, -1.0],
          [-1.0, -1.0, -1.0]], "square")

    # ---- EDGE3x3 (E3a-E3d): edge predictors derived from S3a ------------------
    # Fridrich-Kodovsky EDGE predictors = directional edge residuals: the center minus the
    # AVERAGE of a half-neighbourhood on one side (a 3-cell L-/T-shaped stencil), NOT a single
    # neighbour (that would just re-create a SPAM-1 2-tap kernel). E3a..E3d span the four
    # cardinal edge orientations and are non-collinear with the SPAM derivatives.
    add3([[ 1.0,  1.0,  1.0],                 # E3a: top edge (avg of the top row)
          [ 0.0, -3.0,  0.0],
          [ 0.0,  0.0,  0.0]], "edge")
    add3([[ 0.0,  0.0,  0.0],                 # E3b: bottom edge (avg of the bottom row)
          [ 0.0, -3.0,  0.0],
          [ 1.0,  1.0,  1.0]], "edge")
    add3([[ 1.0,  0.0,  0.0],                 # E3c: left edge (avg of the left column)
          [ 1.0, -3.0,  0.0],
          [ 1.0,  0.0,  0.0]], "edge")
    add3([[ 0.0,  0.0,  1.0],                 # E3d: right edge (avg of the right column)
          [ 0.0, -3.0,  1.0],
          [ 0.0,  0.0,  1.0]], "edge")

    # ---- Laplacian-4 center-surround (honest label; NOT SPAM) -----------------
    # The classic 4-neighbour Laplacian/4. Kept as a legitimate linear residual but labelled
    # honestly as 'lap4' (it is the center-vs-4neighbours operator, distinct from SPAM and
    # from the rejected large-central-weight Laplacian-8).
    add3([[0.0,  1.0, 0.0],
          [1.0, -4.0, 1.0],
          [0.0,  1.0, 0.0]], "lap4")

    # =====================================================================
    # 5x5 family
    # =====================================================================

    # ---- SQUARE S5a: 5x5 circular-symmetric predictor ------------------------
    # S5a: center minus a weighted average over its 5x5 neighbourhood (the 8 king-distance-1
    # cells and the 16 king-distance-2 cells). This is the 5x5 L2-optimal shift-invariant
    # edge predictor of the SQUARE family; DC-suppressed by construction.
    add5(np.array([
        [-1, -2, -1, -2, -1],
        [-2, -1, -1, -1, -2],
        [-1, -1, 24, -1, -1],
        [-2, -1, -1, -1, -2],
        [-1, -2, -1, -2, -1]], dtype=np.float32), "square")

    # ---- SPAM14: the 4th-order 2-D SPAM residual (the SRM workhorse) ----------
    # spam14 = outer product of the 1-D [1,-4,6,-4,1] 4th-difference with itself, scaled so
    # the center balances the sum (the canonical 5x5 2-D 4th-order residual).
    spam14 = np.array([
        [ 1, -4,  6, -4,  1],
        [-4, 16, -24, 16, -4],
        [ 6, -24, 36, -24, 6],
        [-4, 16, -24, 16, -4],
        [ 1, -4,  6, -4,  1]], dtype=np.float32)
    add5(spam14, "spam14")

    # ---- SPAM 4th-order 1-D: [1,-4,6,-4,1] in H / V / 2 diagonals ------------
    add5(np.array([
        [0, 0, 0, 0, 0],
        [1, -4, 6, -4, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]], dtype=np.float32), "spam4")   # horizontal 4th
    add5(np.array([
        [0, 1, 0, 0, 0],
        [0, -4, 0, 0, 0],
        [0, 6, 0, 0, 0],
        [0, -4, 0, 0, 0],
        [0, 1, 0, 0, 0]], dtype=np.float32), "spam4")   # vertical 4th
    add5(np.array([
        [1, 0, 0, 0, 0],
        [0, -4, 0, 0, 0],
        [0, 0, 6, 0, 0],
        [0, 0, 0, -4, 0],
        [0, 0, 0, 0, 1]], dtype=np.float32), "spam4")   # diagonal (NW-SE) 4th
    add5(np.array([
        [0, 0, 0, 0, 1],
        [0, 0, 0, -4, 0],
        [0, 0, 6, 0, 0],
        [0, -4, 0, 0, 0],
        [1, 0, 0, 0, 0]], dtype=np.float32), "spam4")   # anti-diagonal (NE-SW) 4th

    # ---- SPAM 3rd-order 1-D: [1,-3,3,-1] in H / V / 2 diagonals --------------
    add5(np.array([
        [0, 0, 0, 0, 0],
        [1, -3, 3, -1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]], dtype=np.float32), "spam3")   # horizontal 3rd (left-pointing)
    add5(np.array([
        [0, 0, 0, 0, 0],
        [0, 1, -3, 3, -1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]], dtype=np.float32), "spam3")   # horizontal 3rd (right-pointing)
    add5(np.array([
        [0, 1, 0, 0, 0],
        [0, -3, 0, 0, 0],
        [0, 3, 0, 0, 0],
        [0, -1, 0, 0, 0],
        [0, 0, 0, 0, 0]], dtype=np.float32), "spam3")   # vertical 3rd (down-pointing)
    add5(np.array([
        [0, 0, 0, 0, 0],
        [0, -1, 0, 0, 0],
        [0, 3, 0, 0, 0],
        [0, -3, 0, 0, 0],
        [0, 1, 0, 0, 0]], dtype=np.float32), "spam3")   # vertical 3rd (up-pointing)
    add5(np.array([
        [1, 0, 0, 0, 0],
        [0, -3, 0, 0, 0],
        [0, 0, 3, 0, 0],
        [0, 0, 0, -1, 0],
        [0, 0, 0, 0, 0]], dtype=np.float32), "spam3")   # diagonal (NW-SE) 3rd
    add5(np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, -1, 0],
        [0, 0, 3, 0, 0],
        [0, -3, 0, 0, 0],
        [1, 0, 0, 0, 0]], dtype=np.float32), "spam3")   # anti-diagonal (NE-SW) 3rd

    # ---- EDGE5x5 (E5a-E5d): edge predictors derived from S5a -----------------
    # 5x5 directional edge residuals: the center minus the AVERAGE of a half 5x5 neighbourhood
    # on one side (top/bottom/left/right -- an 8-cell directional stencil on the corresponding
    # half of the 5x5 window). Distinct from the SPAM 1-D derivatives and from the E3a-d kernels.
    e5a = np.array([
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [2, 2, -8, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0]], dtype=np.float32)             # E5a: left half-ring edge
    add5(e5a, "edge")
    add5(np.fliplr(e5a), "edge")                        # E5b: right half-ring edge
    add5(e5a.T, "edge")                                 # E5c: top half-ring edge
    add5(np.flipud(e5a.T), "edge")                      # E5d: bottom half-ring edge

    bank = np.stack(filters, axis=0)[:, None, :, :].astype(np.float32)  # (N,1,5,5)

    # ---- MINMAX source pairs -------------------------------------------------
    # The paper's MINMAX is the pointwise min/max of >=2 directional LINEAR residual maps.
    # We pair the axis/diagonal channels of the 1st- and 2nd-order SPAM families (the most
    # discriminative directional residuals). Each pair yields a min channel + a max channel.
    # Indices refer to positions in `families`.
    spam1_idx = [i for i, f in enumerate(families) if f == "spam1"]
    spam2_idx = [i for i, f in enumerate(families) if f == "spam2"]
    # Pair the matching directional axes: (spam1 horizontal, spam2 horizontal), etc. We pair
    # same-direction 1st/2nd residuals (the paper's directional minmax), and also reduce each
    # family across its directions (omnidirectional minmax).
    minmax_pairs = []
    n_dir = min(len(spam1_idx), len(spam2_idx))
    for d in range(n_dir):
        minmax_pairs.append((spam1_idx[d], spam2_idx[d]))  # directional 1st-vs-2nd min/max
    # Omnidirectional: min/max across all 1st-order directions, and across all 2nd-order.
    # (Realized pairwise on the first vs each subsequent direction, then the channel count is
    # bounded -- see SRM_MINMAX_SOURCE_PAIRS computed below.)
    if len(spam1_idx) >= 2:
        minmax_pairs.append((spam1_idx[0], spam1_idx[1]))
    if len(spam2_idx) >= 2:
        minmax_pairs.append((spam2_idx[0], spam2_idx[1]))

    return bank, families, minmax_pairs


# Build once at import (numpy-only). bank: (N,1,5,5); families: per-channel labels;
# _SRM_MINMAX_PAIRS: index pairs whose residual maps the nonlinear MINMAX branch reduces.
_SRM_BANK, _SRM_FAMILIES, _SRM_MINMAX_PAIRS = _build_srm_bank()
# Number of LINEAR filters in the SRM front-end (exposed for tests + the CNN to size its Conv2d).
SRM_FILTER_COUNT = int(_SRM_BANK.shape[0])
# Paper-family index map: family name -> sorted list of linear channel indices.
SRM_FAMILY_INDEX = {}
for _i, _f in enumerate(_SRM_FAMILIES):
    SRM_FAMILY_INDEX.setdefault(_f, []).append(_i)
# Index pairs (i, j) into the linear bank whose residual maps the nonlinear MINMAX branch
# reduces with torch.minimum / torch.maximum. Each pair produces 2 channels (min + max).
SRM_MINMAX_SOURCE_PAIRS = [tuple(p) for p in _SRM_MINMAX_PAIRS]
# Number of NONLINEAR minmax channels appended after the linear bank (2 per source pair:
# one pointwise-min, one pointwise-max). These are computed in the CNN forward (not a kernel).
SRM_MINMAX_CHANNEL_COUNT = 2 * len(SRM_MINMAX_SOURCE_PAIRS)
# Total input channels the first VGG conv receives: linear bank + nonlinear minmax channels.
SRM_FRONTEND_CHANNEL_COUNT = SRM_FILTER_COUNT + SRM_MINMAX_CHANNEL_COUNT

# Backward-compatible 3x3 Laplacian-4 center-surround high-pass kernel. The 'lap4' SRM filter
# (embedded in a 5x5 frame) IS this kernel after DC-normalization; highpass_kernel returns the
# bare 3x3 form so existing callers / tests that expect a (3,3) kernel keep working. It is
# honestly the Laplacian-4, NOT a SPAM kernel.
_HIGHPASS_KERNEL = np.array(
    [[0.0, 1.0, 0.0],
     [1.0, -4.0, 1.0],
     [0.0, 1.0, 0.0]], dtype=np.float32) / 4.0


def highpass_bank() -> np.ndarray:
    """Return a copy of the LINEAR SRM high-pass filter bank, shape (SRM_FILTER_COUNT, 1, 5, 5),
    float32. Every filter sums to ~0 (DC-suppressed) and the filters are pairwise distinct.

    This is the LINEAR part of the front-end only. The nonlinear MINMAX channels are produced
    in the CNN forward (they are pointwise min/max of directional SPAM residual maps and
    cannot be conv kernels). Use SRM_FRONTEND_CHANNEL_COUNT for the full first-conv input size.
    """
    return _SRM_BANK.copy()


def highpass_kernel() -> np.ndarray:
    """Return a copy of the hard-coded 3x3 Laplacian-4 center-surround high-pass kernel
    (DC-suppressed: sums to 0). Honestly the Laplacian-4 (NOT SPAM). Backward-compatible
    accessor for callers/tests that expect a bare (3,3) kernel.
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
            def __init__(self, num_classes, channels, dropout, bank_tensor,
                         minmax_pairs, minmax_channels):
                super().__init__()
                self.num_classes = int(num_classes)
                # Frozen LINEAR SRM high-pass front-end: Conv2d(1, SRM_FILTER_COUNT, 5,
                # padding=2, bias=False). The bank_tensor is (N,1,5,5). Every weight is frozen.
                self.highpass = nn.Conv2d(1, SRM_FILTER_COUNT, 5, padding=2, bias=False)
                with torch.no_grad():
                    self.highpass.weight.copy_(bank_tensor)
                for p in self.highpass.parameters():
                    p.requires_grad = False
                # Nonlinear MINMAX branch metadata (fixed). For each (i, j) pair the forward
                # appends torch.minimum(Ri, Rj) and torch.maximum(Ri, Rj) channels, where Ri/Rj
                # are the i-th/j-th LINEAR residual maps. These are pointwise, non-trainable,
                # and differentiable; they CANNOT be conv kernels (they are the paper's most
                # discriminative signal -- minmax24 was the single best SRM submodel).
                self.minmax_pairs = list(minmax_pairs)
                self.minmax_channels = int(minmax_channels)
                # First VGG conv receives (linear + minmax) channels.
                last = SRM_FRONTEND_CHANNEL_COUNT
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
                # LINEAR residual maps: (B, SRM_FILTER_COUNT, H, W). DC-suppressed per filter.
                resid = self.highpass(x)
                # NONLINEAR MINMAX branch: pointwise min/max over the directional SPAM residual
                # maps (the paper's most discriminative family). Fixed + differentiable; not a
                # conv. Produces 2 channels per source pair (min + max).
                minmax_outs = []
                for (i, j) in self.minmax_pairs:
                    ri = resid[:, i:i + 1, :, :]
                    rj = resid[:, j:j + 1, :, :]
                    minmax_outs.append(torch.minimum(ri, rj))
                    minmax_outs.append(torch.maximum(ri, rj))
                if minmax_outs:
                    mm = torch.cat(minmax_outs, dim=1)
                    x = torch.cat([resid, mm], dim=1)
                else:
                    x = resid
                for block in self.blocks:
                    x = block(x)
                x = self.gap(x).flatten(1)
                return self.head(x)

        bank = torch.from_numpy(_SRM_BANK)  # (N,1,5,5)
        self.model = _Net(self.num_classes, self.channels, self.dropout, bank,
                          SRM_MINMAX_SOURCE_PAIRS, SRM_MINMAX_CHANNEL_COUNT)

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
