"""Tests for lib/ganfp (residual/spectrum fingerprint extraction).

Pure-Python: numpy + scipy + Pillow only. torch is NOT imported by ganfp, so these run under
CI (no torch) - they cover the feature math, dimensionality, scanning, and realignment.
"""
import os

import numpy as np
from PIL import Image

from lib import ganfp


def _toy_image(path, size=64, seed=0):
    """Write a deterministic random RGB image."""
    arr = (np.random.RandomState(seed).rand(size, size, 3) * 255).astype(np.uint8)
    Image.fromarray(arr, "RGB").save(path)


def test_feature_dim():
    assert ganfp._feature_dim(256, 32, "both") == 2 * 32 * 32
    assert ganfp._feature_dim(256, 32, "residual") == 32 * 32
    assert ganfp._feature_dim(256, 24, "spectrum") == 24 * 24


def test_extract_shape_and_norm(tmp_path):
    p = tmp_path / "a.png"
    _toy_image(str(p))
    X, kept = ganfp.extract_fingerprints([str(p)], common_size=64, feat_size=16, mode="both")
    assert X.shape == (1, 2 * 16 * 16)
    assert X.dtype == np.float32
    assert kept == [str(p)]
    # L2-normalized fingerprint vector
    assert abs(float(np.linalg.norm(X[0])) - 1.0) < 1e-4


def test_extract_modes(tmp_path):
    p = tmp_path / "a.png"
    _toy_image(str(p))
    for mode, blocks in [("residual", 1), ("spectrum", 1), ("both", 2)]:
        X, _ = ganfp.extract_fingerprints([str(p)], common_size=64, feat_size=12, mode=mode)
        assert X.shape == (1, blocks * 12 * 12), mode


def test_skip_unreadable(tmp_path):
    good = tmp_path / "g.png"
    _toy_image(str(good))
    X, kept = ganfp.extract_fingerprints(
        [str(good), str(tmp_path / "missing.png")], common_size=64, feat_size=8)
    assert X.shape[0] == 1
    assert kept == [str(good)]


def test_empty_paths():
    X, kept = ganfp.extract_fingerprints([], common_size=64, feat_size=8)
    assert X.shape == (0, 2 * 8 * 8)
    assert kept == []


def test_scan_sample_dir(tmp_path):
    for gen in ("StyleGAN3-FFHQ", "FFHQ"):
        d = tmp_path / gen
        d.mkdir()
        _toy_image(str(d / "1.png"))
        _toy_image(str(d / "2.jpg"))
    paths, generators = ganfp.scan_sample_dir(str(tmp_path))
    assert len(paths) == 4
    assert set(generators) == {"StyleGAN3-FFHQ", "FFHQ"}
    assert all(p.lower().endswith((".png", ".jpg")) for p in paths)


def test_features_from_samples(tmp_path):
    rows = []
    for gen in ("StyleGAN3-FFHQ", "FFHQ"):
        d = tmp_path / gen
        d.mkdir()
        for i in range(2):
            _toy_image(str(d / ("%d.png" % i)), seed=i)
            rows.append((str(d / ("%d.png" % i)), gen,
                         "real" if gen == "FFHQ" else "fake"))
    paths = [r[0] for r in rows]
    generators = [r[1] for r in rows]
    labels = [r[2] for r in rows]
    X, g, l, p = ganfp.features_from_samples(
        paths, generators, labels, common_size=64, feat_size=10, mode="both")
    assert X.shape[0] == 4
    assert X.shape[1] == 2 * 10 * 10
    assert set(g.tolist()) == {"StyleGAN3-FFHQ", "FFHQ"}
    assert set(l.tolist()) == {"real", "fake"}
    assert set(p.tolist()) == set(paths)
