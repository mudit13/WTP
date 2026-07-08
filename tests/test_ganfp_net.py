"""Tests for lib/ganfp_net (the GAN-fp CNN path) + lib/ganfp FingerprintStandardizer.

Pure-Python tests (numpy only, no torch) cover the SRM high-pass front-end (faithful
family-level reconstruction of Fridrich-Kodovsky 2012: linear SPAM/SQUARE/EDGE kernels + the
nonlinear MINMAX branch metadata), the GAN-only attribution slice helper (lib.metrics), and
the numpy luminance helper, so they run under CI with no torch installed. The CNN forward /
param budget / one-step / Dataset / MINMAX-branch tests are torch-GATED via
pytest.importorskip("torch"): they only run when torch is available (local / venv_sd15), never
in CI.

GANFpDataset and GANFpCNN import torch INSIDE their class bodies, so importing
ganfp_net at module top here is torch-free; the torch-gated block re-imports torch.
"""
import numpy as np
import pytest
from PIL import Image

from lib import ganfp_net, ganfp, metrics


# ---------------------------------------------------------------------------
# Pure-Python (no torch): SRM high-pass front-end (faithful family reconstruction)
# ---------------------------------------------------------------------------
def test_highpass_kernel_dc_suppressed():
    """The backward-compatible 3x3 Laplacian-4 high-pass kernel sums to 0 (DC-suppressed);
    convolving a constant image gives ~0 residual everywhere. (Honest label: this IS the
    Laplacian-4 center-surround, NOT a SPAM kernel.)"""
    kernel = ganfp_net.highpass_kernel()
    assert kernel.shape == (3, 3)
    assert abs(float(kernel.sum())) < 1e-12, "high-pass kernel must sum to zero"

    ones = np.ones((16, 16), dtype=np.float32)
    # Manual 'same' convolution with edge (replicate) padding so a constant image stays
    # constant at the borders (zero-padding would break DC-suppression at the edges; the
    # CNN's BatchNorm/MaxPool absorb that border effect, but the pure-kernel invariance is
    # only exact under replicate padding).
    pad = np.pad(ones, 1, mode="edge")
    out = np.zeros_like(ones)
    for i in range(ones.shape[0]):
        for j in range(ones.shape[1]):
            out[i, j] = float(np.sum(pad[i:i + 3, j:j + 3] * kernel))
    assert float(np.abs(out).max()) < 1e-6, "constant image must yield ~0 residual"


def test_srm_bank_shape_and_count():
    """The LINEAR SRM front-end bank is (SRM_FILTER_COUNT, 1, 5, 5) float32 and declares a
    clean count of genuine paper-family linear kernels (~24-30)."""
    bank = ganfp_net.highpass_bank()
    assert bank.dtype == np.float32
    assert bank.ndim == 4
    assert bank.shape == (ganfp_net.SRM_FILTER_COUNT, 1, 5, 5)
    # Faithful family reconstruction: ~24-30 genuine linear kernels spanning SPAM/SQUARE/EDGE.
    assert 24 <= ganfp_net.SRM_FILTER_COUNT <= 34, \
        "SRM linear bank should be ~24-30 kernels, got %d" % ganfp_net.SRM_FILTER_COUNT


def test_srm_bank_paper_families():
    """The linear bank spans the genuine Fridrich-Kodovsky families: SPAM (1st/2nd/3rd/4th
    + spam14), SQUARE (S3a 3x3, S5a 5x5), EDGE (E3a-d 3x3, E5a-d 5x5), and one honest
    Laplacian-4. The rejected denoising kernels (Laplacian-8, LoG, cubic) are ABSENT."""
    fam = ganfp_net.SRM_FAMILY_INDEX
    # SPAM families present (the paper's predictor classes).
    assert "spam1" in fam and len(fam["spam1"]) >= 4, "need >=4 1st-order SPAM directional kernels"
    assert "spam2" in fam and len(fam["spam2"]) >= 4, "need >=4 2nd-order SPAM directional kernels"
    assert "spam14" in fam and len(fam["spam14"]) >= 1, "need the spam14 2-D 4th-order workhorse"
    # SQUARE family: S3a (3x3) and S5a (5x5).
    assert "square" in fam and len(fam["square"]) >= 2, "need S3a (3x3) + S5a (5x5)"
    # EDGE family: E3a-d (3x3) + E5a-d (5x5) = >=8 edge predictors.
    assert "edge" in fam and len(fam["edge"]) >= 8, "need E3a-d (3x3) + E5a-d (5x5) edge predictors"
    # Honest Laplacian-4 (NOT mislabelled as SPAM).
    assert "lap4" in fam and len(fam["lap4"]) == 1
    # No rejected denoising family labels leak in.
    for forbidden in ("lap8", "log", "kv", "cubic"):
        assert forbidden not in fam, "rejected denoising family %s must be ABSENT" % forbidden


def test_srm_bank_dc_suppressed():
    """EVERY LINEAR SRM kernel sums to ~0 (DC-suppressed): a constant image yields ~0 on
    every filter. This is the defining high-pass property (so a flat image cannot leak DC into
    the conv blocks)."""
    bank = ganfp_net.highpass_bank()  # (N,1,5,5)
    n = bank.shape[0]
    sums = np.array([float(bank[i, 0].sum()) for i in range(n)])
    assert np.all(np.abs(sums) < 1e-6), "every SRM kernel must sum to ~0; got %s" % sums
    # And the DC-suppression must hold under convolution of a constant image (replicate-pad
    # so borders stay constant). The CNN uses zero-pad=2; the border effect there is absorbed
    # by BatchNorm, but the per-filter kernel invariant is exact under replicate padding.
    ones = np.ones((16, 16), dtype=np.float32)
    pad = np.pad(ones, 2, mode="edge")
    for i in range(n):
        k = bank[i, 0]
        out = np.zeros_like(ones)
        for a in range(ones.shape[0]):
            for b in range(ones.shape[1]):
                out[a, b] = float(np.sum(pad[a:a + 5, b:b + 5] * k))
        assert float(np.abs(out).max()) < 1e-5, \
            "filter %d: constant image must yield ~0 residual (max=%g)" % (i, float(np.abs(out).max()))


def test_srm_bank_no_per_kernel_l2_norm():
    """Per-kernel L2 normalization is NOT applied (the paper does not prescribe it). Kernels
    keep their natural magnitudes: e.g. the 2-tap SPAM-1 kernel [1,-1] has L2 norm sqrt(2),
    NOT 1. The center-cell DC-zeroing (_norm0) is the only normalization applied."""
    fam = ganfp_net.SRM_FAMILY_INDEX
    bank = ganfp_net.highpass_bank()
    # A spam1 kernel embedded in 5x5: only two nonzero cells (+1, -1) -> L2 norm = sqrt(2).
    i = fam["spam1"][0]
    nrm = float(np.linalg.norm(bank[i, 0]))
    assert abs(nrm - np.sqrt(2.0)) < 1e-4, \
        "spam1 kernel must NOT be L2-normalized; expected ~sqrt(2), got %g" % nrm


def test_srm_bank_distinct():
    """The LINEAR SRM kernels are pairwise DISTINCT (no duplicate / collinear filters
    inflating the bank)."""
    bank = ganfp_net.highpass_bank()  # (N,1,5,5)
    n = bank.shape[0]
    flat = bank.reshape(n, -1)
    for i in range(n):
        for j in range(i + 1, n):
            a = flat[i]
            b = flat[j]
            assert not np.allclose(a, b), "filters %d and %d are identical" % (i, j)
            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            cos = float(np.abs(np.dot(a, b)) / (na * nb)) if na > 1e-12 and nb > 1e-12 else 0.0
            assert cos < 0.9999, "filters %d and %d are collinear (|cos|=%.6f)" % (i, j, cos)


def test_minmax_metadata_nonlinear():
    """The MINMAX branch is the paper's MOST discriminative NONLINEAR family. It is NOT a conv
    kernel: it is the pointwise min/max of >=2 LINEAR residual maps. The metadata declares
    source index pairs into the linear bank, each yielding 2 channels (min + max), so
    SRM_MINMAX_CHANNEL_COUNT == 2 * len(SRM_MINMAX_SOURCE_PAIRS) > 0, and the total front-end
    channel count = linear + minmax."""
    pairs = ganfp_net.SRM_MINMAX_SOURCE_PAIRS
    assert len(pairs) > 0, "MINMAX branch must reduce >=1 directional SPAM pair"
    n_lin = ganfp_net.SRM_FILTER_COUNT
    for (i, j) in pairs:
        assert i != j, "minmax source pair must be two DISTINCT linear channels"
        assert 0 <= i < n_lin and 0 <= j < n_lin, "minmax source index out of range"
    assert ganfp_net.SRM_MINMAX_CHANNEL_COUNT == 2 * len(pairs)
    assert ganfp_net.SRM_FRONTEND_CHANNEL_COUNT == n_lin + ganfp_net.SRM_MINMAX_CHANNEL_COUNT
    # The source pairs are genuine SPAM directional channels (the paper reduces directional
    # SPAM residuals), not arbitrary indices.
    fam = ganfp_net.SRM_FAMILY_INDEX
    spam_idx = set(fam.get("spam1", []) + fam.get("spam2", []))
    for (i, j) in pairs:
        assert i in spam_idx and j in spam_idx, \
            "minmax source channels must be SPAM directional residuals (got %d,%d)" % (i, j)


def test_luminance_shape_no_torch():
    """The numpy luminance helper returns float32 in [0,1], shape (H,H)."""
    rgb = (np.random.RandomState(0).rand(64, 64, 3) * 255).astype(np.uint8)
    img = Image.fromarray(rgb, "RGB")
    lum = ganfp_net.luminance_array(img, common_size=64)
    assert lum.dtype == np.float32
    assert lum.shape == (64, 64)
    assert float(lum.min()) >= 0.0 - 1e-6
    assert float(lum.max()) <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Pure-Python (sklearn OK in CI per requirements.txt): FingerprintStandardizer / pipeline
# ---------------------------------------------------------------------------
def test_fingerprint_standardizer_train_only():
    """Fit on a train matrix large enough that PCA is not rank-clamped; transformed train is
    ~0 mean, transformed val is NOT standardized (leakage guard)."""
    rng = np.random.RandomState(0)
    X_train = rng.randn(200, 2048).astype(np.float32) * 3.0 + 1.0
    X_val = rng.randn(50, 2048).astype(np.float32) * 5.0 - 2.0
    std = ganfp.FingerprintStandardizer(pca_components=64)
    std.fit(X_train)
    Xtr = std.transform(X_train)
    Xva = std.transform(X_val)
    assert Xtr.shape == (200, 64)
    assert Xva.shape == (50, 64)
    # Train PCA output should be ~0 mean (PCA centers); std is on the SCALED input so the
    # magnitude is not literally 1, but train is centered.
    assert abs(float(Xtr.mean())) < 1e-4
    # Leakage guard: val is transformed with TRAIN statistics, so it should NOT be zero-mean.
    assert abs(float(Xva.mean())) > 1e-3, "val must not be standardized (leakage guard)"


def test_build_pca_pipeline_in_dim():
    """build_pca_pipeline returns the correct in_dim (64 without, 96 with DCT fusion) and a
    deterministic transform. Uses enough rows that PCA is not rank-clamped."""
    rng = np.random.RandomState(1)
    X_train = rng.randn(200, 2048).astype(np.float32)
    std, in_dim = ganfp.build_pca_pipeline(X_train, pca_components=64, dct_fuse=False)
    assert in_dim == 64
    t1 = std.transform(X_train)
    t2 = std.transform(X_train)
    assert np.allclose(t1, t2), "transform must be deterministic"

    D_train = rng.randn(200, 64).astype(np.float32)
    std2, in_dim2 = ganfp.build_pca_pipeline(
        X_train, pca_components=64, dct_fuse=True, dct_components=32, dct_train=D_train)
    assert in_dim2 == 96
    t = std2.transform(X_train, D_train)
    assert t.shape == (200, 96)


def test_pca_rank_clamp_small_train():
    """On a small train set PCA n_components is clamped to n_samples-1 (rank-safe)."""
    rng = np.random.RandomState(2)
    X_small = rng.randn(15, 2048).astype(np.float32)
    std, in_dim = ganfp.build_pca_pipeline(X_small, pca_components=64, dct_fuse=False)
    assert in_dim == 14, "PCA must clamp to n_samples-1 on a rank-deficient train set"
    assert std.transform(X_small).shape == (15, 14)


# ---------------------------------------------------------------------------
# torch-gated: CNN forward / param budget / one step / Dataset
# ---------------------------------------------------------------------------
def test_cnn_forward_shape():
    torch = pytest.importorskip("torch")
    model = ganfp_net.GANFpCNN(num_classes=7, input_size=256, channels=(32, 64, 128))
    out = model.model(torch.zeros(2, 1, 256, 256))
    assert tuple(out.shape) == (2, 7)


def test_cnn_frontend_channels_and_frozen():
    """The frozen LINEAR front-end produces SRM_FILTER_COUNT channels; the first VGG conv
    receives SRM_FRONTEND_CHANNEL_COUNT = linear + minmax channels; the whole front-end is
    frozen (requires_grad=False) and excluded from the trainable parameters."""
    torch = pytest.importorskip("torch")
    model = ganfp_net.GANFpCNN(num_classes=3, input_size=64, channels=(8, 16))
    net = model.model
    # Linear conv: 1 -> SRM_FILTER_COUNT, frozen.
    assert net.highpass.in_channels == 1
    assert net.highpass.out_channels == ganfp_net.SRM_FILTER_COUNT
    assert not net.highpass.weight.requires_grad
    # First VGG conv input = linear + nonlinear minmax channels.
    first_conv = net.blocks[0][0]
    assert first_conv.in_channels == ganfp_net.SRM_FRONTEND_CHANNEL_COUNT
    assert first_conv.in_channels == ganfp_net.SRM_FILTER_COUNT + ganfp_net.SRM_MINMAX_CHANNEL_COUNT
    # Frozen front-end excluded from trainable params.
    trainable = model.trainable_parameters()
    assert all(p is not net.highpass.weight for p in trainable)


def test_cnn_minmax_branch_function_of_linear():
    """The NONLINEAR MINMAX channels are a function of the linear directional SPAM residual
    maps: perturbing an input pixel changes the minmax channels (so they are not dead weight),
    they are differentiable (gradient flows to the input), and a CONSTANT image yields ~0 on
    every minmax channel (min/max of ~0 residuals is ~0)."""
    torch = pytest.importorskip("torch")
    model = ganfp_net.GANFpCNN(num_classes=3, input_size=32, channels=(8, 16))
    net = model.model

    # 1) Constant image -> ~0 linear residuals -> ~0 minmax channels IN THE INTERIOR. The
    #    CNN zero-pads (padding=2), so a constant image is NOT constant at the border and the
    #    [1,-1] SPAM kernels ring there (+/-1) -- that border effect is absorbed downstream
    #    by BatchNorm (documented). The DC-suppression invariant holds exactly in the interior
    #    (border margin = kernel half-extent), so check it there.
    ones = torch.ones(1, 1, 32, 32)
    b = 4  # interior margin (5x5 kernel half-extent, conservative)
    with torch.no_grad():
        resid = net.highpass(ones)
        mm = []
        for (i, j) in ganfp_net.SRM_MINMAX_SOURCE_PAIRS:
            ri = resid[:, i:i + 1]; rj = resid[:, j:j + 1]
            mm += [torch.minimum(ri, rj), torch.maximum(ri, rj)]
        mm = torch.cat(mm, dim=1)
    interior = mm[:, :, b:-b, b:-b]
    assert float(interior.abs().max()) < 1e-4, \
        "constant image must give ~0 minmax channels in the interior (border absorbed by BN)"

    # 2) Minmax channels respond to a perturbation (they depend on the linear residuals).
    xa = torch.randn(1, 1, 16, 16)
    xb = xa.clone()
    xb[0, 0, 5, 5] += 1.0
    with torch.no_grad():
        ra = net.highpass(xa); rb = net.highpass(xb)
        mma, mmb = [], []
        for (i, j) in ganfp_net.SRM_MINMAX_SOURCE_PAIRS:
            mma += [torch.minimum(ra[:, i:i + 1], ra[:, j:j + 1]),
                    torch.maximum(ra[:, i:i + 1], ra[:, j:j + 1])]
            mmb += [torch.minimum(rb[:, i:i + 1], rb[:, j:j + 1]),
                    torch.maximum(rb[:, i:i + 1], rb[:, j:j + 1])]
        mma = torch.cat(mma, dim=1); mmb = torch.cat(mmb, dim=1)
    assert float((mma - mmb).abs().max()) > 0.0, "minmax must respond to a linear perturbation"

    # 3) Differentiable: gradient flows through the minmax ops to the input.
    x = torch.randn(1, 1, 16, 16, requires_grad=True)
    out = net(x)
    out.sum().backward()
    assert x.grad is not None and float(x.grad.abs().sum()) > 0.0, "minmax must be differentiable"


def test_cnn_param_budget_small():
    """The compact [16,32,64] config stays under ~100k trainable params (~82k)."""
    pytest.importorskip("torch")
    model = ganfp_net.GANFpCNN(num_classes=10, input_size=256, channels=(16, 32, 64))
    n = model.param_count()
    assert n < 100_000, "compact CNN must stay under ~100k trainable params, got %d" % n


def test_cnn_param_budget_bigger():
    """The bumped [32,64,128] config is ~4x the compact head (~330k) -- the bigger CNN."""
    pytest.importorskip("torch")
    small = ganfp_net.GANFpCNN(num_classes=10, input_size=256, channels=(16, 32, 64)).param_count()
    big = ganfp_net.GANFpCNN(num_classes=10, input_size=256, channels=(32, 64, 128)).param_count()
    assert big > small, "bigger CNN must have MORE params than the compact one"
    assert 200_000 < big < 600_000, "bigger CNN ~330k params; got %d" % big
    assert big > 3.0 * small, "bigger CNN should be ~4x the compact head (%d vs %d)" % (big, small)


def test_cnn_one_step():
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    model = ganfp_net.GANFpCNN(num_classes=3, input_size=64, channels=(8, 16))
    opt = torch.optim.Adam(model.trainable_parameters(), lr=1e-2)
    crit = nn.CrossEntropyLoss()
    x = torch.randn(4, 1, 64, 64)
    y = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    # Run several steps; assert no NaN/Inf and that loss decreases over the run (a single
    # Adam step on a frozen-front-end micro-batch can transiently rise due to BN stats, so
    # the contract is "trains without error and trends down", per the spec).
    first = None
    last = None
    model.model.train()
    for _ in range(20):
        opt.zero_grad()
        logits = model(x)
        loss = crit(logits, y)
        loss.backward()
        opt.step()
        v = float(loss.item())
        assert not np.isnan(v) and not np.isinf(v), "loss must stay finite"
        if first is None:
            first = v
        last = v
    assert last < first, "training should reduce loss over 20 steps (got %s -> %s)" % (first, last)


def test_cnn_highpass_frozen():
    torch = pytest.importorskip("torch")
    model = ganfp_net.GANFpCNN(num_classes=3, input_size=64, channels=(8, 16))
    # The frozen front-end conv must have requires_grad=False and be EXCLUDED from trainable.
    assert not model.model.highpass.weight.requires_grad
    trainable = model.trainable_parameters()
    assert all(p is not model.model.highpass.weight for p in trainable)


def test_dataset_yields_tensor(tmp_path):
    torch = pytest.importorskip("torch")
    arr = (np.random.RandomState(0).rand(64, 64, 3) * 255).astype(np.uint8)
    p = tmp_path / "a.png"
    Image.fromarray(arr, "RGB").save(str(p))
    ds = ganfp_net.GANFpDataset([str(p)], [2], common_size=256)
    assert len(ds) == 1
    x, y = ds[0]
    assert torch.is_tensor(x)
    assert tuple(x.shape) == (1, 256, 256)
    assert x.dtype == torch.float32
    assert y.dtype == torch.long
    assert int(y) == 2


# ---------------------------------------------------------------------------
# Pure-Python (no torch): GAN-only attribution slice (lib.metrics.attribution_slice)
# ---------------------------------------------------------------------------
# Mirror the constants used by scripts/benchmark_attribution.py so this test pins the exact
# exclusion set the headline relies on.
_GAN_CLASSES = ["StyleGAN3-FFHQ", "PGGAN-v1", "PGGAN-v2", "StarGAN", "FaceApp"]
_DIFFUSION_CLASSES = ["SD1.5", "FLUX"]
_REAL_CLASSES = ["London-DB", "FFHQ", "CelebA"]


def test_ganonly_slice_excludes_diffusion():
    """The GAN-only slice scores GAN classes + reals ONLY: a diffusion true-label is excluded
    from the scored rows, and a diffusion PREDICTION (on a kept row) is folded to the synthetic
    'diffusion_mismatch' bucket so it counts as wrong without adding diffusion to the per-class
    report. Reals + GANs are scored normally."""
    classes = _GAN_CLASSES + _DIFFUSION_CLASSES + _REAL_CLASSES
    # Ground truth: one of each. Predictions: correct on GAN+real, a GAN predicted as FLUX
    # (should fold to diffusion_mismatch -> wrong), and the two diffusion rows (excluded).
    y_true = ["StyleGAN3-FFHQ", "PGGAN-v1", "PGGAN-v2", "StarGAN", "FaceApp",
              "SD1.5", "FLUX", "London-DB", "FFHQ", "CelebA"]
    y_pred = ["StyleGAN3-FFHQ", "PGGAN-v1", "PGGAN-v2", "FLUX", "FaceApp",
              "StyleGAN3-FFHQ", "London-DB", "London-DB", "FFHQ", "CelebA"]

    keep = sorted(set(_GAN_CLASSES) | set(_REAL_CLASSES))
    res = metrics.attribution_slice(y_true, y_pred, classes, keep,
                                    other_label="diffusion_mismatch")

    # Labels are the keep set + the fold bucket; diffusion generators must NOT appear.
    labels = res["labels"]
    for d in _DIFFUSION_CLASSES:
        assert d not in labels, "diffusion class %s must be EXCLUDED from the GAN-only slice" % d
    assert "diffusion_mismatch" in labels
    # The two diffusion true-rows are excluded -> scored n drops by 2.
    assert res["n"] == len(y_true) - len(_DIFFUSION_CLASSES)
    # Of the 8 scored rows, 7 are correctly attributed in the kept set; StarGAN was predicted
    # as FLUX (diffusion) -> folded to diffusion_mismatch -> wrong. So top-1 = 7/8.
    assert res["top1_accuracy"] == pytest.approx(7.0 / 8.0)
    # Per-class recall on the GAN+real classes is present; StarGAN recall is 0 (its one row
    # was predicted as FLUX -> folded -> wrong).
    assert res["per_class"]["StarGAN"]["support"] == 1
    assert res["per_class"]["StarGAN"]["recall"] == 0.0
    assert res["per_class"]["London-DB"]["recall"] == 1.0


def test_attribution_slice_no_keep():
    """If no true label is in the keep set, the slice scores zero rows (no crash)."""
    res = metrics.attribution_slice(["SD1.5", "FLUX"], ["SD1.5", "FLUX"],
                                    ["SD1.5", "FLUX", "London-DB"], ["London-DB"])
    assert res["n"] == 0
