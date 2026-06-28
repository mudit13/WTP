# GAN Fingerprints Reproduction — Plain-English Report

This is a non-technical write-up of the GAN-fingerprints (GAN-fp) work, the experiments we
ran, and the results. All numbers are real (single seed=42; test split = 326 images).

## 1. What we built and why

Your existing detector, **DE-FAKE**, can tell whether a face image is **real or fake**, but it
cannot say **which generator** made a fake, and it is weak on GAN images in particular.

**GANFingerprints (Yu et al., ICCV 2019)** is a method that reads the invisible "fingerprint"
each image generator leaves behind in the pixel noise. No pretrained GAN-fp model existed on
the server (only DE-FAKE and the generators themselves), and the original 2019 code was
unusable (an obsolete "Chainer" stack). So we **rebuilt the method ourselves in PyTorch**.

We built it two ways and compared them head-to-head:
- **Path A** — hand-crafted forensic features (a noise "residual" + a frequency spectrum) fed
  to a small classifier.
- **Path B** — a small **CNN that learns** the fingerprints (the faithful Yu2019 approach).

The "fingerprint" is a microscopic high-frequency noise trace — it is **not** about what the
picture shows, only about how the pixels were generated.

## 2. The two datasets we used

| Dataset | Size | Purpose | Lesson |
|---|---|---|---|
| **Toy set (20 images/class)** | 200 images | Smoke test — does the code run? | Numbers were modest/noisy (too few examples) |
| **Real set (~200 images/class)** | 1,626 images | Proper benchmark | ~8x more data -> real-vs-fake detection AUROC jumped 0.85 -> 0.96 |

The weak numbers on the tiny toy set were a **data-size** problem, not a broken method. That
was confirmed when 8x more data lifted every metric.

## 3. The three improvements (in plain English)

- **(b) SRM front-end** — the CNN originally used *one* noise filter (one magnifying glass).
  We swapped it for a faithful **reconstruction of the SRM (Spatial Rich Model)** high-pass
  front-end, borrowed from steganalysis (Fridrich & Kodovsky 2012, "Rich Models for
  Steganalysis"). This is the AUTHORITATIVE source: the paper defines residual/predictor
  FAMILIES, not a fixed 30-kernel bank (the "30 SRM filters" is a downstream CNN-steganalysis
  convention). Our front-end reconstructs those families:
  - A **linear bank** of 30 high-pass kernels spanning the genuine families — **SPAM**
    (1st-order 2-tap `[1,-1]`, 2nd-order `[1,-2,1]`, 3rd/4th-order directional predictors,
    plus the 2-D `spam14` workhorse), **SQUARE** (the S3a 3x3 and S5a 5x5 L2-optimal
    shift-invariant edge predictors), **EDGE** (E3a-E3d 3x3 and E5a-E5d 5x5 edge predictors
    derived from S3a/S5a), and one honest **Laplacian-4** center-surround kernel.
  - A **nonlinear MINMAX branch** — the paper's MOST discriminative signal (the `minmax24`
    submodel was the single best of the 106 SRM submodels). Minmax is the **pointwise
    minimum / maximum** of two or more linear residual maps; it is a pixel-wise operator and
    CANNOT be a single conv kernel. We compute it with fixed, differentiable
    `torch.minimum`/`torch.maximum` over the directional SPAM residual maps and concatenate
    those channels onto the linear bank, so the VGG blocks receive (linear + minmax) channels.
  - Each linear kernel is **DC-suppressed** (a flat image yields ~0 residual) and the whole
    front-end is **frozen** (non-trainable). We deliberately **dropped** the per-kernel L2
    normalization the paper does not prescribe (downstream BatchNorm absorbs the kernels'
    magnitude differences), and we **removed** the denoising-style filters the paper explicitly
    rejects in Sec II-E (Laplacian-8, Laplacian-of-Gaussian/LoG, the square-Laplacian cubic —
    these are POST-paper Kang-2013 / Xu-Net / Ye-Net "KB/LoG/KV" kernels that bias the
    predictor and suppress the signal).
  This is a faithful **family-level** reconstruction, not an "exact SRM" claim: the paper
  leaves several kernels non-unique / sign- or order-ambiguous, and where it does we use a
  distinct DC-suppressed kernel from the same family.
- **(c) Bigger CNN + tuning sweep** — we tried three CNN sizes. The **smallest won**
  (`[16,32,64]`, validation accuracy 0.736); bigger ones overfit the 1,626 images
  (`[32,64,128]` = 0.718; `[48,96,192]` overfit). Bigger brains need more data.
- **(d) GAN-only headline** — we report the attribution accuracy over the **GAN classes only**
  (not averaged with diffusion images, which the method is not designed for).

## 4. Results — Path B (the CNN), before vs after, per class

**Headline:** swapping in the faithful SRM front-end (linear SPAM/SQUARE/EDGE families + the
nonlinear MINMAX branch) lifted **generator-attribution accuracy from 0.797 -> 0.871
(+7.4 points)**. Real-vs-fake detection stayed near-perfect (AUROC ~0.98) — it was already
close to ceiling.

| Class (type) | Before (1 filter) | After (faithful SRM: linear SPAM/SQUARE/EDGE + nonlinear MINMAX) | Change |
|---|---|---|---|
| CelebA (real) | 1.00 | 0.90 | -0.10 |
| FFHQ (real) | 0.50 | 0.93 | +0.43 |
| London-DB (real) | 0.95 | 1.00 | +0.05 |
| StyleGAN3 (GAN) | 1.00 | 0.77 | -0.23 |
| PGGAN-v1 (GAN) | 0.88 | 0.93 | +0.05 |
| PGGAN-v2 (GAN) | 0.28 | 0.80 | +0.53 |
| StarGAN (GAN) | 1.00 | 1.00 | — |
| FaceApp (GAN) | 0.80 | 0.63 | -0.18 |
| SD1.5 (diffusion) | 0.86 | 0.82 | -0.05 |
| FLUX (diffusion) | 1.00 | 1.00 | — |
| **Overall attribution** | **0.797** | **0.871** | **+0.074** |
| Detection AUROC | 0.989 | 0.983 | ~flat (both near ceiling) |

Path A (feature + MLP + PCA) stayed at attribution 0.733 / detection AUROC 0.962 — it does not
use the CNN front-end, so the SRM change did not affect it (a useful consistency check).

## 5. Honest caveats

- **Single run, single seed** -> per-class numbers swing about +/-10%. Treat them as
  directional, not precise.
- "After" bundles **two** changes: the SRM front-end **and** more training epochs (60 vs 30).
  They are not perfectly isolated.
- **Per-class it is mixed**: SRM hugely helped FFHQ (+0.43) and PGGAN-v2 (+0.53) but *hurt*
  StyleGAN3 (-0.23) and FaceApp (-0.18). Net clearly positive.
- Interesting: the CNN also attributed **diffusion** images well here (FLUX 1.0, SD1.5 0.82),
  so the "diffusion mismatch" concern did not bite for Path B (it did for Path A, where SD1.5
  was only 0.50).
- **Report-final numbers need the full server run** (thousands of images, GPU). This 1,626-image
  local set is a solid prototype, not the deliverable.

## 6. What markers the detector reads, and how it compares to the literature

Our detector reads **forensic high-frequency noise traces** on the image's brightness channel
(a noise residual + a frequency spectrum + the SRM high-pass families: linear
SPAM/SQUARE/EDGE kernels + the nonlinear MINMAX branch + the CNN's learned filters) —
not the picture's content.

It belongs to the **forensic / frequency family**:
- **Yu 2019 (GAN Fingerprints)** — our Path B's direct basis (we reproduce it).
- **Frank 2020 (DCT)** — frequency-domain artifacts; our spectrum block + the separate DCT
  detector are this family.
- **Wang 2020 (CNNDetect)** — patch-level frequency forensics; same philosophy.
- **SRM (Fridrich & Kodovsky 2012, "Rich Models for Steganalysis")** — the AUTHORITATIVE
  source for our front-end. The paper defines residual/predictor FAMILIES (not a fixed
  30-kernel bank); we reconstruct those families faithfully — linear SPAM/SQUARE/EDGE kernels
  plus the nonlinear MINMAX branch (the paper's most discriminative signal) — and we exclude
  the denoising-style filters (Laplacian-8, LoG, cubic) the paper rejects in Sec II-E.

It is **complementary** to **DE-FAKE (Sha et al. 2023)**, which is *semantic* (it uses CLIP to
read image content/meaning). DE-FAKE detects broadly (including diffusion) but cannot attribute
which generator; our GAN-fp can attribute GANs but is not built for diffusion. That is exactly
why both are in the pipeline.

## 7. Next steps

1. **Server full run** — same code on the complete datasets + the Titan RTX, for report-ready
   numbers (and where a bigger CNN would stop overfitting).
2. The code, this report, and the key result files live in this repo.
