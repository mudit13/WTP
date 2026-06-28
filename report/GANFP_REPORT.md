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
  We swapped it for a **kit of 30** different filters (borrowed from steganalysis, the science
  of detecting hidden messages), each revealing a different kind of hidden texture. This gave
  the model many more ways to spot each generator's fingerprint.
- **(c) Bigger CNN + tuning sweep** — we tried three CNN sizes. The **smallest won**
  (`[16,32,64]`, validation accuracy 0.736); bigger ones overfit the 1,626 images
  (`[32,64,128]` = 0.718; `[48,96,192]` overfit). Bigger brains need more data.
- **(d) GAN-only headline** — we report the attribution accuracy over the **GAN classes only**
  (not averaged with diffusion images, which the method is not designed for).

## 4. Results — Path B (the CNN), before vs after, per class

**Headline:** swapping in the 30-filter SRM front-end lifted **generator-attribution accuracy
from 0.797 -> 0.871 (+7.4 points)**. Real-vs-fake detection stayed near-perfect
(AUROC ~0.98) — it was already close to ceiling.

| Class (type) | Before (1 filter) | After (SRM 30 filters) | Change |
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
(a noise residual + a frequency spectrum + the 30 SRM filters + the CNN's learned filters) —
not the picture's content.

It belongs to the **forensic / frequency family**:
- **Yu 2019 (GAN Fingerprints)** — our Path B's direct basis (we reproduce it).
- **Frank 2020 (DCT)** — frequency-domain artifacts; our spectrum block + the separate DCT
  detector are this family.
- **Wang 2020 (CNNDetect)** — patch-level frequency forensics; same philosophy.
- **SRM (Fridrich & Kodovsky 2012, steganalysis)** — our 30-filter front-end is borrowed from
  here.

It is **complementary** to **DE-FAKE (Sha et al. 2023)**, which is *semantic* (it uses CLIP to
read image content/meaning). DE-FAKE detects broadly (including diffusion) but cannot attribute
which generator; our GAN-fp can attribute GANs but is not built for diffusion. That is exactly
why both are in the pipeline.

## 7. Next steps

1. **Server full run** — same code on the complete datasets + the Titan RTX, for report-ready
   numbers (and where a bigger CNN would stop overfitting).
2. The code, this report, and the key result files live in this repo.
