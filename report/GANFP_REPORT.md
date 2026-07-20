# GAN Fingerprints Reproduction — Plain-English Report

> **Appendix-only historical workstream.** GAN-fp is not part of the professor-facing core
> pipeline. The primary result is DCT detection followed by eight-way DE-FAKE attribution.

A non-technical writeup of the GAN-fingerprints (GAN-fp) work: the problem, why we had to
build it ourselves, what we built, the results, and the honest caveats.

## 1. The problem

The brief was: *"Run GANFingerprints on all datasets, particularly StyleGAN3 and DFFD — the
GAN images that DE-FAKE can't attribute properly."*

**DE-FAKE** (the project's existing detector) can say whether a face image is **real or
fake**, but it **cannot say which generator** made a fake, and it is weak on GAN images. It
works from **semantics** (it uses CLIP, which understands image *content*).

**GANFingerprints (Yu et al., ICCV 2019)** is a different idea: every image generator leaves
an invisible **"fingerprint"** in the microscopic pixel noise, and a model can learn to read
that fingerprint to (a) detect real-vs-fake and (b) attribute the source generator. This is
exactly the signal DE-FAKE misses — so the two methods are **complementary**, not redundant.

## 2. Why we had to build it ourselves

We verified on the GPU server that **no pretrained GAN-fp weights exist** (`models/` holds
only DE-FAKE checkpoints + the generators), and the legacy `/workspace/GANFingerprints` repo
is **unusable** (an obsolete Chainer/CUDA-10 stack, built for *other* GANs, with no bundled
weights — and DFFD's leftover `Fingerprints_*` dirs were empty, confirming a prior attempt
left nothing). Per `REVIEW_SAFEGUARDS.md` / `PROJECT_LOG §5`, we therefore **reproduced the
method in PyTorch**.

## 3. What we built

Two attribution paths, run on the **same** seeded train/val/test split and compared
head-to-head (`scripts/benchmark_attribution.py`):

- **Path A — forensic features + MLP:** a **noise residual** (image minus a blurred estimate)
  and the **FFT frequency spectrum**, compressed with train-only PCA, fed to a small MLP.
- **Path B — Yu2019-inspired CNN:** a fixed **SRM high-pass front-end** (see §4) + VGG conv
  blocks that **learn** the fingerprints. Yu2019 learns the fingerprint with a plain RGB CNN;
  we keep that idea and add the SRM front-end, so the *method* is **inspired by** Yu2019 (not a
  byte-faithful port). The **SRM front-end**, separately, *is* a faithful family-level
  reconstruction of Fridrich-Kodovsky 2012 (see §4).

Both output a per-generator prediction; real-vs-fake is a fold of that (any real class →
"real"). The CNN's learned filters become the model-specific "fingerprints."

## 4. The SRM front-end (and how we made it faithful)

The CNN's first layer is a **frozen** bank of high-pass filters that strips away the visible
image and keeps only the forensic noise. We cross-checked this bank against the authoritative
source — **Fridrich & Kodovsky 2012, "Rich Models for Steganalysis"** — and rebuilt it to be
faithful at the family level:

- **Linear kernels** from the paper's genuine families: **SPAM** (1st-order `[1,-1]`,
  2nd/3rd/4th-order directional predictors, and the 2-D `spam14` workhorse), **SQUARE**
  (`S3a` 3x3, `S5a` 5x5 L2-optimal predictors), and **EDGE** (`E3a-d`, `E5a-d` edge
  predictors).
- A **nonlinear MINMAX branch** — the paper's *most discriminative* signal. Minmax is the
  pointwise **minimum/maximum** of two or more linear residual maps; it cannot be a single
  conv kernel, so we compute it with `torch.minimum`/`torch.maximum` over the directional
  SPAM residuals and concatenate those channels.
- We **removed** the denoising-style filters the paper explicitly rejects in Sec II-E
  (Laplacian-8, Laplacian-of-Gaussian, the square-Laplacian cubic — these are *post-paper*
  kernels that bias the predictor).
- We **dropped** the per-kernel L2 normalization the paper does not prescribe (downstream
  BatchNorm absorbs the magnitude differences), keeping the center-cell DC-suppression
  (a flat image yields ~0 residual).

Net: a **30 linear + 12 minmax = 42-channel frozen front-end**, honestly scoped as a
"family-level reconstruction" (the paper leaves several kernels sign/order-ambiguous).

## 5. The datasets

| Dataset | Size | Purpose | Lesson |
|---|---|---|---|
| **Toy set (20 images/class)** | 200 images | Smoke test — does the code run? | Numbers modest/noisy (too few examples) |
| **Real set (~200 images/class)** | 1,626 images | Proper benchmark | ~8x more data -> real-vs-fake detection AUROC jumped ~0.85 -> ~0.96 |

The weak numbers on the toy set were a **data-size** problem, not a broken method — confirmed
when 8x more data lifted every metric.

## 6. Results (and an honest caveat about variance)

Latest GPU run on the 1,626-image set (faithful SRM, `[16,32,64]` VGG, 60 epochs):

| Path | Attribution top1 | Detection AUROC |
|---|---|---|
| A (features + MLP + PCA) | 0.733 | 0.962 |
| **B (faithful SRM CNN)** | **0.819** | **0.968** |

Per-class recall (Path B): CelebA 1.00 · FFHQ 0.53 · FLUX 1.00 · FaceApp 0.97 · London-DB
1.00 · PGGAN-v1 0.75 · PGGAN-v2 0.68 · SD1.5 0.91 · StarGAN 0.80 · StyleGAN3 0.73. Slices:
**GAN-only** top1 0.798; **diffusion-mismatch** 0.955 (the CNN cleanly separates the two
diffusion models, FLUX 1.00 / SD1.5 1.00).

**Caveat — the prototype is variance-limited.** GPU training is not fully deterministic, and
the "best validation" checkpoint is chosen from only ~163 val images, so the selected
checkpoint — and therefore the test numbers — swing run-to-run. Observed attribution band
across runs: **~0.82-0.87 (+/-0.05)**. That variance is *larger* than the old-vs-faithful
difference, so on this prototype we **cannot** declare one front-end better than the other.
Detection AUROC is steadier (~0.97-0.99), and per-class numbers are volatile for the same
reason. **Report-grade numbers require the full server run** (thousands of images, larger
val/test) — the same code scales there unchanged.

## 7. How it compares to the literature

Our detector reads **forensic high-frequency noise traces** (residual + spectrum + the SRM
families + learned CNN filters) — **not** image content. It belongs to the
**forensic/frequency family**: **Yu 2019** (our basis), **Frank 2020** (DCT), **Wang 2020**
(CNNDetect), **SRM / Fridrich-Kodovsky 2012** (our front-end). It is **complementary** to
**DE-FAKE (Sha 2023)**, which is *semantic* (CLIP). DE-FAKE detects broadly (including
diffusion) but cannot attribute which generator; ours attributes GANs but is not built for
diffusion — which is exactly why both belong in the pipeline.

## 8. Status and next steps

- The faithful SRM reproduction is complete, tested (**51 tests passing**), and verified
  against the paper. It is on the `ws4-ganfp` branch, PR'd into the team repo.
- **Remaining:** the **full server run** (complete datasets + Titan RTX) for stable,
  report-grade numbers — where a bigger CNN also stops overfitting. The local 1,626-image
  prototype is a validated proof, not the deliverable.
- Optional follow-ups: a leave-one-generator-out (LOGO) out-of-set test, and the
  DE-FAKE / DCT cross-comparison on the same splits.
