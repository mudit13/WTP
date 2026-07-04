# Project log - changes and the reasons behind them

Topic 8: AI Image Detection & Attribution. Code is authored locally and executed inside the
GPU container; the repo root maps to the container project root (`/pitsec_sose26_topic8`).
This log explains WHAT changed and WHY, so anyone (team or examiner) can follow the reasoning.
For how each change maps to the interim "GOLD" review, see `docs/GOLD_ALIGNMENT.md`.

---

## 1. Unified the repository (one repo, server-importable)

**What:** Merged the analysis/experiments layer with the team's existing server code
(generation + DE-FAKE inference) into a single tree: shared library in `scripts/lib/`, all
entry points in `scripts/`, vendored DE-FAKE in `De-Fake-patched/`, docs in `docs/`. Flattened
the nested `server/` clone, dropped the empty `main.py`, and folded the separate
`update_master_index_dffd.py` into the config-driven `build_master_index.py`.

**Why:** The server layout was flat and brittle - loose scripts at the root, hardcoded
absolute paths everywhere, a vendored dependency pinned by an absolute `sys.path`, and a
`.gitignore` with unresolved merge-conflict markers. A single organized repo can be pushed to
GitHub and `git pull`-ed onto the server, keeping future work maintainable.

## 2. Centralized the data schema (`scripts/lib/schema.py`)

**What:** One module defines the canonical column names of `master_metadata.csv`
(`filename, full_path, label, generator, category, source_dataset, width, height`) and the
DE-FAKE prediction columns (`defake_predict, prob_real, prob_fake, blip_caption`), plus
helpers `is_fake_label` / `is_fake_predict`.

**Why:** Every script previously risked drifting from the real CSV columns. A single source of
truth prevents silent column-name bugs and keeps the analysis layer compatible with the
team's existing pipeline outputs.

## 3. Config + path handling aligned to the real server

**What:** `configs/config.yaml` mirrors the actual dataset layout, generator names, and DFFD
sources; `configs/paths.example.env` lists the three generation venvs and the DE-FAKE
interpreter (`venv_sd15`). The moved DE-FAKE scripts (`run_defake_batch.py`,
`run_defake_dffd.py`, `merge_predictions.py`) now read their paths from the environment with
the original absolute values as defaults (behaviour unchanged if the env is not sourced).

**Why:** Removes hardcoded paths so the repo location is flexible and reproducible, while not
changing the logic of scripts the team already ran successfully.

## 4. .gitignore + privacy hygiene

**What:** Replaced the conflicted `.gitignore` with one canonical file that ignores datasets,
model weights, all three venvs, results/logs, and secrets. Replaced the supervisor's name in
all committed files with "the supervisor", and kept internal coordination notes in
`docs/OPEN_QUESTIONS.md` (git-ignored, local only).

**Why:** Keep large/non-relocatable artifacts and any internal/identifying content out of a
GitHub repo; venvs in particular must never be committed (they hardcode absolute paths).

## 5. Server inspection 

We inspected `/workspace`, `/share`, and `/pitsec_sose26_topic8` to resolve open questions
ourselves rather than asking blindly. Findings drove the decisions below.

- **DE-FAKE is binary-only.** `models/` has `clip_linear.pt` (binary real/fake head) +
  `finetune_clip.pt`; there is no pretrained multi-class attribution head. -> Attribution must
  come from our own fine-tuned head (RQ2 framed accordingly).
- **GANFingerprints repo = code only, deprecated stack** (Chainer/cupy/CUDA 10), built for its
  own GANs. -> We will REPRODUCE the fingerprint method ourselves in PyTorch on our generators
  (residual/spectrum fingerprints + a small learned classifier) instead of running the legacy
  code. Less risk, and it covers our actual generators.
- **GANDCTAnalysis = code only.** -> We train the DCT detector ourselves (as planned).
- **Plenty of data on the server.** DFFD has full train/val/test per generator (thousands of
  images) and `img_align_celeba` = 202,599 real faces. -> Attribution data is not a bottleneck;
  we can scale beyond the 100/generator originally sampled.

## 6. Fixed the real-class imbalance + narrowness (key scientific fix)

**What:** The index was 724 fake vs only 202 real, with just 2 real sources (London-DB 102 +
FFHQ 100), both aligned faces. `configs/config.yaml` now uses London-DB 102 (studio) + FFHQ
100->300 (Flickr) + CelebA 320 (web) ~= 722 real vs 724 fake, across three distinct capture
conditions. We report balanced metrics (AUROC, balanced accuracy).

**Why:** A 78/22 split lets a trivial "always fake" baseline score 78%, and a single narrow
real source risks the detector learning "studio portrait" instead of "real". Balancing and
diversifying the real class is required for a fair, defensible evaluation (GOLD concern #1).

## 7. Controlled the format/resolution confound (most important data finding)

**What:** Inspecting pixels showed format/size almost perfectly predict the label before any
model sees content: reals include JPEG (CelebA, London-DB) while every fake is PNG, and
resolutions separate classes (512=>fake, 1350/178=>real; only 299 has both). Decisions, now in
`configs/config.yaml`:
- `common_size: 256` (down from 512) - mostly downscales, avoiding the heavy upscaling
  artifacts that 512 would inject into CelebA (178) and DFFD (299).
- `augmentation: { jpeg_train: true, jpeg_quality_range: [30,100] }` - random JPEG compression
  applied to ALL classes during detector/attribution training.
- Plan to report RAW vs NORMALIZED/augmented to demonstrate the confound's effect.

**Why:** A frequency detector could score ~99% by learning compression + resolution rather
than generator traces - exactly the failure the GOLD review warned about. Re-encoding JPEG->PNG
does NOT remove baked-in JPEG artifacts, so uniform JPEG augmentation is needed so
"has-been-compressed" no longer leaks the label (Frank 2020 / Wang 2020).

**Code wiring (this is live, not just config):**
- `scripts/lib/image_ops.py`: `make_jpeg_augmenter(quality_range, seed)` - per-path
  deterministic random JPEG (reproducible, order-independent).
- `scripts/lib/clip_features.py`: `extract_features(..., augment=...)` applies it before CLIP.
- `scripts/lib/features_cache.py`: `build_features(..., jpeg_aug, jpeg_quality_range, seed)`;
  a cache built with a different `jpeg_aug` setting is not silently reused.
- `scripts/dct_extract_features.py`: `--jpeg_aug` (+ `--jpeg_qmin/qmax/seed`) for the DCT path.
- `scripts/finetune_defake_head.py` and `scripts/leave_one_generator_out.py`: `--jpeg_aug
  {auto,on,off}`, defaulting to `auto` = follow `config.augmentation.jpeg_train`.

## 8. Datasheet provenance captured

**What:** From the DFFD `readme.txt` (Dang et al., CVPR 2020): reals = FFHQ + CelebA; fakes =
FaceApp (FFHQ-derived), PGGAN, StarGAN, StyleGAN; all face-aligned (RetinaFace); license
CC BY-NC-SA 4.0 (cite sources). FaceApp is a manipulation, not pure synthesis.

**Why:** The GOLD review requires processing history per dataset so we can argue the detector
learns generator traces, not preprocessing artifacts. Recorded for `docs/DATASHEET_TEMPLATE.md`.

## 9. DE-FAKE detection results so far (pretrained binary head)

**What:** Scored the pretrained DE-FAKE detector with `score_defake_detection.py`.

- Baseline index (202 real = FFHQ100 + London-DB102 / 724 fake): balanced acc **0.586**, AUROC **0.710**.
- Balanced index (722 real = CelebA320 + FFHQ300 + London-DB102 / 724 fake): balanced acc
  **0.591**, AUROC **0.713**, fake recall **0.80**, real specificity **0.378**.
- Per real source (specificity): CelebA 27.5%, London-DB 12.7%, FFHQ 57.3%.
- Per fake (recall): SD1.5 100%, FLUX 99%, StarGAN 91%, PGGAN-v1 83%, PGGAN-v2 73%,
  FaceApp 70%, **StyleGAN3 46%** (blind spot).

**Why it matters:** AUROC is stable (~0.71) across both real-class compositions, so it is the
fair cross-run metric (AUPRC shifts with base rate). The low accuracy at the default 0.5
threshold reflects a **fake-biased, miscalibrated operating point** on out-of-distribution real
faces (domain shift: DE-FAKE's training reals are MSCOCO-style, not faces), NOT a London-DB
artifact - confirmed because the over-prediction holds across studio/web/Flickr reals. The
scorer now also reports the balanced-accuracy-optimal threshold (Youden's J) to separate
ranking quality from operating-point choice.

**Pipeline note:** the unified `master_metadata.csv` already contains DFFD rows, and
`run_defake_batch.py` processes every row, so detection inference is `run_defake_batch.py`
ALONE (no `run_defake_dffd.py` + `merge_predictions.py`, which would double-count DFFD).

## 9b. DE-FAKE multi-class attribution results (fine-tuned head, aspect variant)

**What:** Fine-tuned the CLIP+MLP attribution head (`finetune_defake_head.py`) on the
aspect-preserving variant over 6 classes (reals CelebA/FFHQ/London-DB + trained fakes
SD1.5/FLUX/StyleGAN3), evaluated with `eval_defake_attribution.py`,
`leave_one_generator_out.py`, and `out_of_set_analysis.py`. The four DFFD GANs
(FaceApp/PGGAN-v1/PGGAN-v2/StarGAN) were held out as genuinely UNSEEN and force-scored.

**In-set attribution (held-out test, n=210):**
- Controlled (JPEG-normalized): top-1 **94.8%**, balanced **94.5%**. Fake-only (eval, n=66):
  balanced **93.9%**.
- Per-class recall: FLUX 100%, SD1.5 100%, FFHQ 96.7%, London-DB 95%, CelebA 93.8%,
  **StyleGAN3 81.8%** (weakest; all 4 errors -> FFHQ).

**Format/JPEG confound, MEASURED (raw vs controlled, both on aspect geometry):**
- Raw (`--jpeg_aug off`): top-1 **96.2%** / balanced 95.9%. Controlled: 94.8% / 94.5%.
- Delta only **~1.4 pts** -> the head relies mostly on generator content, not compression/
  format. Directly answers Dennis: the confound is real but small. (scaled-vs-aspect geometry
  ablation + metadata-only classifier still pending.)

**Out-of-set (4 unseen GANs, n=400) - the closed-set limitation, quantified:**
- top-1 = 0 BY CONSTRUCTION (true class absent). Informative signal: **~98% (393/400) of unseen
  GAN images are forced onto a REAL class** (CelebA/FFHQ) at mean confidence 0.82.
- false-known rate: **0.96 @0.5**, 0.76 @0.7, **0.44 @0.9**. Entropy separates populations
  (in-set 0.19 vs out-of-set 0.47) -> entropy-based open-set rejection is a partial fix only.

**LOGO (retrain WITHOUT the target) - THE key finding (family asymmetry):**
- Unseen DIFFUSION (FLUX out, n=108): forced to the other diffusion SD1.5 **81.5%**; ~94% land
  on a FAKE class -> detection survives, misattribution is family-consistent.
- Unseen GAN (StyleGAN3 out, n=108): forced to FFHQ **85%** (97% to real classes) -> detection
  FAILS; StyleGAN3 collapses onto its FFHQ training source.

**Why it matters:** three independent measurements triangulate the same mechanism - face GANs
trained on real face datasets collapse onto the real manifold, while diffusion generalizes
within-family: binary detection (StyleGAN3 46% fake recall), in-set attribution (StyleGAN3->
FFHQ errors), and out-of-set/LOGO (unseen GANs -> real). Caveat: small per-fake-class test
support (~22 each); the out-of-set-> real result is partly confounded by real over-
representation and the DFFD GANs being face manipulations close to the real manifold.

## 9c. Confound-controlled binary detection + DCT cross-method (aspect variant)

**What:** Re-ran binary detection on the confound-controlled `index_aspect.csv` (same 722 real /
724 fake set as the balanced baseline) and DCT-SVM on the aspect variant.

**DE-FAKE, confound MEASURED (apples-to-apples, raw originals vs aspect):**
- AUROC **0.713 -> 0.674** (-0.04); balanced acc 0.591 -> 0.560 (0.641 at Youden's J); real
  specificity 0.378 -> 0.292; StyleGAN3 recall 0.46 -> 0.51.
- Reading: the raw format/geometry confound gave DE-FAKE only ~4 AUROC points, and removing it
  does NOT fix the weak face detection (DE-FAKE's reals are MSCOCO-style, not faces). Honest
  answer for the detector: confound present but small.

**DCT-SVM (Frank2020) on the controlled set beats DE-FAKE:** AUROC **0.777 / balanced 0.703** vs
DE-FAKE 0.674 / 0.560. Frequency artifacts generalize better than CLIP-semantic on normalized
data -> supports feature fusion (future work).

**Out-of-set: both methods fail.** DCT-SVM on held-out generators ~chance (balanced 0.57, AUROC
0.60), mirroring the DE-FAKE attribution collapse.

**Caveat / still pending:** the existing `dct_svm_raw`/`dct_svm_jpegaug` runs are on an older
926-image sample (NOT comparable to the 1446 aspect run) - a DCT run on `index_scaled`@1446 is
needed for a clean DCT confound delta. Also still pending: the metadata-only confound probe
(Dennis's direct question) and the attribution scaled-vs-aspect comparison.

## 10. GAN Fingerprints (Yu2019-inspired) reproduced in PyTorch

> **STATUS: PARKED (removed from `main`).** Sections 10-11 are the historical record of the
> GAN-fp exploration. Per the supervisor (DE-FAKE multi-class attribution takes priority), the
> GAN-fp code was removed from `main` and preserved on the `ganfp-integrated` branch. Re-add
> only if time allows after the DE-FAKE attribution deliverable.

**What:** Added a GAN-fp attribution path as a second method beside the CLIP/DE-FAKE head.
`scripts/lib/ganfp.py` extracts residual + FFT-spectrum fingerprint features (luminance,
downsampled to a fixed grid, L2-normalized); `scripts/train_ganfp.py` trains
`defake_head._MLPHead` on them (multi-class attribution + secondary binary detection, writes
`ganfp_metrics.json`/confusion matrix/per-image/`ganfp_head.pt`); `scripts/run_ganfp_infer.py`
emits a per-image CSV the existing `eval_defake_attribution.py` consumes unchanged. A `ganfp:`
block was added to `configs/config.yaml`; `tests/test_ganfp.py` covers the feature math (no
torch). `run_ganfp.py` is now just the weight-discovery/scope-note helper.

**Why:** Verified on the server that no pretrained GAN-fp weights exist (`models/` = DE-FAKE +
generators only) and the legacy `/workspace/GANFingerprints` repo is Chainer/cupy (dead) built
for other GANs (ProGAN/SNGAN/MMDGAN/CramerGAN). DE-FAKE (CLIP/semantic) cannot attribute GAN
images, so we reproduce the fingerprint method ourselves (it targets the GAN-specific traces).
Expected to attribute GAN sources (StyleGAN3, DFFD families) and to category-mismatch on
diffusion (SD1.5, FLUX) - documented behavior, not a failure.

**Local prototype:** built + validated locally on CPU against a downloaded ~20-image/generator
sample (`ganfp_sample/`, gitignored) across StyleGAN3 + DFFD families + reals + diffusion. The
full server-GPU run over the complete datasets is the remaining step (item A).

## 11. GANFingerprints rebuild: CNN path, train-only PCA, head-to-head benchmark (2026-06-28)

**What:** Rebuilt the GAN-fp attribution method as a DUAL path with a shared, reproducible
head-to-head benchmark:
- **Path B (CNN, new):** `scripts/lib/ganfp_net.py` - a Yu2019-INSPIRED CNN (Yu2019 learns
  the fingerprint with a CNN; we keep that idea and add a fixed SRM forensic front-end - this
  is NOT a byte-faithful Yu2019 reimplementation). A FIXED (non-trainable, DC-suppressed) SRM
  high-pass front-end `Conv2d(1,30,5,bias=False)` (Spatial Rich Model bank, Fridrich &
  Kodovsky 2012; 30 distinct DC-suppressed filters, `requires_grad=False`) feeds 3 VGG-style
  conv blocks (Conv-BN-ReLU x2 + MaxPool) + AdaptiveAvgPool2d(1) + Linear(128)+ReLU+
  Dropout(0.3)+Linear(C). Channels are configurable: code default is [32,64,128] (~330K
  trainable), but `config.yaml` uses the sweep-winning [16,32,64] (~82K, val_top1=0.736 on the
  1626-image set; bigger nets overfit). Single-channel Rec.601 luminance input (256x256),
  no z-scoring (SRM high-pass + BN replaces it). `GANFpDataset` (torch Dataset, lazy torch
  import) reuses `image_ops.load_rgb`/`scale_to`/`make_jpeg_augmenter`; `GANFpClassifier`
  wraps the CNN + Adam + weighted CrossEntropy and mirrors `defake_head._MLPHead`
  (`fit`/`predict_proba`/`predict`/`save(path,classes)`, best-val checkpoint).
- **Path A rigor (PCA, modified):** `scripts/lib/ganfp.py` gains
  `FingerprintStandardizer` (StandardScaler + PCA fit on TRAIN ONLY -> leakage guard) and
  `build_pca_pipeline` (optional additive 8x8 block-DCT fusion channel -> 96-dim input when
  `config.ganfp.pca.dct_fuse` is true). Existing `extract_fingerprints`/`build_features`
  signatures and the `_signature` cache key are UNCHANGED so existing caches still load.
- **Benchmark (new):** `scripts/benchmark_attribution.py` builds ONE seeded stratified split
  and runs BOTH paths on identical tr/va/te index arrays, emitting `benchmark_metrics.json`
  (split + classes + per-path attribution/detection + comparison) plus per-path confusion
  matrices and per-image CSVs, and the two saved heads (`ganfp_pca_head.pt`, `ganfp_cnn.pt`).
  Optional `--defake_csv`/`--dct_csv` ingest DE-FAKE/DCT per-image CSVs as extra comparison
  rows.

**Why:** The original Path A (hand-crafted residual/spectrum features + MLP) is fast and
interpretable but limited by its fixed feature design. A Yu2019-inspired CNN lets the conv
filters themselves BECOME the learned model fingerprints (the method's actual claim), giving
a fair end-to-end comparison against the feature path over the same data and split. The
train-only PCA removes the leakage risk the GOLD review flagged, and the benchmark gives one
script that reports CLIP/DE-FAKE vs GAN-fp-feature vs GAN-fp-CNN side by side. Config gains
`ganfp.cnn` and `ganfp.pca` sub-blocks; `tests/test_ganfp_net.py` covers the high-pass kernel
(DC-suppressed), the luminance helper, the FingerprintStandardizer leakage guard, and (torch-
gated via `pytest.importorskip`) the CNN forward shape / param budget / one-step / Dataset.
`compileall` stays torch-free (every torch import is inside class/method bodies).

## 12. Supervisor feedback (Dennis) + resulting decisions

**What:** Dennis answered our email. Captured here so the decisions are traceable:
- **GAN-Fingerprints re-implementation: approved.** Caveat: if we use generative AI to
  re-implement the method, we must **disclose it, not hide it**. -> The team will add the
  AI-assistance disclosure statement at the final-submission stage (owner: team).
- **Priority steer:** expanding **DE-FAKE to multi-class (fine-tuned head) should take PRIORITY
  over GAN-Fingerprints** if treated as separate tasks. -> Next active workstream is
  `finetune_defake_head.py` (attribution), GAN-fp polish is secondary.
- **Format/resolution confound:** good observation, note it for the report as a boundary of the
  work. **Open question he raised:** did the model output actually CONFIRM the data is
  separable by format/resolution, or is the model only *partially* using it? -> We have NOT yet
  run that ablation; added it as a planned experiment (see below). Honest current status: the
  confound is a *design risk we control for*, not yet a *measured effect*.
- **256px normalization / aspect ratio:** Dennis flagged that squashing every image to
  256x256 (`scale_to`) **distorts non-square images** (CelebA 178x218, London-DB), while our
  square 512 fakes downscale cleanly - so squashing risks **replacing** the format/size
  confound with an **aspect-distortion** confound rather than removing it. -> Added
  `image_ops.resize_shortest_center_crop` (aspect-PRESERVING: resize shortest side then
  center-crop, no stretch; square fakes reduce to a pure downscale = identical resample to
  reals) and a new `"aspect"` variant in `prepare_variants.py`. The confound-controlled runs
  should use `"aspect"`, not `"scaled"`.
- **OpenForensics: advised to add it.** Real+fake faces in the SAME image mitigate the
  confound. Requires preprocessing (parse the per-image JSON, extract + label faces). Dennis is
  uploading the missing JSON files to the GPU PC (a subset was missing). -> Added to the data
  backlog (blocked on the JSON upload + a face-extraction script).

**Planned confound-verification experiment (to answer Dennis's question directly):**
1. Train a trivial classifier on *metadata only* (width, height, on-disk format) -> if it
   scores high, the confound is real and strong.
2. Compare DE-FAKE / DCT detection on the `"scaled"` (squashed) vs `"aspect"` (undistorted)
   variants -> a large drop isolates how much the model leaned on distortion/format.
3. Report both as a measured result (turns the confound into evidence, per GOLD).

---

## Open items still needing the supervisor

Answered by Dennis (see section 12): (A) GAN-Fingerprints re-implementation - APPROVED (with AI
disclosure); (B) reals - ADD OpenForensics (pending JSON upload). Still open: (C) the report
submission date; and an optional BBB call offered for this Thu/Fri if needed.
