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
ALONE (no separate DFFD pass + `merge_predictions.py`, which would double-count DFFD). The
DFFD-only subset run is now `run_defake_batch.py --dataset_filter dffd_`.

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

**Update - clean matched deltas now in (scaled vs aspect on the SAME 1446 build):**
- **Attribution geometry confound:** in-set balanced acc 95.5% (scaled/squash) vs 93.9% (aspect)
  -> distortion buys only ~1.5 pts. StyleGAN3 in-set recall 0.86 vs 0.82. Out-of-set collapses to
  0 in both. The fine-tuned head is NOT exploiting aspect distortion.
- **DCT geometry confound:** AUROC 0.761 (scaled) vs 0.777 (aspect); balanced 0.697 vs 0.703.
  Distortion does not help DCT (slightly hurts). Supersedes the old 926-sample caveat.
- **DCT out-of-set (matched, holdout FLUX+StyleGAN3):** balanced 0.54, AUROC 0.62 -> ~chance.
- **LOGO raw baseline (JPEG-aug OFF)** reproduces the GAN-collapse asymmetry: StyleGAN3->FFHQ
  102/108 (FKR 0.95), FLUX->SD1.5 81/108 (FKR 0.87). The finding is not a normalization artifact.

**Metadata-only confound probe (Dennis's direct question) - DONE:** RandomForest on
width/height/aspect/log-area/format ONLY (no pixels). RAW master: balanced acc **0.79** /
AUROC **0.89** - real vs fake is strongly separable from metadata alone, driven by format
(is_png 0.28 + is_jpeg 0.25 = 53% importance). Normalized aspect variant: **0.50 / 0.50** with
all feature importances 0 - the leak is gone. The 0.89->0.50 gap is the measurement. Per-source:
FFHQ reals (PNG) get called fake in raw, JPEG reals (CelebA/London-DB) stay real -> confirms
format, not content, is the raw leak. Confound section is now complete end-to-end.

## 9d. Robustness (WS7) - DE-FAKE binary under perturbation

**What:** Held-out test split (n=290, aspect variant), 8 perturbations via robustness_perturb.py:
JPEG q30/50/70, Gaussian blur sigma 1/2, resize round-trip 0.5/0.75, sharpen. Clean baseline =
DE-FAKE on the unperturbed `results/test_index.csv`; each perturbation scored vs it.

**Result - stable metric, unstable predictions:**
- Aggregate accuracy STABLE: clean 0.555, perturbed 0.524-0.600 (max |drop| 0.045). No collapse.
- Per-image labels are NOT stable: label-flip rate jpeg30 **0.334**, sharpen **0.303**, jpeg50
  0.252, blur2 0.186, jpeg70/blur1/resize0.5 ~0.13-0.14, resize0.75 0.107. Aggregate looks flat
  only because flips are ~symmetric.
- High-frequency edits (jpeg30, sharpen) drop prob_fake ~0.21 -> since the baseline over-calls
  faces fake (real spec ~0.29), this slightly RAISES accuracy (0.60). Low-pass edits (blur/resize)
  nudge prob_fake up, negligible accuracy change.

**Reading:** robustness is entangled with the fake-bias/threshold issue from section 9. DE-FAKE's
headline number is perturbation-insensitive, but individual decisions are volatile (up to 1/3 flip
under mild JPEG). Honest characterization for the report; also a caveat that "stable accuracy"
here partly reflects a near-chance, fake-biased classifier, not genuine invariance.

**Runbook note:** `generate` mode writes only the 8 perturbation indices (no index_clean.csv);
the clean baseline is `results/test_index.csv` itself. PIPELINE.md updated accordingly.

## 10. GAN Fingerprints (Yu2019-inspired) reproduced in PyTorch

> **STATUS: ON MAIN.** GAN-fp was temporarily parked on the `ganfp-integrated` branch while
> DE-FAKE multi-class attribution took priority; it has now been consolidated back onto `main`
> in complete form (feature + CNN paths, faithful Fridrich-Kodovsky SRM front-end, benchmark)
> alongside the `--hflip` and per-generator-skip-count fixes. It is a SECOND attribution method
> beside DE-FAKE, not a replacement. See scripts/train_ganfp*.py + benchmark_attribution.py and
> the WS4 block in docs/PIPELINE.md.

**What:** Added a GAN-fp attribution path as a second method beside the CLIP/DE-FAKE head.
`scripts/lib/ganfp.py` extracts residual + FFT-spectrum fingerprint features (luminance,
downsampled to a fixed grid, L2-normalized); `scripts/train_ganfp.py` trains
`defake_head._MLPHead` on them (multi-class attribution + secondary binary detection, writes
`ganfp_metrics.json`/confusion matrix/per-image/`ganfp_head.pt`); `scripts/run_ganfp_infer.py`
emits a per-image CSV the existing `eval_defake_attribution.py` consumes unchanged. A `ganfp:`
block was added to `configs/config.yaml`; `tests/test_ganfp.py` covers the feature math (no
torch). (The old `run_ganfp.py` weight-discovery/scope-note helper was retired once the
reproduction landed on main.)

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
  - **Decision (how OF is wired):** because Dennis's strongest steer was to DIVERSIFY the narrow
    real class (GOLD concern #1), OpenForensics **reals become a TRAINED real class** (added to
    `attribution.real_generators`, `sample_size: 300` to size-match FFHQ/CelebA). OpenForensics
    **fakes are kept OUT-OF-SET** (`OpenForensics-fake`) as a genuinely unseen manipulation type
    = clean generalization probe. Same-photo OF real/fake pairs (shared JPEG q95) still serve as
    the format-matched DETECTION confound benchmark. Trade-off accepted: OF reals are no longer a
    held-out real probe (diversify intent wins for the real side); OF-fake stays the held-out fake
    probe. Requires re-running the attribution fine-tune + detection with the new real class.
  - **Confound gate (colleague's crop-size concern):** before trusting OF numbers, run the
    metadata-only probe restricted to OF rows (`metadata_confound_probe.py --source_filter
    openforensics`). ~0.5 balanced acc/AUROC = OF is clean; if high (crop bounding-boxes leak the
    label by size), route OF through the `aspect` normalization and re-probe to ~0.5 first.
  - **Confound gate RESULT (measured, 600 OF crops = 300 real + 300 fake):** RAW OF is real/fake
    separable from crop GEOMETRY alone at **balanced acc 0.608 / AUROC 0.634** (importances: aspect
    0.36, log_area 0.22, width 0.21, height 0.21; is_png/is_jpeg = 0.0 -> format is NOT the leak,
    as designed since OF is uniformly JPEG q95). So raw OF has a moderate crop-SIZE confound (real
    vs fake faces differ in bbox size/aspect). After the `aspect` normalization (every crop ->
    common square) it collapses to **exactly chance (0.500 / 0.500, all importances 0)**. Therefore
    OF is ONLY used on the aspect variant; raw-geometry OF numbers are not reported. This mirrors
    the format-confound story (0.89 -> 0.50) on the size axis.

**Planned confound-verification experiment (to answer Dennis's question directly):**
1. Train a trivial classifier on *metadata only* (width, height, on-disk format) -> if it
   scores high, the confound is real and strong.
2. Compare DE-FAKE / DCT detection on the `"scaled"` (squashed) vs `"aspect"` (undistorted)
   variants -> a large drop isolates how much the model leaned on distortion/format.
3. Report both as a measured result (turns the confound into evidence, per GOLD).

## 13. Scientific rigor upgrades (colleague review)

**What:** A colleague reviewed the project and flagged that the headline numbers were reported
as bare point estimates on a small test set (~22 images/fake class) with a single seed and an
oracle-picked threshold. Adopted the credibility-focused subset of that feedback. These are
add-ons: **no existing result is invalidated**, we quantify its reliability. All new scripts are
numpy/PIL/sklearn-only (no new server deps).

- **Uncertainty quantification.**
  - `scripts/bootstrap_metrics.py`: stratified bootstrap (N=2000) 95% CIs for detection and
    attribution headline metrics, plus per-class recall CIs. Auto-detects detection vs
    attribution input.
  - `scripts/seed_sweep.py`: re-splits + re-trains ONLY the MLP head from the cached CLIP
    features over K seeds (fast, no CLIP recompute) and reports mean/std/95% CI of in-set
    balanced accuracy and per-class recall - isolates head/split variance.
  - `scripts/compare_models_significance.py`: paired DE-FAKE-vs-DCT test on the SHARED test
    paths (McNemar exact + paired AUROC/bal-acc bootstrap). Enabled by two small DCT changes:
    `dct_svm.py` now writes `dct_per_image.csv` (`dct_extract_features.py` already stored paths).
- **Detection threshold hygiene** (`scripts/score_defake_detection.py`): the old single
  "best threshold on all rows" is now split into three clearly-labeled blocks under
  `overall.thresholds` - `fixed_0p5` (honest default), `validation_selected` (threshold picked
  on a seeded stratified val holdout, metrics on disjoint test rows -> the reportable operating
  point), and `oracle_upper_bound` (best-on-all-rows, explicitly non-achievable ceiling).
- **Split-leakage audit** (`scripts/audit_split_leakage.py`): dependency-free exact (SHA-256) +
  near-duplicate (dHash Hamming) checks across the train/val/test partition + per-generator
  balance counts. DIAGNOSTIC only - we are NOT switching to identity-group splitting (no
  identity labels; per-source grouping is degenerate because source == generator == class).
- **Metadata-confound variant sweep (measured).** Completeness table for the metadata-only
  real/fake probe (RandomForest on width/height/aspect/log-area/format, NO pixels):

  | Variant | balanced acc | AUROC | reading |
  |---|---|---|---|
  | raw master | ~0.89 | ~0.89 | format/resolution leak is real and strong |
  | aspect (256 PNG) | 0.50 | 0.50 | leak removed by normalization |
  | scaled (256 PNG) | 0.50 | 0.50 | leak removed |
  | cropped (256 PNG) | 0.50 | 0.50 | leak removed |
  | jpeg30 test (256 PNG) | 0.50 | 0.50 | leak removed |

  Every normalized variant collapses to chance; the ~0.89 -> 0.50 gap IS the measurement that
  the preprocessing removes the metadata confound (directly answers Dennis's question).
- **GAN-fp fixes (now on main via the consolidation below):**
  `train_ganfp_cnn.py` `--hflip` parsing bug fixed (`bool("false")` was `True`; now explicit
  `== "true"`); `ganfp.py` feature/DCT extraction now logs per-generator skip counts for
  unreadable images (no silent class-shrinking bias); wording audit confirmed the honest
  "Yu2019-inspired / not byte-faithful" phrasing throughout (no "we reproduce GAN Fingerprints"
  overclaims).
- **Branch consolidation:** everything is now on `main` - the rigor upgrades, the GAN-fp
  implementation (un-parked from `ganfp-integrated` in complete form), and OpenForensics wiring
  (`openforensics_fake` config entry with a DISTINCT `OpenForensics-fake` generator so the
  real/fake taxonomy stays exclusive; `scripts/ingest_openforensics.py` sorts the flat crops
  into real/ and fake/ for the config-driven `build_master_index.py`). The `ganfp-integrated`
  branch is redundant after this and removed.

**Why:** Turns "DE-FAKE in-set bal-acc 0.94" into "0.94 (95% CI ...), stable across 10 seeds",
reports a threshold we could actually pick without peeking, proves the split is clean, and shows
the confound is gone across every normalized variant - the difference between a plausible result
and a defensible one (GOLD).

## 14. Colleague review response: leakage fix, fair benchmark, LOGO reframe, coupling audit

**What:** A colleague reviewed the project ahead of a supervisor meeting and flagged 15 issues
plus an honest question about OpenForensics same-source-photo coupling. Addressed the parts
fixable without a server re-run; documented the rest as explicit BLOCKING items with a runbook
to close them. No existing reported number is silently changed - every fix either corrects a
bug going forward (re-run required) or adds a caveat/column to make an existing comparison
self-documenting.

- **DCT-SVM train/test-boundary leakage (blocking).** `dct_svm.py --mode random` drew its own
  internal split stratified on the BINARY label, while the robustness test set
  (`make_split.py`) stratifies on the 12-class GENERATOR column - same seed/test_size, different
  partition, so a fraction of `test_index.csv` could sit inside the SVM's own training data.
  `robustness_perturb.py` then scored PERTURBED copies of some of those training rows against
  the SVM, inflating the "clean" DCT baseline that every §9.2 robustness delta was measured
  against. Fix: `dct_svm.py --mode random` now accepts `--test_index <csv>`, which makes the
  train/test boundary IDENTICAL to the shared split instead of re-deriving one. All DCT
  robustness numbers must be regenerated with this flag (docs/PIPELINE.md updated: the DCT-SVM
  used for robustness is now fit with `--test_index` before being reused for
  `compare_models_significance.py` too, closing a second copy of the same bug that the original
  runbook had by re-fitting the SVM a second time without the flag).
- **Benchmark §8-style comparison not apples-to-apples (blocking).** GAN-fp trains on all 12
  index classes; the DE-FAKE head trains on only its own 7 (4 reals + SD1.5/FLUX/StyleGAN3).
  `benchmark_attribution.py` already supported `--classes` to restrict GAN-fp's class set, but
  the comparison table did not document either method's training regime. Added
  `classes_trained_on`/`n_classes_trained_on` to every row of `benchmark_metrics.json`'s
  `comparison` list (DE-FAKE's own classes are read back from its `finetune_metrics.json`), plus
  a logged warning when the two methods' class sets differ, telling you to re-run with
  `--classes` restricted to the shared 7 for the fair number.
- **LOGO naming (leave-one-generator-out vs leave-new-class-out).** The default
  `leave_one_generator_out.py --targets` is `in_set_generators + finetune_new_classes` = only
  FLUX.1-schnell and StyleGAN3-FFHQ - both classes the regular head IS trained on. That is
  "leave-NEW-class-out", not a full LOGO sweep. Added `--all_trained_classes` (holds out every
  real + fake trained class in turn, including a real class like CelebA) plus an
  `is_real_class` field in `logo_summary.json`, and a runtime warning when the default targets
  are used unmodified. `report/REPORT_OUTLINE.md` section 8 renamed/reframed accordingly and a
  new §8b placeholder added for the full-sweep result.
- **Out-of-set top1=0.000 footnote.** Added the one-line reframe to §8 of the outline:
  top-1=0 is definitional (no unseen label in the output space), the false-known rate is the
  metric that matters.
- **Robustness AUROC gap.** `robustness_perturb.py --mode score` now computes
  `auroc_clean`/`auroc_perturbed`/`auroc_drop` whenever a numeric `--conf_col` is present (e.g.
  the DCT SVM's decision-function `score`), so the DCT robustness table can show ranking
  degradation, not just balanced accuracy at a fixed threshold.
- **`sd15_img2img` config comment.** It was never generated (absent from every dataset table
  and result); the config note now says so explicitly ("NOT GENERATED for this run") instead of
  the ambiguous "may not exist yet".
- **Path A (GAN-fp feature+MLP) 73-point discrepancy.** `train_ganfp.py` (standalone) and
  `benchmark_attribution.py`'s Path A use the identical content-stable split + an
  index-content-hashed feature-cache signature that refuses to reuse a mismatched cache, so a
  same-index/same-classes run cannot legitimately diverge this much. Far more likely: the
  standalone number is stale, from an earlier/smaller run (the report already documents an
  analogous 200-image-toy-set-vs-1626-image-real-set swing in `report/GANFP_REPORT.md` section
  5-6). Recommendation logged in `report/REPORT_OUTLINE.md` section 6: either re-run standalone
  with the exact index/classes/fresh cache the current benchmark used and confirm agreement, or
  simply stop reporting a separate standalone number and cite `benchmark_metrics.json`'s
  `path_a` block as the single source of truth.
- **OpenForensics same-source-photo coupling.** Flagged by the colleague as a real leakage
  vector the dHash near-duplicate audit cannot see (real and fake crops of one photo are
  different pixels). `extract_openforensics.py` names crops by annotation id and drops the
  source image id, so we cannot yet do group-aware splitting without a re-extraction. Added
  `scripts/audit_openforensics_coupling.py`: a re-extraction-free QUANTIFICATION that re-parses
  the original `*_poly.json` (metadata only) to recover annotation_id -> image_id and
  cross-references it against the crop filenames + current split, reporting how many source
  photos contributed both a real and a fake crop and how many of those straddle the train/test
  boundary. Decision on document-only vs. group-aware re-extraction is deferred until that
  number is in (see Open items below) - both paths remain available.
- **Datasheets, hyperparameter search, CLIP+BLIP confound, no-SOTA-baseline:** all converted
  from silent gaps into explicit, worded limitations in `report/REPORT_OUTLINE.md` section 10
  (CNN channel sweep is now documented as an informal grid search since it demonstrably was
  one; MLP/DCT hyperparameters are documented as fixed-at-defaults, not searched).

**Why:** A colleague's review ahead of a supervisor meeting is exactly the moment to fix
measurement bugs (leakage, unfair comparisons) rather than defend numbers that would not survive
scrutiny, and to convert every remaining gap into an explicit, worded limitation rather than a
silent omission - consistent with the project's GOLD framing (scientific reliability over
maximal accuracy).

**What (follow-up, same session): OpenForensics coupling - DECIDED, group-aware fix implemented.**
Went with the scientifically cleanest option (over document-only or quantify-first):
- `extract_openforensics.py` now records each crop's source `image_id` in
  `openforensics_metadata.csv` (new `source_image_id`/`source_split`/`annotation_id` columns)
  AND writes a dedicated `openforensics_groups.csv` sidecar (`full_path,source_image_id`).
- `scripts/lib/defake_head.py`'s `stratified_split`/`_hash_stratified_split` gained an optional
  `groups` argument: samples sharing a group id are assigned to a split side as a WHOLE group
  (hashed on the group id), while every other row (no sidecar entry -> its own path is a
  singleton group) is split via the EXACT original per-class hash-ranked algorithm - verified
  byte-identical via `test_group_none_is_byte_identical_to_ungrouped` and a coupled-pair test in
  `tests/test_defake_head.py`. Fully backward compatible: no non-OpenForensics split changes.
- New `scripts/lib/io_utils.load_group_map` / `apply_group_map` / `default_group_map_paths`
  helpers (auto-load `<dataset_root>/openforensics/openforensics_groups.csv` unless
  `--group_map` overrides it), wired into `finetune_defake_head.py`, `train_ganfp.py`,
  `benchmark_attribution.py`, `leave_one_generator_out.py`, and `make_split.py` (also switched
  from its own ad-hoc sklearn split to the shared `defake_head.stratified_split` for consistency
  with every other split-consuming script).
- `audit_split_leakage.py` now also loads the group map when reconstructing the finetune split
  and reports a new `group_straddle` field (expected 0) as a standing regression check.
- `scripts/audit_openforensics_coupling.py` kept as an independent cross-check (re-derives the
  coupling straight from the raw `*_poly.json` rather than trusting the sidecar).
- **Still needs the GPU server:** re-run `extract_openforensics.py` (the sidecar only exists
  after a fresh extraction) and then every downstream stage that touches OpenForensics rows.

## 15. Pre-server-rerun check: orchestrator + group-aware-split completeness

**What:** Before deleting the server copy and re-running this version, checked (and fixed) three
specific `run_experiment.py` gaps, plus two more of the same class found while checking them:

- **`stage_dct` ordering + `--test_index` (was NOT wired at all).** `run_experiment.py`'s `dct`
  stage called `dct_svm.py --mode random` with no `--test_index`, so it still drew the SVM's own
  internal binary-stratified split - section 14's leakage fix never actually got exercised by
  the orchestrator. Fixed: `stage_dct` now runs `make_split.py` FIRST (writing
  `results/{train,test}_index.csv`, added as `Ctx.train_index`/`test_index` so
  `stage_robustness` points at the identical files instead of re-deriving the path string), then
  trains with `--test_index` unconditionally. `stage_robustness`'s own `make_split.py` call is
  left in place (idempotent/deterministic) so `--stages robustness` alone still works.
- **`stage_ganfp`'s benchmark step missing `--jpeg_aug`/`--device`.** The first two GAN-fp steps
  already passed both correctly; the THIRD step (`benchmark_attribution.py`) passed neither - so
  every orchestrated GAN-fp-vs-DE-FAKE benchmark run silently trained Path B's CNN on CPU (that
  script's `--device` defaults to `"cpu"`) and never JPEG-augmented (its `--jpeg_aug` is a bare
  flag, unlike the `{auto,on,off}` choice flag the other two scripts use). Fixed: `--device
  c.device` always added; `--jpeg_aug` appended only when `c.jpeg_aug == "on"`.
- **`stage_attribution`'s LOGO step only ran leave-NEW-CLASS-out.** Kept that narrow run and
  added a second step using `--all_trained_classes` into a separate
  `results/logo_full_<variant>_<augtag>/` output - the actual full leave-one-generator-out sweep
  report/REPORT_OUTLINE.md section 8b describes, which the orchestrator previously never
  produced at all.
- **Two more group-aware-split gaps found while checking the above:** `train_ganfp_cnn.py`
  (Path B standalone trainer - used directly by `stage_ganfp`) and `ganfp_sweep.py`/
  `seed_sweep.py` (hyperparameter/seed-variance tools) all called `defake_head.stratified_split`
  WITHOUT a `groups=` argument, so they silently kept splitting on the OLD, non-group-aware
  scheme even after section 14's fix. This mattered concretely for `train_ganfp_cnn.py`:
  `stage_ganfp` runs `train_ganfp.py` (already group-aware) and `train_ganfp_cnn.py` (was NOT)
  side by side and expects them to share the same test images - with OpenForensics coupling
  present, they would have silently diverged. All three now load the group map and pass
  `groups=` the same way as every other split-consuming script, with a `--group_map` override
  flag added to each for consistency.

**Why:** A code-level fix that no orchestrated run actually exercises does not fix anything in
practice. This pass specifically targeted "will the next server run actually produce the
corrected numbers" rather than re-auditing already-covered ground; `--dry_run` output was used to
confirm each fix appears in the actual generated command plan, in the right order, before this
was reported as done. Verified: `python -m compileall -q scripts tests` and `pytest -q` both
clean after every change (54 passed, 8 skipped at the time).

## 16. Follow-on gap in the robustness AUROC fix (section 15/`robustness_perturb.py`)

**What:** A colleague caught that the AUROC/accuracy-drop block added to `robustness_perturb.py
--mode score` (section 14) never actually activates for DCT scoring. It is gated behind
resolving a ground-truth `label_col`, which only checked for `schema.LABEL` ("real"/"fake"
string) - present in DE-FAKE's per-image CSVs, ABSENT from `dct_svm.py`'s `dct_per_image.csv`
(columns: `full_path, generator, y_true, score, pred`; `y_true` is already numeric 1=fake).
So every DCT robustness drop JSON silently ended up with only `n` + `label_flip_rate`, never
`accuracy_clean`/`accuracy_perturbed`/`performance_drop`/`auroc_*` - exactly the fields §9.2
needs for the DCT rows, while DE-FAKE's JSONs got the full block (the asymmetry only shows up by
diffing the two JSON shapes side by side, which is why it shipped quietly). Fixed: `score()` now
also recognizes a numeric `y_true`/`y_true_clean` column as ground truth (falls back to it only
when no `label`/`label_clean` column exists, so DE-FAKE's path is unchanged). Added
`tests/test_robustness_perturb.py` (DE-FAKE-schema and DCT-schema cases) so this asymmetry
cannot silently regress again.

**Why:** As flagged by the colleague, this was harmless for a run that never adds AUROC columns
to the §9.2 table, but is exactly the fix needed if it does - worth closing now rather than
discovering it after the next server run's DCT robustness JSONs come back missing fields again.

## 17. Group membership must be identity-based, not call-population-based (deeper coupling fix)

**What:** A colleague raised a design-level note: the attribution head's split
(`finetune_defake_head.py`) uses the content-stable hash split, while the robustness split
(`make_split.py`) was (at the time) a plain generator-stratified sklearn split - different
partitions, meaning `attr_clean.csv` could include images the head was trained on. That specific
premise was already resolved by section 14/15's `make_split.py` rewrite (it now calls the SAME
`defake_head.stratified_split`) - but checking it empirically (not just by inspection) surfaced a
DEEPER version of the same risk that the rewrite had not actually closed:

`_hash_stratified_split`'s group-vs-singleton decision was based on **counting how many rows of
a group id are present in the arrays passed to that specific call** (`size_by_group[g] <= 1`).
That count is not stable across callers: `finetune_defake_head.py` restricts to the TRAINED
class set BEFORE splitting, which removes `OpenForensics-fake` (out-of-set) entirely - so an
`OpenForensics` (real) row whose paired source photo also produced a sampled
`OpenForensics-fake` crop loses its only groupmate and gets treated as a lone SINGLETON there
(split via the per-class hash-rank algorithm), while `make_split.py`'s unrestricted call still
sees both members and treats the SAME real row as part of a 2-member GROUP (split via the
group-id hash). Two different algorithms picking two different buckets for the identical row ->
exactly the leakage the colleague was worried about, just via a more specific mechanism than
"different split functions." Verified numerically before/after (synthetic index with trained +
out-of-set classes + 10 coupled pairs): pre-fix, 2/10 coupled real rows disagreed between the two
call styles (`make_split test rows that were in finetune train/val: 2`); post-fix, 0.

**Fix:** Changed the grouped/ungrouped decision in `_hash_stratified_split`
(`scripts/lib/defake_head.py`) from a co-occurrence COUNT (`size_by_group[g] <= 1`) to an
IDENTITY check (`groups[i] != keys[i]`, i.e. "did `io_utils.apply_group_map` find an explicit
sidecar entry for this row, regardless of whether any other member of that group happens to be
present in this call"). A grouped row's bucket now depends ONLY on `(group_id, seed)`, so it is
identical across every caller no matter how that caller filtered its population beforehand -
including the case where a row is the ONLY member of its group present in a given call because
its sibling was filtered out as out-of-set. `groups=None` (or every key its own group, the
default for every non-OpenForensics dataset) is unaffected: `is_grouped` is then `False`
everywhere, same as before. Added two regression tests to `tests/test_defake_head.py`:
`test_group_membership_is_id_based_not_call_population_based` (unit-level: a group whose second
member is simply absent from the call must still resolve via the group hash, not per-class
ranking) and `test_group_decision_matches_across_differently_filtered_calls` (end-to-end:
simulates the actual finetune-restricted vs. make_split-unrestricted scenario and asserts zero
disagreement/leakage).

**Why:** This is the difference between "group-aware splitting exists" and "group-aware
splitting actually prevents the leak in the one case (OpenForensics real paired with an
out-of-set fake) that motivated it in the first place." Confirms the value of checking a fix
empirically against the real config shape (trained vs. out-of-set classes) rather than trusting
that "groups=" being threaded through every script is sufficient on its own.

## 18. Two more `run_experiment.py` review findings

**What:** A colleague flagged two more things in the orchestrator:

- **`PERTURBATIONS` was a hand-typed copy of `configs/config.yaml`'s `robustness:` block** -
  adding a perturbation to the config (e.g. `sharpen: [1.0, 2.0]`) would silently never reach
  the orchestrator's `stage_robustness`. Fixed: added `_perturbation_names(config_path)`, which
  loads the YAML and calls `robustness_perturb._perturbations` - the SAME function
  `robustness_perturb.py` itself uses to build the real perturbation ops - so the two can never
  drift apart again. Deliberately uses a plain `yaml.safe_load` rather than
  `io_utils.load_config` (which resolves `${WTP_ROOT}`-style placeholders and needs
  `configs/paths.env` / the real environment set up): the `robustness:` block has no
  placeholders, and `--dry_run` should keep working on a machine with no server environment
  configured at all, matching how every other `Ctx` field already avoids that dependency.
  `Ctx.perturbations` is now computed once per run instead of a module-level constant.
  Verified: a temporary config with an added `sharpen: 2.0` value produced exactly 7 more
  dry-run steps (one full perturbation's worth) without any code change, and `--dry_run` still
  works with `WTP_ROOT` unset. Covered by `tests/test_run_experiment.py` (matches the real
  config, grows when the config grows, and works without env placeholders).
- **`stage_robustness` writes DE-FAKE prediction CSVs (`robust_clean_pred.csv`,
  `robust_<name>_pred.csv`) under `dataset/`, not `results/`.** Checked against
  `docs/PIPELINE.md`'s manual runbook (`$DS/robust_clean_pred.csv` etc.) - this is the
  established, intentional convention: every raw `WTP_PRED_CSV` output from
  `run_defake_batch.py` lives in `dataset/` alongside `master_metadata.csv` and the other
  `defake_predictions*.csv` files; `results/` holds only DERIVED analysis artifacts.
  `stage_detect`'s `c.pred` already follows the same convention. No change made - confirmed as
  intended, not a bug.

**Why:** The `PERTURBATIONS` fix is the same class of issue as sections 14-17: a
manually-duplicated copy of something that has a single real source of truth elsewhere will
eventually drift. The `dataset/`-vs-`results/` question was worth checking explicitly rather
than assuming, since it looks inconsistent at first glance; cross-referencing the manual runbook
confirmed it is not.

---

## Open items still needing the supervisor

Answered by Dennis (see section 12): (A) GAN-Fingerprints re-implementation - APPROVED (with AI
disclosure); (B) reals - ADD OpenForensics (pending JSON upload). Still open: (C) the report
submission date; and an optional BBB call offered for this Thu/Fri if needed.

(D) OpenForensics same-source-photo coupling (section 14/17): DECIDED - group-aware fix,
implemented in code (extract_openforensics.py sidecar + defake_head.stratified_split groups= +
wired through every split-consuming script, with the deeper identity-based-grouping fix in
section 17). Not yet closed out: needs a host re-extraction (to actually generate the
openforensics_groups.csv sidecar) and a full re-run of every OpenForensics-touching stage on the
server before the fix is reflected in reported numbers.

## 19. Provenance of the CURRENT (pre-fix) OpenForensics crops + a real bug in the audit script

**What:** The team clarified how the OpenForensics crops currently on the server were actually
produced: an older, ad-hoc host-side script (`extract_openforensics_faces.py`, not the version
in this repo), used because `/vol1` isn't mounted in the container. Comparing it to this repo's
`scripts/extract_openforensics.py` clarified two things:

- It defaults to **all four splits** (`Val, Train, Test-Dev, Test-Challenge`), not Val only, and
  has no `--per_class_limit`/seeded cap - it writes every annotation it finds, flat (no real/
  fake subdirectories; a separate `ingest_openforensics.py` pass - already in this repo - sorts
  the flat output + its metadata CSV into `real/`/`fake/` for `build_master_index.py`). It also
  never recorded `image_id` either, so the same-source-photo coupling risk (section 14) was
  present in the historical data too, for the same reason.
- None of this changes the re-extraction PLAN already given: `dataset/openforensics/` gets wiped
  and rebuilt from scratch with THIS repo's `scripts/extract_openforensics.py --splits Val
  --per_class_limit 300 --seed 42` (the documented, current command), which writes real/fake
  subdirectories itself (no `ingest_openforensics.py` step needed) and now records the
  `source_image_id` sidecar. The old script's quirks (flat output, hardcoded `generator:
  "OpenForensics"` for both labels in ITS OWN metadata CSV, uncapped count) are all moot once
  that directory is wiped - none of them are read by `build_master_index.py` anyway (it derives
  `generator`/`category` from `configs/config.yaml`'s dataset entry, never from a per-dataset
  metadata CSV; the DFFD/CelebA-style `sample_size` cap already runs at index-build time
  regardless of how many raw crops exist on disk).

**Real bug this surfaced, fixed regardless of the above:** `scripts/audit_openforensics_coupling.py`
(section 14) is designed to accept multiple `--polygon_json` files (`nargs="+"`), but its
`_load_ann_to_image` keyed the annotation_id -> image_id map by a BARE annotation id, unioned
across every file given. COCO-style annotation/image ids are only guaranteed unique WITHIN one
split's JSON export - Val_poly.json and Train_poly.json can (and do, in OpenForensics) reuse the
same small integer id for completely unrelated annotations. A second `--polygon_json` file could
silently overwrite an earlier split's (correct) mapping on any id collision, corrupting the
coupling counts for whichever rows got mapped to the wrong photo - exactly the kind of scenario
the old multi-split-by-default script makes plausible. Fixed: every lookup is now keyed by
`(split, id)`, with the split derived from the JSON's own filename (`<Split>_poly.json`) and from
each crop's own filename (`openforensics_<Split>_<ann_id>.jpg`) - matching
`extract_openforensics.py`'s `source_split`/`source_image_id` convention exactly. A crop whose
split isn't covered by any `--polygon_json` given is now logged and dropped, never silently
mismatched. Added `tests/test_audit_openforensics_coupling.py` (asserts two different splits'
colliding `id=5` annotations both survive, keyed separately). Full suite now 64 passed, 8 skipped.

**Why:** Good thing to check rather than assume - a script written to "support multiple splits"
that was never actually exercised with more than one file had a latent correctness bug exactly
where COCO id semantics are least intuitive. Worth fixing regardless of whether OpenForensics
extraction ends up Val-only (the current plan) or eventually needs more splits.

## 20. Two more findings during the actual server run: caption remapping + missing --device

**What:** A colleague, watching the run's `--dry_run` plan (section 15-18 territory), caught two
more real issues:

- **Attribution robustness captions silently go empty for every perturbed image.**
  `predict_defake_head.py` (used by `stage_robustness`'s "Attribution predict" steps) passes
  `--index` (a robustness_perturb.py perturbation index, e.g. `index_jpeg30.csv`) straight to
  `features_cache.build_features`, which looks up each row's caption from `--captions_csv` KEYED
  BY THE INDEX'S OWN `full_path`. A perturbation index's `full_path` values point at the NEW
  perturbed images (`dataset/robust/jpeg30/...`), which never appear as a key in a captions CSV
  built from the ORIGINAL images - every perturbed row's caption lookup missed and silently fell
  back to `""` (`cap_map.get(p, "")`), which then gets encoded through CLIP text and mixed into
  the 1024-dim feature. The CLEAN baseline (`test_index.csv`, no perturbation, real paths)
  correctly got real captions. Net effect: the measured attribution label-flip-rate and
  confidence-drop under perturbation were conflating the actual image perturbation with a
  caption-mismatch artifact that the clean baseline never had - likely inflating both. Does NOT
  affect DE-FAKE detection robustness (`run_defake_batch.py` runs BLIP live on each image, no
  external caption CSV) or DCT robustness (image-only features) - isolated to attribution
  robustness only. Fixed: `predict_defake_head.py` gained `_resolve_captions_csv()`, which checks
  for a `source_path` column (present on every perturbation index, absent on `test_index.csv`)
  and, when present, builds a temporary captions CSV keyed by the CURRENT full_path but with each
  caption looked up via `source_path` in the original captions CSV - so a perturbed image
  inherits its source image's real caption. Indices without `source_path` are passed through
  unchanged (zero behavior change for the clean baseline). Added
  `tests/test_predict_defake_head.py` (remap case, no-source_path passthrough case, None
  passthrough case).
- **`stage_ganfp`'s Path A step (`train_ganfp.py`) never got `--device`.** Its default is `"cpu"`
  (unlike `train_ganfp_cnn.py`/`benchmark_attribution.py`, which sections 15/18 already fixed to
  receive `--device`) - Path A's small MLP head would have silently trained on CPU instead of the
  requested `cuda`. Not a correctness bug (a tiny MLP trains to the same result either way, just
  slower) but an inconsistency worth closing. Fixed: `--device`, `c.device` added to that step.

Both verified via `--dry_run` after the fix (Path A's command now shows `--device cuda`) and the
full test suite (67 passed, 8 skipped).

**Note on the run already in progress:** because `run_experiment.py` shells out to a FRESH
subprocess per step (reading the script file from disk at the moment that step starts, not
pre-loaded at launch), any step not yet reached when this fix lands will pick it up
automatically - but which steps that covers depends on timing we can't verify remotely. Safest
plan: after the main run finishes, unconditionally re-run just the "Attribution predict" +
"Attribution score" steps for all 8 perturbations (cheap relative to the full run) rather than
trust a timing guess, so the attribution-robustness numbers are correct regardless of when the
fix actually landed relative to the run.

**Why:** Both were real, verified in the exact code path (not just plausible) before being
called confirmed. The caption bug specifically would have inflated exactly the kind of number
(attribution robustness under perturbation) this project's whole rigor push has been about
getting right - worth catching before it ships in the report.

## 21. The group-aware split fix was a silent no-op the entire first real server run

**What:** After the first full `run_experiment.py` run completed, `audit_openforensics_coupling.py`
reported **12/12 coupled OpenForensics source photos STRADDLING the split** - i.e. every single
real/fake pair sharing a source photo landed on opposite sides. That is worse than pure chance
would even produce (~46% expected with zero grouping, for this test_size/val_size), and directly
contradicts sections 14/17's group-aware split fix. Passing `--group_map` explicitly (section 20's
fix for the audit script's own path auto-detection) did NOT change the result, which ruled out
"the audit script couldn't find the sidecar" and pointed at something deeper.

**Root cause, traced through the actual code (not guessed):** `extract_openforensics.py` runs on
the HOST (required - `/vol1` isn't mounted in the container) with `--out_dir
/vol2/pitsec_sose26_topic8/sharedDockerDir/dataset/openforensics`, and wrote `full_path` into
BOTH `openforensics_metadata.csv` and the `openforensics_groups.csv` sidecar as `str(dst)` where
`dst = out_dir / label / filename` - i.e. the HOST-absolute path
(`/vol2/pitsec_sose26_topic8/sharedDockerDir/dataset/openforensics/real/...`). But
`build_master_index.py` runs INSIDE THE CONTAINER, where `dataset["dir"]` resolves
`${WTP_ROOT}` to `/pitsec_sose26_topic8` - so `master_metadata.csv` (and everything derived from
it: `index_aspect.csv`, every split, every feature cache) records the SAME physical files under
the CONTAINER-absolute prefix (`/pitsec_sose26_topic8/dataset/openforensics/real/...`) instead.
Same files, two completely different absolute-path strings. `apply_group_map`'s exact-string
lookup therefore missed EVERY row, on EVERY split-consuming script, in the ACTUAL in-container
training run too - not just in the host-run audit. The group-aware fix's algorithm (sections
14/17) was correct; the data feeding it was not, at the extraction step, and no test caught this
because every unit test used internally-consistent paths on both sides (this is fundamentally an
integration-boundary bug between two machines, not something a same-process test can see).

**Fix:**
- `extract_openforensics.py` gained `--record_prefix`: the CONTAINER-side equivalent of
  `--out_dir`, used ONLY for the `full_path` strings written into the CSVs (files are still
  physically written under the real `--out_dir`). Files are written under the host path; the
  sidecar and metadata CSV now RECORD the container path, matching what `build_master_index.py`
  will independently reconstruct.
- `io_utils.apply_group_map` gained an optional `logger` parameter: it still does NOT
  basename-match (that would risk false grouping for datasets with non-unique filenames), but
  when a path misses an exact match while a DIFFERENT-prefix/same-filename entry exists in the
  group map, it now logs a loud `GROUP MAP PREFIX MISMATCH` warning - so this exact failure mode
  is surfaced immediately in any future run's logs instead of silently degrading to "no
  grouping happened, nothing looks wrong." Threaded `logger=` through every call site
  (`finetune_defake_head.py`, `train_ganfp.py`, `train_ganfp_cnn.py`, `benchmark_attribution.py`,
  `leave_one_generator_out.py`, `make_split.py`, `seed_sweep.py`, `ganfp_sweep.py`,
  `audit_split_leakage.py`).
- Added regression tests in `tests/test_io_and_config.py`: exact match (no warning), prefix
  mismatch (still correctly falls back to ungrouped, but warns loudly), and no-logger-given
  (silent, safe, matches the pre-fix default).

**Practical remediation (server):** the EXISTING `openforensics_groups.csv` sidecar (and
`openforensics_metadata.csv`) can be repaired in place with a plain prefix rewrite - no need to
re-run image extraction:
```bash
sed -i 's|/vol2/pitsec_sose26_topic8/sharedDockerDir|/pitsec_sose26_topic8|' \
    dataset/openforensics/openforensics_groups.csv dataset/openforensics/openforensics_metadata.csv
```
For any FUTURE extraction, use `--record_prefix /pitsec_sose26_topic8/dataset/openforensics`
from the start so this never recurs. Either way, because fixing the sidecar changes WHICH SPLIT
side each coupled OpenForensics row lands on, every stage downstream of splitting (`dct`,
`attribution`, `oos`, `ganfp`, `robustness`) needs to be RE-RUN against the corrected sidecar for
the group-aware guarantee to actually be reflected in reported numbers - the just-completed run's
results should be treated as "the group-aware fix was not actually active" until that re-run
happens and `n_real_fake_pairs_straddling_splits` / `group_straddle` both read `0`.

**Why:** This is the most important finding of the whole OpenForensics-coupling workstream:
"the code runs without error" and "the fix actually works" are different claims, and only the
second one matters. The 12/12 number - specifically because it was WORSE than a no-grouping
baseline would produce - was the tell that something structural (not statistical noise) was
wrong, and tracing it to its root (rather than re-running and hoping) turned up a genuine,
previously-undiscovered integration bug that a full re-run would otherwise have silently
shipped.

## 22. The real, SECOND root cause: variant indices rewrite full_path, breaking the group lookup

**What:** Section 21's fix (repair the sidecar's host-vs-container path prefix) was applied and
verified correct (`head` showed container-style paths), then re-verified on the server -
`n_real_fake_pairs_straddling_splits` was STILL `12/12`, completely unchanged. Rather than
re-guessing, ran one diagnostic first (as instructed): dump the actual `full_path`/`source_path`
values from `results/index_aspect.csv` for OpenForensics rows and directly test membership
against the (now prefix-corrected) sidecar. Result: `index full_path IN sidecar: 0/600`,
`index source_path IN sidecar: 600/600`. Conclusive, not ambiguous.

**Root cause:** `prepare_variants.py` (which produces `index_aspect.csv`/`index_scaled.csv`/
`index_cropped.csv`) rewrites every row's `full_path` to point at a NEW derived variant file
(`dataset/variants/aspect/openforensics_real/openforensics_Val_10035.png`), preserving the
ORIGINAL pre-variant path only in a separate `source_path` column
(`dataset/openforensics/real/openforensics_Val_10035.jpg`). `openforensics_groups.csv` was
written against the ORIGINAL extraction paths (source_path's target) - so looking up a variant
index's `full_path` directly in the group map, no matter how correct the prefix is, was
GUARANTEED to never match. Section 21's fix was real and necessary but not sufficient; this is
the actual, complete root cause. Notably this affects EVERY split-consuming script that runs on
a variant index (i.e. the entire real pipeline, since `run_experiment.py` always uses
`index_aspect.csv`), not just the audit scripts - group-aware splitting has been a complete
no-op for the ENTIRE first server run, for this reason on top of section 21's.

**Fix:** Added `io_utils.group_lookup_map_from_df` / `load_group_lookup_map` (build a
`{full_path: lookup_key}` map preferring `source_path` over `full_path` when present) and
`apply_group_map_with_lookup` (resolves each path's lookup key BEFORE checking the group map,
but - critically - falls back to the row's OWN full_path, not the resolved lookup key, when
there is no match, so `defake_head._hash_stratified_split`'s singleton/group identity check
`groups[i] != keys[i]` stays correct for unmatched rows instead of spuriously treating every row
as "grouped" just because source_path differs from full_path structurally). Every call site that
previously called `apply_group_map` directly (`finetune_defake_head.py`, `train_ganfp.py`,
`train_ganfp_cnn.py`, `benchmark_attribution.py`, `leave_one_generator_out.py`, `make_split.py`,
`seed_sweep.py`, `ganfp_sweep.py`, `audit_split_leakage.py` x2) now goes through the lookup-aware
version instead. Verified three ways: unit tests (`group_lookup_map_from_df` preference logic,
`apply_group_map_with_lookup`'s correct singleton fallback), an end-to-end test reproducing the
exact variant-index-vs-sidecar shape, and a live simulation using real file I/O with the SAME
data shapes confirmed on the server (10 synthetic coupled pairs, all 10 correctly grouped).

**Practical remediation:** no further data repair needed (the sidecar itself was already fixed
in section 21; this fix is purely in the lookup logic) - just `git pull` and re-run the
split-dependent stages again. Expect `n_real_fake_pairs_straddling_splits` and `group_straddle`
to both read `0` this time.

**Why:** Two independent, unrelated bugs (host/container prefix, then variant-path rewriting)
happened to point at the same symptom (group-aware splitting doing nothing), and fixing only the
first one left the exact same failure mode intact - which is exactly why re-verifying with a
concrete measurement after every fix (rather than assuming "I fixed the bug I found, therefore
the symptom is resolved") matters. This also validates the section-21 defensive warning
(`GROUP MAP PREFIX MISMATCH`) as insufficient on its own for this failure mode - it only fires on
a same-filename/different-prefix near-miss, which does NOT occur here (the variant file has a
completely different filename AND directory from the original, e.g. `.png` under `variants/`
vs `.jpg` under `openforensics/`) - so this class of mismatch is silent by that check's design.
Something to keep in mind: `apply_group_map_with_lookup`'s own near-miss check (inherited from
`apply_group_map`) is checked against the RESOLVED (source_path) key, so it would still catch a
prefix-only mismatch even after this fix; it simply cannot catch "the sidecar and index disagree
on which file is the source of truth," which is a structurally different problem.
