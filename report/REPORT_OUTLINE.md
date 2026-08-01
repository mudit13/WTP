# Final scientific report outline

Target: approximately 15 pages excluding references and appendix. Use only evidence from
authoritative run `2026-08-01_eightway_v1`; superseded 7-class and historical GAN-fp metrics are
not final evidence.

Primary writing sources:

- `docs/DATA_PROVENANCE.md`
- `docs/EXPERIMENT_CATALOG.md`
- `report/AUTHORITATIVE_RESULTS_DRAFT.md`
- Extracted immutable evidence under `results/2026-08-01_eightway_v1_evidence/`

## Abstract — approximately 0.5 page

- Detection-versus-attribution problem
- Eight fake-generator classes and external OpenForensics-fake challenge
- Sequential DCT-SVM → fine-tuned DE-FAKE pipeline
- Main detection, attribution and cascade metrics
- Main conclusion: strong known-class attribution but detection/OOS bottleneck

## 1. Introduction — 1–1.5 pages

### 1.1 Motivation

- AI-generated and manipulated face images
- Difference between detecting a fake and attributing its source
- Why a sequential detector→attributor is operationally meaningful

### 1.2 Research questions

- **RQ1:** How effectively do DCT-SVM and pretrained binary DE-FAKE detect fake face images,
  including an unseen manipulation dataset?
- **RQ2:** How accurately can a fine-tuned DE-FAKE head attribute known fake images to eight
  generator/source classes?
- **RQ3:** How does the sequential DCT→DE-FAKE system generalize under LOGO, external OOS data and
  perturbations?

### 1.3 Contributions

- Reconstructed eight-label fake-source benchmark with documented provenance
- Added London-DB-based SD1.5 img2img class with identity grouping
- Confound-controlled preprocessing and split-integrity safeguards
- DCT detection, eight-way DE-FAKE attribution and end-to-end cascade
- LOGO/external OOS analysis with Cohen’s κ, CIs and seed sensitivity
- FFHQ, image-only and robustness diagnostics explaining failure modes

Do not claim a novel model architecture; the contribution is a controlled evaluation, dataset
design and failure analysis.

## 2. Background and Related Work — 1–1.5 pages

### 2.1 Generator and manipulation families

- GAN synthesis: PGGAN, StyleGAN3
- Diffusion synthesis: SD1.5, FLUX
- Face translation/manipulation: StarGAN, FaceApp, OpenForensics

Keep this task-specific; do not provide a broad tutorial on AI model types.

### 2.2 Detection and attribution

- DE-FAKE and CLIP/BLIP representations
- Frank et al. log-DCT frequency detection
- Closed-set attribution, OOS rejection and LOGO

### 2.3 Relevant datasets

- DFFD, FFHQ, CelebA, London-DB and OpenForensics
- GAN-fp mentioned only as optional historical context

## 3. Dataset Design and Construction — 2–2.5 pages

Use `docs/DATA_PROVENANCE.md` as the factual authority.

### 3.1 Design goals

- Eight mutually named fake-source classes
- Optional merged Real class
- OpenForensics-fake test-only
- Balanced support where feasible
- Measured confound reduction
- Source/identity leakage control
- Reproducible generation and indexing

### 3.2 Label taxonomy and class counts

Fake classes:

1. SD1.5 txt2img
2. SD1.5 img2img (London-DB, strength 0.6)
3. FLUX.1-schnell
4. StyleGAN3-FFHQ
5. FaceApp
6. PGGAN-v1
7. PGGAN-v2
8. StarGAN

Real sources: London-DB, FFHQ, CelebA and OpenForensics-real. External OOS:
OpenForensics-fake.

Include one compact table: source, role, count, native format/resolution and train/test use.

### 3.3 Generated dataset construction

- SD1.5 txt2img prompts, seeds, steps and guidance
- FLUX shared prompts, seeds and four-step schnell configuration
- StyleGAN3 official FFHQ checkpoint, seeds, truncation and resize
- SD1.5 img2img London inputs, pinned revision, strength/CFG/steps, fixed prompt and grouping
- Persistent eye artifacts retained without cherry-picking

Prompt details belong here, not in a standalone chapter.

### 3.4 Preprocessing and confound control

- Aspect-preserving resize/center crop to 256 and PNG export
- Training-only JPEG augmentation
- Raw metadata separability versus chance after normalization
- Content-stable split and source-group sidecars
- Zero group straddles and zero exact duplicates

### 3.5 Dataset-specific threats

- Different face pools across labels
- London-only/fixed-prompt img2img narrowness
- DFFD upstream settings not fully available
- OpenForensics as one manipulation benchmark

## 4. Methodology — 2–2.5 pages

### 4.1 Overall design

- Corrected sequential architecture diagram
- Primary fake-only task versus auxiliary merged-Real task

### 4.2 Binary detection

- Log-DCT 128×128 grayscale features and linear SVM
- Training-only JPEG features and clean test features
- Pretrained binary DE-FAKE external baseline
- Fairness caveat: in-domain supervised versus externally pretrained

### 4.3 Eight-way DE-FAKE attribution

- Frozen CLIP ViT-B/32 image features
- BLIP caption text features
- 1024-dimensional concatenated representation
- MLP head, class weighting and validation-based checkpoint selection
- Auxiliary nine-way merged-Real model

Fine-tuning belongs here, not as a standalone top-level chapter.

### 4.4 Sequential cascade

- DCT fake gate
- Conditional attribution
- Undetected fakes counted as end-to-end failures

### 4.5 Generalization analyses

- Eight-fold LOGO
- OpenForensics-fake forced-label/confidence analysis
- Confidence, entropy and false-known rates

## 5. Experimental Design — 1–1.5 pages

### 5.1 Splits and reproducibility

- Seed 42, attribution 70/10/20, DCT fixed 80/20 with shared test boundary
- Group-aware source identities
- Run ID, commit and config hash

### 5.2 Metrics and uncertainty

- Detection: balanced accuracy, macro-F1, AUROC/AUPRC, Cohen’s κ
- Attribution: top-1, balanced accuracy, macro-F1, κ and per-class recall
- Bootstrap 95% CIs
- Ten-seed sweep
- Paired bootstrap and McNemar test

### 5.3 Supporting studies

- FFHQ removal
- Image-only CLIP versus image+BLIP
- JPEG, blur, resize and sharpen robustness
- Split-integrity audit

## 6. Results — 3–3.5 pages

Use `report/AUTHORITATIVE_RESULTS_DRAFT.md` for exact values and wording.

### 6.1 Detection and unseen-manipulation generalization

- DCT versus pretrained DE-FAKE on shared test rows
- Significant balanced-accuracy difference but non-significant AUROC difference
- DCT OpenForensics-fake below-chance OOS result

### 6.2 Primary eight-way attribution

- Fixed-seed metrics with bootstrap intervals
- Eight-way confusion matrix
- Per-class PGGAN/FaceApp weaknesses
- Fixed-seed versus ten-seed mean

### 6.3 Sequential cascade

- Detection pass rate
- Conditional versus end-to-end attribution
- Per-generator detection bottlenecks
- Real false positives

### 6.4 LOGO and external OOS

- Dominant forced-label relationships
- PGGAN-v1/v2 and SD1.5-mode overlap
- OpenForensics confidence/entropy and false-known rates
- No ordinary OOS top-1/κ claims

### 6.5 Supporting ablations

- Auxiliary nine-way merged-Real result
- FFHQ with/without diagnostic
- Image-only versus image+BLIP

### 6.6 Robustness

- DCT blur/downscale weakness
- Better AUROC preservation under JPEG and sharpening
- Per-image instability caveat

Ablations and robustness support the main RQs; they are not independent major chapters.

## 7. Discussion and Threats to Validity — 1.5–2 pages

### 7.1 Interpretation by research question

- RQ1: DCT operating-point advantage but weak OOS/blur generalization
- RQ2: strong known-class source attribution with seed sensitivity
- RQ3: family overlap and detector-limited cascade

### 7.2 Failure mechanisms

- Loss of frequency evidence under blur/downscale
- PGGAN sibling overlap
- StyleGAN3/FaceApp/FFHQ source-manifold sensitivity
- Closed-set confidence on absent classes

### 7.3 Threats to validity

- Approximately 20–27 test images per fake class
- Small validation sets and checkpoint noise
- Different source face pools
- CLIP image embeddings remain semantic even without BLIP
- London-only/fixed-prompt img2img
- Hyperparameters not exhaustively optimized
- One external OOS benchmark
- DCT/DE-FAKE training-regime asymmetry
- Unknown upstream DFFD parameters
- dHash similarities are diagnostic, not exact leakage

Do not repeat the Results section; explain what the results mean and what they cannot establish.

## 8. Conclusion and Future Work — approximately 0.5 page

- Answer RQ1–RQ3 directly
- State the positive known-class attribution result
- State detector/OOS limitations
- Future multi-source img2img
- Better open-set rejection
- Larger balanced identity-diverse datasets
- Optional forensic-signal control

Do not create a separate long Future Work chapter.

## Individual Contributions — approximately 0.5 page

Each member must own practical methodology/evaluation work, not only literature:

- Dataset generation/provenance/confounds
- DCT detection/robustness
- DE-FAKE attribution/statistics
- LOGO/OOS/cascade
- FFHQ/image-only ablations/reproducibility

## Appendix — excluded from main page count

- Full dataset provenance and generation prompts
- Complete confusion matrices
- Full LOGO distributions
- Robustness tables
- Image-only details
- Split audit/dHash diagnostic
- Run manifest, experiment catalog and commands
- SD1.5 img2img pilot/eye-artifact documentation
- GAN-fp method history without superseded prototype metrics
- AI-assistance disclosure

## Main-report figure/table plan

1. Corrected pipeline diagram
2. Compact dataset taxonomy/count table
3. Paired detection comparison table
4. Eight-way confusion matrix
5. Cascade per-generator table
6. LOGO dominant-label table
7. In-set/OOS confidence histogram

Keep nine-way/FFHQ confusion matrices and full robustness outputs in the appendix.
