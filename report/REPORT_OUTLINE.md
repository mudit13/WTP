# Scientific report outline

Use only evidence from one professor-aligned immutable run. Do not reuse metrics from the
superseded 7-class experiment.

## 1. Introduction

- Problem: detect fake face images, then attribute detected fakes to their generator.
- RQ1: How well does log-DCT with a linear SVM detect fake faces? How does the provided
  pretrained binary DE-FAKE baseline compare?
- RQ2: How well can a fine-tuned DE-FAKE head distinguish eight fake generators?
- RQ3: What happens when one generator is omitted from training, and how does this affect the
  end-to-end cascade?

## 2. Related work

- DE-FAKE: CLIP/BLIP-based binary detection and attribution framework.
- Frank et al.: frequency-domain log-DCT detection.
- DFFD and the included FaceApp/PGGAN/StarGAN subsets.
- StyleGAN3, Stable Diffusion 1.5, and FLUX.1-schnell.
- GAN Fingerprints only as appendix context.

## 3. Dataset

### 3.1 Fake attribution classes

1. SD1.5 txt2img
2. SD1.5 img2img (London-DB, strength=0.6)
3. FLUX.1-schnell
4. StyleGAN3-FFHQ
5. FaceApp
6. PGGAN-v1
7. PGGAN-v2
8. StarGAN

### 3.2 Real data

- London-DB
- FFHQ
- CelebA
- OpenForensics-real

Real data trains the DCT detector and the auxiliary joint attribution model. The joint model
collapses all four sources into one source-balanced `real` class.

### 3.3 Test-only data

OpenForensics-fake is never used for fitting. Its paired real source crops are excluded from the
DCT OOS training population.

### 3.4 Generation provenance

For every generated class, report model/checkpoint revision, prompts, negative prompt, seeds,
steps, guidance, dimensions, source preprocessing, and output count. For StyleGAN3 report the
official FFHQ checkpoint and truncation psi. Include generated datasheets and licenses.

## 4. Confound and leakage controls

- Raw metadata-only separability
- Aspect-preserving 256-pixel headline variant
- Scaled/squashed comparison
- Training-only JPEG augmentation with clean validation/test
- Content-hashed feature caches
- London/img2img identity sidecar
- OpenForensics source-photo sidecar
- Runtime no-group-straddle assertions
- Exact and perceptual duplicate audit

State that normalization controls known format/geometry cues but cannot prove that all
content-related confounds are removed.

## 5. Methods

### 5.1 Detection

- Primary: log-DCT features plus balanced linear SVM
- Baseline: provided pretrained binary DE-FAKE checkpoint
- Shared fixed test boundary
- OpenForensics-fake held-out challenge

### 5.2 Attribution

- Frozen CLIP image and BLIP-caption features
- Small fine-tuned MLP head
- Primary eight-fake class space
- Auxiliary nine-way class space with merged Real
- Balanced checkpoint selection

Clarify that the provided DE-FAKE checkpoint is binary; the multi-class head is trained by this
project.

### 5.3 Cascade

DCT predicts real/fake first. Only a fake decision can receive a generator attribution.
Undetected known fakes count as end-to-end attribution failures.

### 5.4 LOGO

Run eight folds. Each fold removes exactly one fake generator and trains on the other seven.
OpenForensics-fake remains excluded. Report forced labels, confidence, entropy, and rejection;
ordinary top-1 is zero by construction because the held-out class is absent.

## 6. Experimental protocol

- Immutable run ID, git commit, config hash, seed 42
- Group-aware train/validation/test split
- Clean evaluation features
- Primary and auxiliary class spaces declared before training
- Per-class support reported
- Bootstrap 95% confidence intervals
- Ten-seed sensitivity analysis

## 7. Results

### 7.1 Dataset and confound checks

- Counts by source and class
- Resolution/format distributions
- Metadata-only classifier before and after normalization
- Group and duplicate audit results

### 7.2 Binary detection

- DCT-SVM and pretrained DE-FAKE
- Balanced accuracy, macro-F1, AUROC, AUPRC
- Per-generator fake recall
- OpenForensics-fake challenge with paired-real exclusion count

### 7.3 Primary eight-way attribution

- Top-1, balanced accuracy, macro-F1
- Per-class recall and support
- Confusion matrix using qualified display names
- Bootstrap interval and seed-sweep variation

### 7.4 Auxiliary nine-way classification

- Same metrics with one merged Real class
- Compare only as a sensitivity analysis; do not replace the primary result.

### 7.5 LOGO

- One row per held-out generator
- Forced-label distribution
- Mean confidence and entropy
- Rejection/false-known rates
- Family-level overlap patterns

### 7.6 End-to-end cascade

- Detection recall on known fakes
- Attribution accuracy conditional on detection
- End-to-end correct attribution
- Undetected fakes
- Per-generator end-to-end recall
- Real false positives and their forced generator labels

### 7.7 OpenForensics-fake

- Detection recall
- Forced attribution distribution after detection
- Confidence, entropy, and rejection behavior
- Explicit statement that no ordinary attribution accuracy exists for this unseen class

## 8. Discussion

- Which generator families overlap?
- Does closed-set performance survive the DCT gate?
- Which errors come from detection versus attribution?
- How does LOGO behavior qualify closed-set accuracy?
- Are observed patterns consistent across seeds and confidence intervals?

Avoid broad claims beyond face-centric generators and the exact checkpoints/settings tested.

## 9. Limitations

- SD1.5 img2img is London-only and strength-specific. Identity grouping prevents leakage but
  does not remove London-specific pose, lighting, background, or acquisition cues.
- Approximately 100 images per fake class yields small per-class test support.
- CLIP+BLIP may exploit semantic content rather than purely forensic traces.
- The generator set is face-centric and temporally/architecturally narrow.
- Hyperparameters are not exhaustively optimized.
- OpenForensics-fake is one manipulation benchmark, not universal OOS evidence.
- A closed-set head cannot identify an absent class without a rejection mechanism.

## 10. Conclusion

Answer RQ1-RQ3 directly, distinguish conditional from end-to-end results, and state whether the
eight generators are separable under the tested conditions.

## Appendix

- Full class counts and datasheets
- Complete confusion matrices and per-class intervals
- Raw/scaled confound comparisons
- Robustness perturbations if run
- GAN-fp results, clearly labeled Yu2019-inspired and optional
- Commands, run manifest, software versions, and AI-assistance disclosure
