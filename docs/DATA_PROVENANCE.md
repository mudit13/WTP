# Dataset provenance and limitations

This document records stable provenance, preparation rules, licensing constraints, and known unknowns. Run-specific sample counts, hashes, and exact file membership belong in the final run's data manifest and release record.

The previous dated provenance snapshot is retained under `docs/history/` for traceability. It does not override a newer data manifest.

## Shared processing policy

- `scripts/build_master_index.py` scans the configured sources and applies deterministic sampling using the configured seed.
- The headline preprocessing condition uses aspect-preserving resize followed by center crop and lossless PNG output at the configured size.
- JPEG augmentation is applied to training features only. Validation, test, OOS, and normal evaluation features remain clean.
- Attribution uses a stable group-aware train, validation, and test split. Detection comparisons reuse the same final test boundary where required.
- Source images and their img2img derivatives share a group identity.
- OpenForensics source relationships are preserved through sidecar metadata.
- Exact cross-split duplicates and group straddles are release-blocking findings.
- Perceptual similarity is diagnostic for aligned-face data and must be reviewed in context.

Exact preprocessing and split settings are defined in `configs/config.yaml` and recorded by each run manifest.

## Generated fake sources

### Stable Diffusion 1.5 text-to-image

- **Model family:** `runwayml/stable-diffusion-v1-5`.
- **Project role:** text-to-image fake source.
- **Generation record:** prompt, seed, inference steps, guidance, dimensions, scheduler, dtype, and model revision should be written to a generation manifest.
- **Historical limitation:** the original prepared dataset did not record every model-revision and scheduler detail. Preserve that uncertainty rather than reconstructing it from memory.
- **Confound risk:** a fixed portrait prompt family and square PNG output can create source-specific cues before shared normalization.

### Stable Diffusion 1.5 image-to-image

- **Source population:** neutral, front-facing Face Research Lab London Set images.
- **Project role:** source-preserving image-to-image fake class.
- **Grouping requirement:** every source image and all generated derivatives must share one group identity.
- **Generation record:** model revision, strength, steps, guidance, scheduler, prompt, negative prompt, seed, source path, and output path.
- **Confound risk:** London-only identities and a narrow studio prompt make the class visually constrained.
- **Quality policy:** artifacts are not selectively removed merely because they make the generation method look worse, unless a documented data-quality rule applies consistently.

### FLUX.1-schnell text-to-image

- **Model family:** `black-forest-labs/FLUX.1-schnell`.
- **Project role:** text-to-image fake source.
- **Generation record:** model revision, prompt, seed, inference steps, guidance behavior, dimensions, dtype, and cache path.
- **Historical limitation:** the original prepared dataset did not pin every model-revision detail.
- **Confound risk:** sibling images generated from the same prompt family may be semantically similar even when they are not duplicate files.

### StyleGAN3-FFHQ

- **Model family:** official NVIDIA StyleGAN3 FFHQ generator.
- **Project role:** unconditional GAN fake source.
- **Generation record:** external repository revision, checkpoint filename and SHA-256, seeds, truncation, noise mode, native resolution, and any resize operation.
- **Confound risk:** the generator is trained on the FFHQ manifold, so an FFHQ sensitivity analysis is required for attribution interpretation.

## DFFD-provided sources

DFFD supplies or aggregates several real and manipulated face sources used by the project. The mounted copy and its original documentation remain external. Do not infer missing upstream generation settings.

### FFHQ real

- **Origin:** FFHQ images distributed through the available DFFD structure.
- **Role:** real source and potential source-manifold overlap with StyleGAN3-FFHQ.
- **Known processing:** the project consumes the aligned/cropped files present in the mounted dataset.
- **Unknown:** the exact conversion history of the mounted DFFD copy may not be fully documented in this repository.

### PGGAN variants

- **Origin:** Progressive GAN family samples distributed through DFFD.
- **Role:** configured fake attribution classes.
- **Unknown:** exact checkpoint distinction, seeds, training subset, and synthesis settings unless supplied by the source documentation.
- **Interpretation risk:** closely related variants may have strong bidirectional confusion.

### StarGAN

- **Origin:** StarGAN face translation samples distributed through DFFD.
- **Role:** fake manipulation or translation source.
- **Unknown:** exact checkpoint and source-to-target domain for each mounted image unless recorded upstream.

### FaceApp

- **Origin:** commercial FaceApp manipulations distributed through DFFD.
- **Role:** fake manipulation source.
- **Unknown:** application version, selected filters, and exact source identity unless recorded upstream.
- **Interpretation risk:** this is a commercial manipulation class, not unconditional synthesis.

## Real sources

### Face Research Lab London Set

- **Role:** real source and input population for SD1.5 img2img.
- **Selected subset:** neutral, front-facing images as configured by the project.
- **Confound risk:** a homogeneous studio population can be separable from web-crawled or generated sources.
- **Redistribution:** confirm the dataset's academic-use and redistribution terms before sharing any image or derivative.

### CelebA

- **Role:** real source.
- **Origin:** aligned CelebA images available through the shared data location.
- **Confound risk:** web-crawled aligned faces differ in content and acquisition from London studio and FFHQ populations.
- **Unknown:** exact JPEG and alignment history of the mounted copy unless documented by the provider.

### OpenForensics real

- **Role:** real crops used with explicit source-scene grouping.
- **Preparation:** polygon or bounding-box crop from source annotations, deterministic selection, and recorded output settings.
- **Grouping requirement:** retain `source_image_id` or equivalent source-scene identity.
- **Confound risk:** raw crop size, format, or geometry can be predictive before shared normalization.

## External test-only source

### OpenForensics fake

- **Role:** unseen manipulation benchmark for detection and forced-label or rejection analysis.
- **Training policy:** it must not enter primary model fitting, validation selection, or the closed-set output taxonomy.
- **Grouping requirement:** related source photos must be excluded from conflicting training roles.
- **Interpretation limit:** it is one external manipulation benchmark, not evidence of universal unseen-generator generalization.

## Metadata and source confounds

The project explicitly measures whether labels can be predicted from metadata such as image dimensions, format, or crop geometry. Shared normalization can remove measured global format cues, but it cannot remove all source-content or semantic differences between datasets.

Every report should disclose:

- which real source populations support each fake source
- whether a generator was trained on or derived from a real source used in evaluation
- prompt-family restrictions
- resolution and format differences before normalization
- missing upstream generation details
- whether metadata-only probes remain above chance

## Run-specific provenance

A final release must include:

- canonical index used by the run
- data manifest with SHA-256 hashes or approved directory-level evidence
- generation manifests for project-generated datasets
- group maps
- split and leakage-audit outputs
- config hash
- checkpoint manifest

Counts and exact filenames should be read from those artifacts. Do not keep updating this stable document with each run's headline metrics.

## Redistribution policy

Do not commit or package raw DFFD, OpenForensics, London Set, CelebA, or FFHQ-derived images unless the applicable terms permit it. Do not redistribute supervisor-provided checkpoints or external generator weights without authorization.

A lightweight handover archive should contain code-independent evidence such as metric JSON, summary CSV, manifests, and logs, subject to path-sensitivity review. See [`RUNBOOK.md`](RUNBOOK.md) for the packaging procedure.

## Known unknowns to preserve

Unless a new generation or source manifest resolves them, preserve these uncertainties:

1. historical SD1.5 text-to-image and FLUX.1 revisions may not be pinned
2. exact DFFD generator checkpoints and settings may be unavailable
3. exact upstream JPEG and alignment histories may be incomplete
4. commercial manipulation versions and parameters may be unavailable
5. source documentation may be accessible on the server but not redistributable in Git

Unknown does not mean unimportant. It is a documented limitation and should remain visible in the final report.
