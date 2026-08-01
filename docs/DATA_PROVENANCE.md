# Authoritative dataset provenance

Run: `2026-08-01_eightway_v1` · core commit `141da128` · 2,154 indexed images
(1,132 fake, 1,022 real).

This document replaces the unresolved manual fields in the auto-generated `datasheets.md`.
Unknown upstream details are stated explicitly rather than inferred.

## Shared processing

- `build_master_index.py` scans configured sources and applies deterministic sampling with seed 42.
- Headline preprocessing uses aspect-preserving resize of the shortest side to 256 followed by
  center crop and lossless PNG export.
- JPEG q30–100 augmentation is applied to training features only. Validation/test features remain
  clean.
- Attribution uses a content-stable 70/10/20 train/validation/test split. DCT uses the same 20%
  test boundary and the remaining 80% for training.
- The corrected integrity audit reports zero explicit group straddles and zero exact cross-split
  duplicates. dHash similarities are diagnostic only for aligned face data.

## Generated fake classes

### SD1.5 txt2img

- **Count/format:** 108 images, 512×512 PNG.
- **Model:** `runwayml/stable-diffusion-v1-5`.
- **Parameters:** 40 steps, CFG 8.5, 512×512, float16, 9 portrait prompts × 12 seeds.
- **Seeds:** `prompt_index * 1000 + seed_index`.
- **Prompt design:** studio/outdoor portrait variations; one shared negative-prompt block.
- **Processing:** generated directly at 512²; no source crop; later converted to the shared
  aspect-256 variant.
- **Citation:** Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models*.
- **Unknown:** the Hugging Face revision and scheduler were not pinned/logged for this historical
  generation run.
- **Limitation:** fixed prompt family and native square PNG format create pre-normalization
  source cues.

### SD1.5 img2img

- **Count/format:** 108 images, 512×512 PNG, from 102 London-DB identities.
- **Inputs:** every London-DB `neutral_front` image; six identities receive a second seeded output.
- **Model:** `runwayml/stable-diffusion-v1-5`, revision
  `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`.
- **Parameters:** strength 0.6, 40 steps, CFG 8.5, 512×512, seeds 200000–200107,
  `PNDMScheduler`, torch 2.1.0+cu118.
- **Prompt:** one fixed frontal studio portrait prompt with neutral expression, even lighting and
  grey backdrop; negative prompt stored verbatim in `generation_manifest.json`.
- **Input processing:** EXIF transpose → RGB → aspect-preserving center crop to 512.
- **Grouping:** `londondb_img2img_groups.csv` keeps each source real and all derivatives under one
  identity ID.
- **Citations:** Stable Diffusion v1.5 and Face Research Lab London Set.
- **Limitations:** London-only identities and one fixed studio prompt make this class visually
  narrow. Persistent eye distortions were retained rather than cherry-picked.

### FLUX.1-schnell txt2img

- **Count/format:** 108 images, 512×512 PNG.
- **Model:** `black-forest-labs/FLUX.1-schnell`.
- **Parameters:** 4 inference steps, guidance 0.0, bfloat16, CPU offload, 9 prompts × 12 seeds.
- **Prompts:** the same nine portrait prompts used for SD1.5 txt2img.
- **Important detail:** the negative prompt is recorded in metadata but is not passed to the
  FLUX pipeline call.
- **Unknown:** model revision was not pinned in the historical generator script.
- **Limitation:** prompt siblings across seeds are visually similar; this is not an exact-file
  leak.

### StyleGAN3-FFHQ

- **Count/format:** 108 images, stored as 512×512 PNG.
- **Model:** official NVIDIA NGC `stylegan3-r-ffhq-1024x1024.pkl` (`G_ema`).
- **Parameters:** unconditional seeds 0–107, truncation ψ=0.7, constant noise.
- **Processing:** generated at native 1024² and resized to 512² with Lanczos interpolation.
- **Citation:** Karras et al., *Alias-Free Generative Adversarial Networks*; FFHQ training source.
- **Unknown:** the Git commit of the server-side StyleGAN3 clone and checkpoint byte hash were not
  recorded.
- **Limitation:** StyleGAN3 is trained on the FFHQ manifold, directly motivating the FFHQ
  sensitivity diagnostic.

## DFFD-provided classes

All DFFD paths come from `/share/DeepFake/DFFD_Images`. The project log records that the DFFD
bundle uses RetinaFace-aligned face images and carries CC BY-NC-SA 4.0 terms. The original DFFD
`readme.txt` is not vendored, so exact upstream synthesis/manipulation settings must not be
invented.

### FFHQ real

- **Count/format:** deterministic sample of 300 from approximately 9,000 test images; 299×299 PNG.
- **Origin:** FFHQ Flickr faces distributed through DFFD.
- **Processing:** DFFD-provided aligned/cropped 299² images. The exact 1024→299 conversion is not
  documented in this repository.
- **Official FFHQ history:** dlib alignment/cropping to 1024² PNG; no documented learned
  super-resolution stage.
- **Limitation:** source manifold overlaps StyleGAN3-FFHQ.

### PGGAN-v1 and PGGAN-v2

- **Count/format:** 100 images each, deterministic samples from DFFD test folders; 299×299 PNG.
- **Origin:** Progressive GAN family (Karras et al.) as distributed by DFFD.
- **Unknown:** exact v1/v2 checkpoint distinction, training subset, seeds and synthesis parameters.
- **Limitation:** the two classes show strong bidirectional confusion and may represent highly
  similar generator conditions.

### StarGAN

- **Count/format:** 100 test images, 299×299 PNG.
- **Origin:** StarGAN multi-domain face translation via DFFD.
- **Unknown:** exact checkpoint and source/target domain for each image.
- **Limitation:** manipulation/translation rather than unconditional synthesis.

### FaceApp

- **Count/format:** 100 test images, 299×299 PNG.
- **Origin:** commercial FaceApp manipulations of FFHQ-derived faces via DFFD.
- **Unknown:** application version, filters and exact source identities.
- **Limitation:** commercial manipulation rather than pure synthesis; shares the FFHQ content
  manifold.

## Real sources

### London-DB

- **Count/format:** all 102 `neutral_front` images, 1350×1350 JPEG.
- **Origin:** Face Research Lab London Set (DeBruine & Jones).
- **Processing:** narrow frontal studio subset; no project-side preprocessing before indexing.
- **Unknown:** JPEG quality and acquisition details beyond the dataset documentation.
- **Limitation:** highly homogeneous studio source and the sole basis for SD1.5 img2img.

### CelebA

- **Count/format:** deterministic sample of 320 from approximately 202,599 aligned images;
  178×218 JPEG.
- **Origin:** CelebA aligned-face distribution accessed through the shared DFFD location.
- **Citation:** Liu et al., *Deep Learning Face Attributes in the Wild*.
- **Unknown:** original JPEG quality and the exact upstream alignment history represented by the
  server copy.
- **Limitation:** web-crawled aligned faces differ strongly from London studio and FFHQ sources.

### OpenForensics real

- **Count/format:** 300 variable-size JPEG face crops.
- **Source:** OpenForensics polygon annotations, host source under
  `/vol1/share/DeepFake/OpenForensics`.
- **Documented extraction protocol:** COCO polygon bounding-box crop, Val split, deterministic
  cap 300, seed 42, JPEG q95. The bundle contains the resulting crops/index and group sidecar,
  not the original host extraction log.
- **Grouping:** `source_image_id` records the source scene for coupling control.
- **Unknown:** exact annotation IDs are not listed in the report bundle.
- **Limitation:** variable crop geometry retains some raw metadata separability; the headline
  aspect-normalized variant removes the global size/format cue.

## External test-only source

### OpenForensics-fake

- **Count/format:** 300 variable-size JPEG q95 manipulated-face crops.
- **Extraction:** documented as the same annotation/crop pipeline as OpenForensics real;
  category 1 manipulated faces.
- **Use:** never trained by the primary eight-way head or DCT OOS model; force-scored only.
- **Grouping:** source-photo IDs prevent paired real crops from entering DCT OOS training.
- **Unknown:** manipulation subtype granularity and exact annotation list.
- **Limitation:** one manipulation benchmark, not evidence for universal unseen-generator
  generalization.

## Measured cross-dataset confounds

- Raw format/size metadata is predictive: balanced accuracy 0.755, AUROC 0.857.
- After aspect normalization, metadata-only performance is chance: balanced accuracy/AUROC 0.5.
- OpenForensics raw crop geometry remains somewhat predictive: balanced accuracy 0.608,
  AUROC 0.634.
- These controls remove measured format/geometry cues but cannot remove source-content or
  semantic differences between FFHQ, CelebA, London and text-generated faces.

## Known unknowns to preserve in the report

1. SD1.5 txt2img and FLUX revisions were not pinned.
2. Exact DFFD generator checkpoints/settings are unavailable in the repository.
3. Exact sampled filename lists are reproducible from seed/config but are not bundled separately.
4. London/CelebA JPEG parameters are unknown.
5. DFFD RetinaFace/alignment details rely on the server readme summary, not a vendored source file.
