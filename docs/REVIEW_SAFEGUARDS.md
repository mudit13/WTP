# Review-derived scientific safeguards

This document preserves valid methodological requirements from earlier reviews. It does not
define the current class space; that authority belongs to the latest professor feedback and
`configs/config.yaml`.

## Dataset safeguards

- Diversify real data across London-DB, FFHQ, CelebA, and OpenForensics-real.
- Keep OpenForensics-fake test-only.
- Record generation model, revision/checkpoint, prompts, seeds, steps, guidance, dimensions,
  source images, preprocessing, and licenses.
- Treat SD1.5 img2img as a qualified single-domain condition:
  `SD1.5 img2img (London-DB, strength=0.6)`.
- Keep source-coupled images in one split through explicit sidecars.
- Audit exact and perceptual duplicates for datasets without identity metadata.

## Confound safeguards

- Measure metadata-only separability before claiming generator-trace learning.
- Use aspect-preserving resize plus center crop for headline experiments.
- Keep the scaled/squashed variant only as a confound comparison.
- Apply JPEG augmentation to training features only.
- Keep validation, test, OOS, and robustness evaluation clean except for the named perturbation.
- Use separate, content-hashed caches for clean and augmented features.

## Model safeguards

- The provided DE-FAKE checkpoint is binary only.
- Multi-class attribution is produced by this project's fine-tuned frozen-CLIP/BLIP head.
- DCT-SVM is linear and must reuse the shared fixed test boundary.
- Preserve the original source-specific attribution result and compare it with a matched
  source-specific run that removes FFHQ; do not replace or discard the earlier condition.
- LOGO must train only on the declared class space minus the held-out target.
- OpenForensics-fake must never enter model fitting, including optional appendix methods.
- GAN-fp must be described as Yu2019-inspired, not a byte-faithful reproduction.

## Evaluation safeguards

- Report AUROC/AUPRC for binary detection and balanced accuracy, macro-F1, and unweighted
  Cohen's kappa for binary and nominal multi-class classification.
- Bootstrap Cohen's kappa alongside the other headline metrics.
- Include per-class support, recall, confusion matrices, and bootstrap confidence intervals.
- Cluster bootstrap repeated img2img derivatives by identity.
- Distinguish conditional attribution from end-to-end cascade accuracy.
- Report detector misses as end-to-end attribution failures.
- For absent classes under LOGO/OOS, report forced-label distributions, confidence, entropy,
  and rejection performance rather than interpreting top-1 as an ordinary accuracy.
- Never reuse historical metrics after taxonomy, index, split, or leakage-control changes.
- Do not report Cohen's kappa for LOGO/OOS forced-label tests where the true class is absent
  from the output space; kappa is not informative for that design.

## FFHQ interpretation safeguards

- The official NVIDIA FFHQ source documents automatic dlib alignment/cropping to 1024x1024 PNG.
- Its reference alignment performs geometric resampling and may use reflected padding with
  Gaussian-blurred boundary blending.
- The official documentation does not identify a learned super-resolution stage. The meeting
  suggestion about super-resolution was explicitly uncertain and must not be repeated as fact.
- A performance change after removing FFHQ establishes sensitivity to the FFHQ class/training
  population, not a causal explanation from preprocessing alone.

## Reproducibility safeguards

- Require a unique run ID and record git commit/config hash.
- Pin generator revisions/checkpoints.
- Abort on missing classes, OOS/training overlap, group-map failure, or group straddling.
- Keep raw data, weights, environments, and generated results out of git.
- Preserve historical reasoning in `PROJECT_LOG.md`, but use `PIPELINE.md` for commands.
