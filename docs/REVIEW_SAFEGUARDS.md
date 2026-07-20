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
- LOGO must train only on the declared class space minus the held-out target.
- OpenForensics-fake must never enter model fitting, including optional appendix methods.
- GAN-fp must be described as Yu2019-inspired, not a byte-faithful reproduction.

## Evaluation safeguards

- Report AUROC/AUPRC for binary detection and balanced accuracy/macro-F1 for classification.
- Include per-class support, recall, confusion matrices, and bootstrap confidence intervals.
- Cluster bootstrap repeated img2img derivatives by identity.
- Distinguish conditional attribution from end-to-end cascade accuracy.
- Report detector misses as end-to-end attribution failures.
- For absent classes under LOGO/OOS, report forced-label distributions, confidence, entropy,
  and rejection performance rather than interpreting top-1 as an ordinary accuracy.
- Never reuse historical metrics after taxonomy, index, split, or leakage-control changes.

## Reproducibility safeguards

- Require a unique run ID and record git commit/config hash.
- Pin generator revisions/checkpoints.
- Abort on missing classes, OOS/training overlap, group-map failure, or group straddling.
- Keep raw data, weights, environments, and generated results out of git.
- Preserve historical reasoning in `PROJECT_LOG.md`, but use `PIPELINE.md` for commands.
