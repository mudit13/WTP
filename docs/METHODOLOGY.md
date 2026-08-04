# Methodology and scientific safeguards

Scientific design for the pipeline. Operational commands belong in [RUNBOOK.md](RUNBOOK.md).

## Research objective

Two questions:

1. Can an image be detected as real or AI-generated?
2. For a detected generated image, can its source generator be attributed?

The system is a cascade — detection errors propagate into end-to-end attribution.

## Active pipeline

### Detection

The primary in-domain detector extracts DCT log-magnitude features and trains a linear SVM. Pretrained DE-FAKE is kept as a comparison baseline.

Report: AUROC, AUPRC, balanced accuracy at the operating point, macro-F1, Cohen's kappa, per-class recall, and confidence intervals.

### Attribution

Attribution uses DE-FAKE's CLIP (and optional BLIP-text) features with a project-trained MLP head. The primary task is fake-only classification among the generator classes declared in `configs/config.yaml`. A joint mode merging real sources into a single class is a sensitivity analysis, not a replacement.

### End-to-end cascade

A generated image missed by the detector counts as an end-to-end attribution failure. Report conditional attribution (among detector-passed samples) separately from full cascade performance.

## Dataset taxonomy

Generator and real-source names are defined in `configs/config.yaml`. Scripts, plots, and report text use those exact strings.

Class rules:

- Closed-set training uses only the declared in-set classes.
- OpenForensics-fake is a held-out external benchmark and never enters model fitting.
- The `in_set_generators` and `finetune_new_classes` config keys are read by `attribution_taxonomy.py`; their union must equal `fake_generators` (tests enforce this).

## Preprocessing

The headline condition is aspect-preserving resize and center crop to the configured resolution. A scaled variant may be retained for confound comparison but does not replace the headline geometry.

Training-only augmentation may include JPEG compression over the configured range. Validation, test, OOS, and standard evaluation data stay clean unless a named robustness perturbation is being evaluated. Clean and augmented feature caches have separate, content-derived identities and are never shared across conditions.

## Split and leakage safeguards

Related observations must stay on one side of the split:

- multiple SD1.5 img2img derivatives of the same source image
- OpenForensics real and fake crops from related source material
- exact duplicates
- repeated observations from the same identity

Comparative models reuse the same fixed test boundary. Model-specific random resplitting is not used for headline claims.

The final run fails when any of the following is true:

- a group appears in more than one split
- an exact duplicate crosses a split boundary
- a required group map is missing
- an OOS target appears in training
- the leakage audit reports a non-zero hard-gate count

## Confound safeguards

Attribution can accidentally learn source-dataset, format, or preprocessing differences rather than generator traces. Guards:

- metadata-only confound probes
- consistent headline geometry
- train-only JPEG augmentation
- matched test populations for model comparison
- FFHQ inclusion/exclusion sensitivity analysis
- robustness evaluation under blur, resize, and compression

A high closed-set score is not by itself evidence of a content-independent generator fingerprint.

## Model-specific notes

**DE-FAKE**: The supervisor-provided checkpoint is a binary real/fake model. Multi-class attribution comes from this project's separately trained head. These are different tasks and must not be described as one.

**DCT-SVM**: Remains linear and uses the shared fixed test boundary. Threshold selection does not inspect final test labels.

**LOGO**: Training excludes the held-out generator from both the population and the output class space. The evaluation asks how an unseen source gets forced into known classes — not whether it is correctly assigned.

**GAN-fingerprint methods**: The handcrafted and CNN paths are Yu2019-inspired project implementations. Unless a byte-faithful reproduction is separately established, they must not be described as one.

## Evaluation

For closed-set classification, report: balanced accuracy, macro-F1, Cohen's kappa, per-class support and recall, confusion matrix, and bootstrap CIs for headline metrics. For detection, also report AUROC and AUPRC with the threshold-selection rule stated.

For OOS and LOGO, the true generator is absent from the output space, so top-1 accuracy and Cohen's kappa are not meaningful. Report instead: forced-label distribution, maximum confidence, predictive entropy, and rejection curves.

Paired model comparisons align predictions on the same observations.

## FFHQ interpretation

A performance change after removing FFHQ shows sensitivity to that class or population. It does not establish a causal preprocessing mechanism. The project does not claim an undocumented super-resolution step in the FFHQ pipeline.

## Reproducibility

Every reportable run records: a unique run ID, git commit, config hash, data and checkpoint manifests, environment locks, leakage-audit evidence, and a completed release manifest. See [RUNBOOK.md](RUNBOOK.md) for the exact sequence.

A fixed seed stabilizes splits and augmentation but does not guarantee bitwise-identical GPU training. Multi-seed sensitivity and confidence intervals quantify numerical uncertainty more honestly than a claim of exact reproducibility.

## Interpretation limits

These claims are not supported by this pipeline:

- universal AI-image detection from one in-domain benchmark
- universal generator attribution from closed-set source classification
- OOS accuracy when the unseen class is absent from the label space
- causal preprocessing explanations from a single ablation
- reuse of metrics after taxonomy, split, data, code, checkpoint, or cache changes

Negative OOS or robustness findings are valid experimental results and should not be hidden or relabeled as failures.
