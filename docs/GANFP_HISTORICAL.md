# GAN-fp historical workstream

This document preserves the method-development history of the optional GAN-fingerprint
workstream. It is not an active result source for the professor-aligned experiment.

## What was built

- Path A: luminance residual and FFT-spectrum features, train-only standardization/PCA, and an
  MLP classifier.
- Path B: a Yu2019-inspired CNN with a frozen SRM-family high-pass front-end and learned
  convolutional blocks.
- Shared content-stable, group-aware train/validation/test splitting.
- Optional same-split comparison with DE-FAKE and DCT through
  `scripts/benchmark_attribution.py`.

The implementation is inspired by Yu et al. (2019); it is not a byte-faithful reproduction.
The SRM filter bank is a family-level reconstruction of Fridrich and Kodovsky (2012).

## Why old metrics are omitted

Prototype runs used superseded class spaces, dataset sizes, and split populations, including
separate real-source classes that are not part of the primary eight-way task. Their numerical
results are therefore not comparable with the authoritative run
`2026-08-01_eightway_v1` and must not appear as final evidence.

Git history and `docs/PROJECT_LOG.md` preserve the original prototype numbers and debugging
history if process provenance is required.

## Current status

GAN-fp is optional appendix work. The professor-facing core is:

1. DCT-SVM real/fake detection
2. Eight-way DE-FAKE attribution
3. LOGO and OpenForensics-fake generalization analysis
4. End-to-end cascade evaluation

No current report claim depends on GAN-fp. If it is run again, it must use the same eight fake
classes, the same test identities, and a new immutable result directory. Residual/high-pass
features reduce direct semantic dependence but are not guaranteed to be completely content-free.
