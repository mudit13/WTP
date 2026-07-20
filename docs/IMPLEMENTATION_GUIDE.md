# Implementation guide

## Project objective

The professor-facing system has two stages:

1. **Detection:** log-DCT features with a linear SVM classify real versus fake. The provided
   pretrained binary DE-FAKE model is the baseline.
2. **Attribution:** images detected as fake are classified by a fine-tuned DE-FAKE head into
   eight fake generators.

The primary attribution classes are:

- SD1.5 txt2img
- SD1.5 img2img (London-DB, strength=0.6)
- FLUX.1-schnell
- StyleGAN3-FFHQ
- FaceApp
- PGGAN-v1
- PGGAN-v2
- StarGAN

An auxiliary nine-way model adds one merged `real` class sampled evenly from London-DB, FFHQ,
CelebA, and OpenForensics-real. OpenForensics-fake is test-only in every trainer.

## Experimental outputs

The default orchestrator produces:

- Preprocessing and metadata-confound measurements
- Pretrained binary DE-FAKE baseline
- DCT-SVM detection and OpenForensics-fake challenge
- Primary eight-way attribution
- Auxiliary nine-way attribution
- Eight fake-generator LOGO folds
- End-to-end DCT-to-attribution cascade metrics
- OpenForensics-fake confidence/entropy analysis
- One aggregated report summary

GAN-fp and robustness are optional appendix stages.

## Scientific invariants

- Use the aspect-preserving 256-pixel variant for headline results.
- JPEG augmentation applies to training features only; validation and test remain clean.
- Use content-stable, source-group-aware splits.
- London real images and their img2img derivatives share one identity group.
- OpenForensics real/fake crops sharing a source photo never cross split boundaries.
- OpenForensics-fake never enters DCT, DE-FAKE, LOGO, or GAN-fp training.
- Report balanced accuracy, macro-F1, per-class recall, and uncertainty.
- Treat OOS/LOGO top-1 as undefined or zero by construction; report forced labels and rejection.
- Use immutable `results/<run_id>/` directories and preserve `run_manifest.json`.

## Repository roles

```text
configs/                  Dataset definitions, taxonomy, experiment defaults
scripts/                  Command-line entry points
scripts/lib/              Shared taxonomy, features, splitting, metrics, IO
tests/                    CPU-safe regression tests
docs/PIPELINE.md          Only active command runbook
docs/PROJECT_LOG.md       Historical decisions and debugging evidence
report/REPORT_OUTLINE.md  Current report plan
De-Fake-patched/          Vendored upstream DE-FAKE reference/package
results/<run_id>/         Generated immutable experiment evidence
```

## Development and execution

- Edit and test locally.
- Run `python -m pytest -q` and `python -m compileall -q scripts tests`.
- Commit and push before server execution so the run manifest can identify a git commit.
- On the server, use `$WTP_PY_DEFAKE`; never use bare `python`.
- Follow `docs/PIPELINE.md` exactly.

See `docs/README.md` for document authority and `docs/SERVER_WORKFLOW.md` for server paths.
