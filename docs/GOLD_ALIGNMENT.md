# GOLD alignment - interim review feedback -> implementation

The interim review (treated as the "GOLD" standard) is the authority for scope. This file
traces each piece of guidance to where it is implemented in this repo, so it is clear that the
project follows the review rather than the older interim plan. (The raw review transcript is
kept by the team, not committed.)

## Summary table

| # | GOLD guidance | What we do | Where |
|---|---------------|------------|-------|
| 1 | Real class is too narrow (London-DB only) - diversify | Real class = London-DB (studio) + FFHQ (Flickr) + CelebA (web) + OpenForensics (in-the-wild, capped 300); OpenForensics reals added per Dennis's strongest steer | `configs/config.yaml` (datasets, attribution.real_generators) |
| 2 | Avoid learning preprocessing artifacts; study scaling vs cropping | Scaled (squash), center-crop, AND aspect-preserving variants (aspect = confound-controlled default), lossless PNG, common size 256; raw-vs-normalized reporting + metadata-only confound probe | `scripts/prepare_variants.py`, `scripts/lib/image_ops.py`, `scripts/metadata_confound_probe.py`, `configs/config.yaml` |
| 2b| Compression/format must not leak the label | Uniform random JPEG augmentation on all classes at training | `configs/config.yaml` (augmentation), `image_ops.make_jpeg_augmenter`, `dct_extract_features --jpeg_aug`, `finetune/LOGO --jpeg_aug` |
| 3 | Retraining/fine-tuning DE-FAKE is allowed/encouraged | Freeze CLIP, fine-tune a small head to ADD generators as real classes | `scripts/finetune_defake_head.py`, `scripts/lib/defake_head.py` |
| 4 | Test out-of-set generalization (unseen generators) | Leave-one-generator-out + out-of-set confidence/entropy analysis | `scripts/leave_one_generator_out.py`, `scripts/out_of_set_analysis.py` |
| 5 | Quantify closed-set limitation (forced labels) | Forced-label distribution, false-known rate, predictive entropy under LOGO | `scripts/leave_one_generator_out.py`, `scripts/lib/metrics.py` |
| 6 | Robustness to perturbations | JPEG/blur/resize/sharpen on held-out test; performance/confidence drop | `scripts/robustness_perturb.py`, `scripts/make_split.py` |
| 7 | Be scientifically correct in metrics | AUROC/AUPRC, balanced accuracy, macro-F1, confusion matrices, entropy | `scripts/lib/metrics.py`, `scripts/score_defake_detection.py` |
| 8 | Document dataset provenance / processing history | Datasheet template + auto-filled measurable fields; DFFD provenance captured | `docs/DATASHEET_TEMPLATE.md`, `scripts/make_datasheets.py`, `docs/PROJECT_LOG.md` |
| 9 | Email the supervisor on any roadblock | Consolidated open questions + ready email drafts | `docs/OPEN_QUESTIONS.md` (local) |

## Notes on the items that changed scope vs the interim plan

- **Attribution (RQ2).** The interim plan implied using a pretrained DE-FAKE attribution head.
  Server inspection confirmed the provided head is BINARY only (real/fake) - there is no
  pretrained multi-class attribution. So attribution comes from our fine-tuned head (item 3),
  and the report frames RQ2 as "can a fine-tuned head attribute in-set vs out-of-set".

- **GAN-Fingerprints (Yu2019-inspired; on main).** The original repo is code-only on a
  deprecated stack (Chainer/cupy/CUDA 10), so we re-implement the METHOD in PyTorch (residual/
  spectrum features + a learned CNN with a fixed Fridrich-Kodovsky SRM front-end) as a second
  attribution method beside DE-FAKE - honestly "Yu2019-inspired", not a byte-faithful port. It
  was briefly parked while DE-FAKE multi-class took priority, then consolidated back onto `main`
  in complete form (both paths + benchmark). DE-FAKE multi-class remains the method of record;
  GAN-fp is the complementary GAN-trace attribution.

- **Preprocessing study is now central, not cosmetic.** Inspecting the data showed format and
  resolution almost perfectly predict the label (reals include JPEG; fakes are all PNG;
  resolutions separate classes). The scaling-vs-cropping + JPEG-augmentation work (items 2/2b)
  is therefore essential to a fair result, and we report raw vs normalized to show the effect.

## How to verify alignment quickly

- Real-class composition: `configs/config.yaml` -> `datasets` (label: real) and
  `attribution.real_generators`.
- Confound controls: `configs/config.yaml` -> `common_size: 256`, `augmentation`.
- Experiment entry points: `docs/PIPELINE.md` lists the exact run order (detection,
  attribution, LOGO, out-of-set, robustness, aggregation).
