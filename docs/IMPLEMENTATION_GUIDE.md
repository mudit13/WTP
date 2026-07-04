# Implementation guide (start here)

A single onboarding doc for the team. Read this first; it points to the detailed docs for each
part. Goal of the project: build and evaluate AI-image **detection** (real vs fake) and
**attribution** (which generator), scientifically and aligned with the interim "GOLD" review.

## 1. The big picture

- **RQ1 - Detection:** how well do a CLIP-based detector (DE-FAKE) and a frequency detector
  (DCT-SVM) separate real photos from AI-generated faces, including unseen ("out-of-set")
  generators?
- **RQ2 - Attribution:** how well can we identify the *source generator*, in-set vs out-of-set?
  (The pretrained DE-FAKE head is binary only, so attribution comes from our own fine-tuned
  head + a reproduced GAN-fingerprint method.)

Two things shape every experiment, both from the GOLD review:
1. The **real class** must be diverse and balanced (not just one narrow dataset).
2. We must not let **preprocessing (size/format/compression)** leak the label - so we normalize
   and JPEG-augment, and report raw vs controlled.

## 2. One-time setup (on the server)

```bash
# connect (EduVPN first), then enter the container
ssh pitsec_sose26_topic8@gensynth.cs.uni-magdeburg.de
sudo pitsec_sose26_topic8.docker PITSEC26

# the repo IS /pitsec_sose26_topic8 ; get/refresh it there
cd /pitsec_sose26_topic8 && git pull          # first time: git clone ... .
cp configs/paths.example.env configs/paths.env
export $(grep -v '^#' configs/paths.env | xargs)
```

Details: `docs/SERVER_WORKFLOW.md`. Environments/venvs: `docs/ENVIRONMENTS.md`.

## 3. The golden rules (please follow - 5 people, shared repo)

- **Interpreter:** use `$WTP_PY_DEFAKE` (= venv_sd15) for all analysis scripts. Never bare
  `python`. Generation uses its own per-generator venv.
- **Run on variants:** experiments run on `index_scaled.csv` AND `index_cropped.csv` (the
  scaling-vs-cropping study).
- **Raw vs controlled:** always produce both a raw and a JPEG-augmented result, with DISTINCT
  cache/output paths. Defaults: DCT is raw unless `--jpeg_aug`; fine-tune/LOGO are controlled
  by default (`--jpeg_aug auto` -> config), use `--jpeg_aug off` for the raw baseline.
- **Report balanced metrics** (AUROC, balanced accuracy, macro-F1) - the classes are balanced
  but never assume; see `scripts/lib/metrics.py`.
- **Don't commit** datasets, model weights, venvs, or `paths.env` (already git-ignored).

## 3b. Working together (git, 5 people)

- **Don't all push to `main`.** Each person works on a branch per workstream, e.g.
  `git checkout -b ws3-dct`, then open a Pull Request for review before merging.
- **Pull before you start**, merge small and often, to avoid big conflicts.
- **Server import = `git pull` in `/pitsec_sose26_topic8`.** Only tracked code moves; datasets,
  models, and venvs are git-ignored and stay put. Don't run experiments on uncommitted code -
  commit/push first so results are reproducible.
- Line endings are normalized to LF via `.gitattributes` (we dev on Windows, run on Linux) -
  no action needed, just don't override it.
- Citations live in `CITATIONS.md` - add any new source you introduce.

## 4. The workstreams (how the work splits)

Each is a self-contained chunk; exact commands are in `docs/PIPELINE.md` (step numbers match).

| WS | Topic | Main scripts | Output |
|----|-------|--------------|--------|
| WS1 | Data + index + datasheets (diverse, balanced real class) | `build_master_index.py`, `make_datasheets.py` | `master_metadata.csv`, datasheets |
| WS2 | Preprocessing variants (scale vs crop, common size 256, PNG) | `prepare_variants.py` | `index_scaled.csv`, `index_cropped.csv` |
| WS3 | Detection: DE-FAKE (binary) + DCT-SVM, raw vs controlled | `run_defake_batch.py`, `score_defake_detection.py`, `dct_extract_features.py`, `dct_svm.py` | detection metrics |
| WS4 | Attribution (DE-FAKE multi-class; GAN-fp PARKED on `ganfp-integrated`) | `eval_defake_attribution.py` | attribution metrics |
| WS5 | Fine-tune the attribution head + leave-one-generator-out | `finetune_defake_head.py`, `leave_one_generator_out.py` | trained head, LOGO results |
| WS6 | Out-of-set analysis (forced labels, confidence, entropy) | `out_of_set_analysis.py` | OOS report |
| WS7 | Robustness (JPEG/blur/resize/sharpen on held-out test) | `make_split.py`, `robustness_perturb.py` | robustness drops |
| WS8 | Aggregate everything for the report | `aggregate_results.py` | `REPORT_SUMMARY.md` |

(The team divides these among the 5 members; WS1->WS2 should land first since others build on
the indices.)

## 5. Where to read more

- `docs/PIPELINE.md` - exact command-by-command run order (copy/paste).
- `docs/GOLD_ALIGNMENT.md` - how each review point maps to code.
- `docs/PROJECT_LOG.md` - what we changed and why (decisions/history).
- `docs/SERVER_WORKFLOW.md` - connecting, paths, sync.
- `docs/ENVIRONMENTS.md` - the three venvs + the "never move a venv" rule.
- `docs/DATASHEET_TEMPLATE.md` - dataset provenance (GOLD requirement).
- `report/REPORT_OUTLINE.md` - the report structure and what evidence goes where.
- `docs/OPEN_QUESTIONS.md` - local-only notes on items still pending the supervisor.

## 6. Current status (snapshot)

- Generation (SD1.5, FLUX.1, StyleGAN3) + initial DE-FAKE inference: DONE on the server.
- Index built (926 images); real class rebalanced in config (London-DB + FFHQ + CelebA).
- Confound controls in place (common size 256 + JPEG augmentation, wired into the code).
- Pending supervisor: GAN-fingerprint approach, OpenForensics (optional), report date.
