# Project log - changes and the reasons behind them

Topic 8: AI Image Detection & Attribution. Code is authored locally and executed inside the
GPU container; the repo root maps to the container project root (`/pitsec_sose26_topic8`).
This log explains WHAT changed and WHY, so anyone (team or examiner) can follow the reasoning.
For how each change maps to the interim "GOLD" review, see `docs/GOLD_ALIGNMENT.md`.

---

## 1. Unified the repository (one repo, server-importable)

**What:** Merged the analysis/experiments layer with the team's existing server code
(generation + DE-FAKE inference) into a single tree: shared library in `scripts/lib/`, all
entry points in `scripts/`, vendored DE-FAKE in `De-Fake-patched/`, docs in `docs/`. Flattened
the nested `server/` clone, dropped the empty `main.py`, and folded the separate
`update_master_index_dffd.py` into the config-driven `build_master_index.py`.

**Why:** The server layout was flat and brittle - loose scripts at the root, hardcoded
absolute paths everywhere, a vendored dependency pinned by an absolute `sys.path`, and a
`.gitignore` with unresolved merge-conflict markers. A single organized repo can be pushed to
GitHub and `git pull`-ed onto the server, keeping future work maintainable.

## 2. Centralized the data schema (`scripts/lib/schema.py`)

**What:** One module defines the canonical column names of `master_metadata.csv`
(`filename, full_path, label, generator, category, source_dataset, width, height`) and the
DE-FAKE prediction columns (`defake_predict, prob_real, prob_fake, blip_caption`), plus
helpers `is_fake_label` / `is_fake_predict`.

**Why:** Every script previously risked drifting from the real CSV columns. A single source of
truth prevents silent column-name bugs and keeps the analysis layer compatible with the
team's existing pipeline outputs.

## 3. Config + path handling aligned to the real server

**What:** `configs/config.yaml` mirrors the actual dataset layout, generator names, and DFFD
sources; `configs/paths.example.env` lists the three generation venvs and the DE-FAKE
interpreter (`venv_sd15`). The moved DE-FAKE scripts (`run_defake_batch.py`,
`run_defake_dffd.py`, `merge_predictions.py`) now read their paths from the environment with
the original absolute values as defaults (behaviour unchanged if the env is not sourced).

**Why:** Removes hardcoded paths so the repo location is flexible and reproducible, while not
changing the logic of scripts the team already ran successfully.

## 4. .gitignore + privacy hygiene

**What:** Replaced the conflicted `.gitignore` with one canonical file that ignores datasets,
model weights, all three venvs, results/logs, and secrets. Replaced the supervisor's name in
all committed files with "the supervisor", and kept internal coordination notes in
`docs/OPEN_QUESTIONS.md` (git-ignored, local only).

**Why:** Keep large/non-relocatable artifacts and any internal/identifying content out of a
GitHub repo; venvs in particular must never be committed (they hardcode absolute paths).

## 5. Server inspection 

We inspected `/workspace`, `/share`, and `/pitsec_sose26_topic8` to resolve open questions
ourselves rather than asking blindly. Findings drove the decisions below.

- **DE-FAKE is binary-only.** `models/` has `clip_linear.pt` (binary real/fake head) +
  `finetune_clip.pt`; there is no pretrained multi-class attribution head. -> Attribution must
  come from our own fine-tuned head (RQ2 framed accordingly).
- **GANFingerprints repo = code only, deprecated stack** (Chainer/cupy/CUDA 10), built for its
  own GANs. -> We will REPRODUCE the fingerprint method ourselves in PyTorch on our generators
  (residual/spectrum fingerprints + a small learned classifier) instead of running the legacy
  code. Less risk, and it covers our actual generators.
- **GANDCTAnalysis = code only.** -> We train the DCT detector ourselves (as planned).
- **Plenty of data on the server.** DFFD has full train/val/test per generator (thousands of
  images) and `img_align_celeba` = 202,599 real faces. -> Attribution data is not a bottleneck;
  we can scale beyond the 100/generator originally sampled.

## 6. Fixed the real-class imbalance + narrowness (key scientific fix)

**What:** The index was 724 fake vs only 202 real, with just 2 real sources (London-DB 102 +
FFHQ 100), both aligned faces. `configs/config.yaml` now uses London-DB 102 (studio) + FFHQ
100->300 (Flickr) + CelebA 320 (web) ~= 722 real vs 724 fake, across three distinct capture
conditions. We report balanced metrics (AUROC, balanced accuracy).

**Why:** A 78/22 split lets a trivial "always fake" baseline score 78%, and a single narrow
real source risks the detector learning "studio portrait" instead of "real". Balancing and
diversifying the real class is required for a fair, defensible evaluation (GOLD concern #1).

## 7. Controlled the format/resolution confound (most important data finding)

**What:** Inspecting pixels showed format/size almost perfectly predict the label before any
model sees content: reals include JPEG (CelebA, London-DB) while every fake is PNG, and
resolutions separate classes (512=>fake, 1350/178=>real; only 299 has both). Decisions, now in
`configs/config.yaml`:
- `common_size: 256` (down from 512) - mostly downscales, avoiding the heavy upscaling
  artifacts that 512 would inject into CelebA (178) and DFFD (299).
- `augmentation: { jpeg_train: true, jpeg_quality_range: [30,100] }` - random JPEG compression
  applied to ALL classes during detector/attribution training.
- Plan to report RAW vs NORMALIZED/augmented to demonstrate the confound's effect.

**Why:** A frequency detector could score ~99% by learning compression + resolution rather
than generator traces - exactly the failure the GOLD review warned about. Re-encoding JPEG->PNG
does NOT remove baked-in JPEG artifacts, so uniform JPEG augmentation is needed so
"has-been-compressed" no longer leaks the label (Frank 2020 / Wang 2020).

**Code wiring (this is live, not just config):**
- `scripts/lib/image_ops.py`: `make_jpeg_augmenter(quality_range, seed)` - per-path
  deterministic random JPEG (reproducible, order-independent).
- `scripts/lib/clip_features.py`: `extract_features(..., augment=...)` applies it before CLIP.
- `scripts/lib/features_cache.py`: `build_features(..., jpeg_aug, jpeg_quality_range, seed)`;
  a cache built with a different `jpeg_aug` setting is not silently reused.
- `scripts/dct_extract_features.py`: `--jpeg_aug` (+ `--jpeg_qmin/qmax/seed`) for the DCT path.
- `scripts/finetune_defake_head.py` and `scripts/leave_one_generator_out.py`: `--jpeg_aug
  {auto,on,off}`, defaulting to `auto` = follow `config.augmentation.jpeg_train`.

## 8. Datasheet provenance captured

**What:** From the DFFD `readme.txt` (Dang et al., CVPR 2020): reals = FFHQ + CelebA; fakes =
FaceApp (FFHQ-derived), PGGAN, StarGAN, StyleGAN; all face-aligned (RetinaFace); license
CC BY-NC-SA 4.0 (cite sources). FaceApp is a manipulation, not pure synthesis.

**Why:** The GOLD review requires processing history per dataset so we can argue the detector
learns generator traces, not preprocessing artifacts. Recorded for `docs/DATASHEET_TEMPLATE.md`.

---

## Open items still needing the supervisor

See `docs/OPEN_QUESTIONS.md` (local). In short: (A) confirm reproducing GAN-Fingerprints
ourselves instead of the legacy repo; (B) whether CelebA+FFHQ+London-DB reals suffice or
OpenForensics is also wanted; (C) the report submission date.
