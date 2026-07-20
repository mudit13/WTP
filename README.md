# WTP Topic 8 - AI Image Detection & Attribution

> New to the project? Start with **`docs/README.md`**, then `docs/IMPLEMENTATION_GUIDE.md`.

Unified repository for the project (github.com/mudit13/WTP). It contains **both** the
generation/inference pipeline that runs on the GPU container **and** the analysis/experiments
layer for the GOLD-review workstreams: diversified real class, scaling-vs-cropping
preprocessing study, DCT linear-SVM detector, fine-tuned attribution head, out-of-set
generalization, and robustness.

Code is authored locally and executed inside the GPU container. The repo root maps to the
container project root (`/pitsec_sose26_topic8`, i.e. it *is* `sharedDockerDir`); large data,
model weights, and venvs live alongside the code but are git-ignored (see `.gitignore`). See
`docs/SERVER_WORKFLOW.md` for the dev-here / run-on-server loop and `docs/ENVIRONMENTS.md`
for the venvs.

## Research questions

- RQ1 (Detection): How well does the log-DCT linear SVM separate real photographs from fake
  faces? The provided binary DE-FAKE model is the pretrained baseline.
- RQ2 (Attribution): Conditional on an image being detected as fake, how well can a fine-tuned
  DE-FAKE head distinguish eight generators: SD1.5 txt2img, SD1.5 img2img, FLUX.1-schnell,
  StyleGAN3-FFHQ, FaceApp, PGGAN-v1, PGGAN-v2, and StarGAN?
- Generalization: In eight LOGO folds, where does a closed-set head force an omitted generator,
  and can confidence/entropy support rejection? OpenForensics-fake is test-only.

## Pipeline overview

Generation + DE-FAKE inference live in `scripts/`:
`generate_sd15_txt2img.py`, `generate_sd15_img2img.py`, `generate_flux1_txt2img.py`,
`generate_stylegan3.py`,
`build_master_index.py`, `run_defake_batch.py` (`--dataset_filter dffd_` for the DFFD-only
subset), `merge_predictions.py`.
(The previously separate `update_master_index_dffd.py` is folded into the config-driven
`build_master_index.py`.) These produce:

- `dataset/master_metadata.csv` columns: `filename, full_path, label, generator, category,
  source_dataset, width, height`
- `dataset/defake_predictions*.csv` adds: `defake_predict` (0=real,1=fake), `prob_real`,
  `prob_fake`, `blip_caption`

The professor-facing system is a real cascade: DCT-SVM detection first, then the primary
eight-fake DE-FAKE attribution head. A secondary nine-way sensitivity model adds one merged
`real` class drawn evenly from London-DB, FFHQ, CelebA, and OpenForensics-real. Executed
experiments require an immutable `run_experiment.py --run_id ...` directory.

Every script reads/writes that exact schema via `scripts/lib/schema.py`. Path constants are
read from `configs/paths.env` (with the original absolute defaults), so no script hardcodes
paths that break when the repo moves.

## Layout

```
configs/            config.yaml + paths.example.env (copy to paths.env; never committed)
scripts/            All Python entry points (argparse CLIs, Python 3.9):
  generate_*.py       generation (per-generator venvs)
  build_master_index.py, merge_predictions.py
  run_defake_batch.py (--dataset_filter for subsets), score_defake_detection.py
  dct_*, finetune_*, eval_*, leave_one_generator_out, out_of_set_*, robustness_*
  prepare_variants, sample_dataset, make_split, make_datasheets, aggregate_results
scripts/lib/        Shared package: schema, metrics, io, image ops, clip/features, head
De-Fake-patched/    Vendored DE-FAKE code (blipmodels package + test.py/train.py)
docs/               README (document map), IMPLEMENTATION_GUIDE, PIPELINE (active runbook),
                    REVIEW_SAFEGUARDS, PROJECT_LOG, SERVER_WORKFLOW, ENVIRONMENTS, datasheets
report/             Report outline
results/  logs/      Generated outputs (git-ignored except .gitkeep)
--- not in git (live alongside on the server, git-ignored) ---
dataset/  models/  venv_sd15/  venv_flux1/  venv_stylegan3/  stylegan3/
```

## Interpreter rule (CRITICAL)

There are three generation venvs (one per generator). DE-FAKE inference AND the analysis
scripts in this repo run inside `venv_sd15` (it has clip + torch + the blipmodels package),
matching the team's working `run_defake_batch.py`.

| Task | Interpreter |
|------|-------------|
| SD1.5 generation | `venv_sd15/bin/python3` |
| FLUX.1 generation | `venv_flux1/bin/python3` |
| StyleGAN3 generation | `venv_stylegan3/bin/python3` |
| DE-FAKE inference + THIS repo's scripts | `venv_sd15/bin/python3` (= `$WTP_PY_DEFAKE`) |

Never use bare `python`. See `docs/PIPELINE.md` for the full run order and
`report/REPORT_OUTLINE.md` for the report structure.

## Security note (model/feature loading)

A few load paths deserialize Python objects and will execute arbitrary code if the file is
malicious. They are safe **only because we load our own trusted artifacts**:

- `run_defake_batch.py` uses `torch.load(..., weights_only=False)` on the supervisor-provided
  `clip_linear.pt` / `finetune_clip.pt` in `$WTP_ROOT/models`.
- `generate_stylegan3.py` uses `pickle.load` on the official StyleGAN3 `.pkl`.
- `features_cache.py` / `dct_svm.py` use `np.load(allow_pickle=True)` on caches this pipeline
  itself wrote.

Rule: never point these at a downloaded/untrusted checkpoint or feature file. If a checkpoint
is a plain `state_dict`, prefer `weights_only=True`.
