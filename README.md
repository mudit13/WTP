# WTP Topic 8 - AI Image Detection & Attribution

> New to the project? Start with **`docs/IMPLEMENTATION_GUIDE.md`**, then `docs/PIPELINE.md`.

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

- RQ1 (Detection): How well do CLIP-based (DE-FAKE) and frequency-based (DCT-SVM) detectors
  separate real photographs from AI-generated faces, including out-of-set generators
  (FLUX.1, StyleGAN3, DFFD GANs)?
- RQ2 (Attribution): How well can a generator be attributed from a fake image, in-set vs
  out-of-set? (Note: the provided DE-FAKE head is binary-only; attribution comes from our
  fine-tuned head.)

## Pipeline overview

Generation + DE-FAKE inference (already run on the server) now live in `scripts/`:
`generate_sd15_txt2img.py`, `generate_flux1_txt2img.py`, `generate_stylegan3.py`,
`build_master_index.py`, `run_defake_batch.py` / `run_defake_dffd.py`, `merge_predictions.py`.
(The previously separate `update_master_index_dffd.py` is folded into the config-driven
`build_master_index.py`.) These produce:

- `dataset/master_metadata.csv` columns: `filename, full_path, label, generator, category,
  source_dataset, width, height`
- `dataset/defake_predictions*.csv` adds: `defake_predict` (0=real,1=fake), `prob_real`,
  `prob_fake`, `blip_caption`

Every script reads/writes that exact schema via `scripts/lib/schema.py`. Path constants are
read from `configs/paths.env` (with the original absolute defaults), so no script hardcodes
paths that break when the repo moves.

## Layout

```
configs/            config.yaml + paths.example.env (copy to paths.env; never committed)
scripts/            All Python entry points (argparse CLIs, Python 3.9):
  generate_*.py       generation (per-generator venvs)
  build_master_index.py, merge_predictions.py
  run_defake_batch.py, run_defake_dffd.py, score_defake_detection.py
  dct_*, finetune_*, eval_*, leave_one_generator_out, out_of_set_*, robustness_*, run_ganfp
  prepare_variants, sample_dataset, make_split, make_datasheets, aggregate_results
scripts/lib/        Shared package: schema, metrics, io, image ops, clip/features, head
De-Fake-patched/    Vendored DE-FAKE code (blipmodels package + test.py/train.py)
docs/               IMPLEMENTATION_GUIDE (start here), PIPELINE runbook, GOLD_ALIGNMENT,
                    PROJECT_LOG (changes+reasons), SERVER_WORKFLOW, ENVIRONMENTS, DATASHEET_TEMPLATE
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
`docs/REPORT_OUTLINE.md` for the report structure.
