# End-to-end pipeline runbook

Run inside the container from the repo root. `$WTP_PY_DEFAKE` (= venv_sd15 python) for
everything here; generation uses the per-generator venvs and is already done.

## Read this first (so 5 people stay consistent)

- **Interpreter:** always use `$WTP_PY_DEFAKE` for these scripts. Never bare `python`. See
  `docs/ENVIRONMENTS.md`.
- **JPEG augmentation default:** `finetune_defake_head.py` and `leave_one_generator_out.py`
  default to `--jpeg_aug auto`, which follows `config.augmentation.jpeg_train` (currently
  `true`). So by default they produce the **confound-controlled** result. For the **raw
  baseline**, pass `--jpeg_aug off`. The DCT script (`dct_extract_features.py`) is the opposite:
  raw by default, add `--jpeg_aug` for the controlled run.
- **Separate caches/outputs for raw vs controlled.** Always give the raw and augmented runs
  DIFFERENT `--features_cache` (or `--out`) and `--out_dir` paths, or you will overwrite/reuse
  the wrong features. The feature cache also refuses to reuse a clean cache for an aug run, but
  keep the names distinct anyway.
- **Variants:** every experiment is run on a preprocessing variant index (`index_scaled.csv`
  and `index_cropped.csv`) to keep the scaling-vs-cropping comparison clean.

```bash
export $(grep -v '^#' configs/paths.env | xargs)
PY=$WTP_PY_DEFAKE
CFG=configs/config.yaml
DS=/pitsec_sose26_topic8/dataset

# 1. Build ground-truth index (real schema; superset of build_master_index +
#    update_master_index_dffd). Reconcile against merged predictions.
$PY scripts/build_master_index.py --config $CFG --out $DS/master_metadata.csv \
    --reconcile $DS/defake_predictions_all.csv

# 1b. (WS1) Diversify reals: DFFD FFHQ is already a config source. Add OpenForensics reals
#     (pending the supervisor), then rebuild. Generate datasheets:
$PY scripts/make_datasheets.py --metadata $DS/master_metadata.csv --out results/datasheets.md

# 2. (WS2) Preprocessing variants (scaled + cropped PNG)
$PY scripts/prepare_variants.py --config $CFG --master $DS/master_metadata.csv \
    --out_root $DS/variants --index_dir results/

# 3a. (WS3) DE-FAKE detection. Inference (binary real/fake) on the full index.
#     IMPORTANT: run_defake_batch.py processes EVERY row in master_metadata.csv (no filter).
#     If your master index already contains the DFFD rows (the current unified build does),
#     run run_defake_batch.py ALONE - it covers everything (generated fakes, London-DB, CelebA,
#     and DFFD). Then score $WTP_PRED_CSV directly. Do NOT also run dffd+merge or the DFFD rows
#     get scored twice.
$PY scripts/run_defake_batch.py            # writes $WTP_PRED_CSV (ALL rows)
#     LEGACY split (only if master was built WITHOUT DFFD, then DFFD added separately):
#$PY scripts/run_defake_dffd.py            # writes $WTP_PRED_DFFD_CSV (dffd_* rows only)
#$PY scripts/merge_predictions.py          # -> $WTP_PRED_ALL_CSV
#     To run on a preprocessing variant instead, point at the variant index that
#     prepare_variants.py actually writes (results/index_scaled.csv / results/index_cropped.csv):
#       WTP_MASTER_CSV=results/index_scaled.csv \
#       WTP_PRED_CSV=$DS/defake_predictions_scaled.csv $PY scripts/run_defake_batch.py
#     Then SCORE the predictions (unified build -> score $WTP_PRED_CSV directly;
#     legacy split -> score $WTP_PRED_ALL_CSV):
$PY scripts/score_defake_detection.py --predictions $DS/defake_predictions.csv \
    --out_dir results/defake_detection/

# 3b. (WS3) DCT linear-SVM (Frank2020) on each variant.
#     Run BOTH: raw (shows the format/resolution confound) and --jpeg_aug (confound removed).
$PY scripts/dct_extract_features.py --index results/index_scaled.csv \
    --out results/dct_features_scaled.npz
$PY scripts/dct_extract_features.py --index results/index_scaled.csv \
    --out results/dct_features_scaled_jpegaug.npz --jpeg_aug
$PY scripts/dct_svm.py --features results/dct_features_scaled.npz \
    --out_dir results/dct_svm_scaled/ --mode random
$PY scripts/dct_svm.py --features results/dct_features_scaled.npz \
    --out_dir results/dct_svm_oos/ --mode out_of_set \
    --holdout_generators "FLUX.1-schnell" "StyleGAN3-FFHQ"

# 4. (WS4) Attribution. NOTE: pretrained DE-FAKE is binary-only -> attribution comes from
#    our fine-tuned head (step 5). GAN Fingerprints:
$PY scripts/run_ganfp.py --mode verify     # then --mode note if no weights

# 5. (WS5) Fine-tune head (faithful 1024-dim image+text via reused BLIP captions) + LOGO.
#     Run BOTH raw and controlled, with DISTINCT caches + out_dirs:
#   --- controlled (JPEG-aug; default via config) ---
$PY scripts/finetune_defake_head.py --config $CFG --index results/index_scaled.csv \
    --jpeg_aug on --out_dir results/finetune_scaled_jpegaug/ \
    --features_cache results/clip_feats_scaled_jpegaug.npz \
    --captions_csv $DS/defake_predictions_all.csv
#   --- raw baseline (no augmentation) ---
$PY scripts/finetune_defake_head.py --config $CFG --index results/index_scaled.csv \
    --jpeg_aug off --out_dir results/finetune_scaled_raw/ \
    --features_cache results/clip_feats_scaled_raw.npz \
    --captions_csv $DS/defake_predictions_all.csv
#   --- evaluate the controlled run's attribution ---
$PY scripts/eval_defake_attribution.py --config $CFG \
    --predictions results/finetune_scaled_jpegaug/finetune_per_image.csv \
    --out_dir results/attr_eval/ --pred_col pred_generator
#   --- LOGO (controlled) ---
$PY scripts/leave_one_generator_out.py --config $CFG --index results/index_scaled.csv \
    --jpeg_aug on --out_dir results/logo_scaled_jpegaug/ \
    --features_cache results/clip_feats_scaled_jpegaug.npz \
    --captions_csv $DS/defake_predictions_all.csv \
    --targets "FLUX.1-schnell" "StyleGAN3-FFHQ"

# 6. (WS6) Out-of-set analysis
$PY scripts/out_of_set_analysis.py --config $CFG --out_dir results/oos/ \
    --inputs finetuned=results/finetune_scaled_jpegaug/finetune_per_image.csv \
             attr_eval=results/attr_eval/attribution_per_image.csv

# 7. (WS7) Robustness on held-out test only
$PY scripts/make_split.py --config $CFG --index results/index_scaled.csv \
    --train_out results/train_index.csv --test_out results/test_index.csv
$PY scripts/robustness_perturb.py --mode generate --config $CFG \
    --index results/test_index.csv --out_root $DS/robust --index_dir results/robust/
#   run DE-FAKE on each results/robust/index_*.csv (via run_defake_batch.py), then:
$PY scripts/robustness_perturb.py --mode score --clean results/clean_pred.csv \
    --perturbed results/jpeg30_pred.csv --out results/robust/jpeg30_drop.json

# 8. (WS8) Aggregate for the report
$PY scripts/aggregate_results.py --results_dir results/ --out results/REPORT_SUMMARY.md
```

Repeat steps 3-7 with `index_cropped.csv` to produce the scaling-vs-cropping comparison.
