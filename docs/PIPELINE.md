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
- **Variants:** every experiment is run on a preprocessing variant index. `prepare_variants.py`
  writes three: `index_scaled.csv` (squash - DISTORTS non-square images; the uncontrolled
  reference), `index_cropped.csv` (center crop), and `index_aspect.csv` (aspect-PRESERVING
  resize+crop; no stretch). Use **`index_aspect.csv` for the confound-controlled runs** and
  compare against `index_scaled.csv` to MEASURE the aspect/format confound (supervisor request).

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

# 2. (WS2) Preprocessing variants -> writes index_{scaled,cropped,aspect}.csv.
#     "aspect" = aspect-preserving (no stretch) -> use for the confound-controlled runs.
#     "scaled" (squash) = the uncontrolled reference for the confound comparison.
$PY scripts/prepare_variants.py --config $CFG --master $DS/master_metadata.csv \
    --out_root $DS/variants --index_dir results/

# 2b. (WS2) MEASURE the confound (answers the supervisor directly, not just asserts it).
#     Metadata-only real/fake separability (width/height/aspect/format, NO pixels):
#       RAW master -> expect HIGH balanced acc/AUROC (the format/resolution leak is real).
#       Normalized variant (256 PNG) -> expect ~0.5 (the pipeline removed the leak).
#     The gap between the two IS the measurement.
$PY scripts/metadata_confound_probe.py --config $CFG \
    --metadata $DS/master_metadata.csv --out_dir results/confound_probe_raw/
$PY scripts/metadata_confound_probe.py --config $CFG \
    --metadata results/index_aspect.csv --out_dir results/confound_probe_aspect/

# 3a. (WS3) DE-FAKE detection. Inference (binary real/fake) on the full index.
#     GEOMETRY NOTE: run_defake_batch.py faithfully mirrors DE-FAKE's test.py, which SQUASHES
#     every image to 224x224 before CLIP. So feeding it RAW originals distorts non-square reals
#     regardless of variant. To geometry-control DE-FAKE DETECTION, run it on the ASPECT variant
#     index (already square, aspect-preserved) via WTP_MASTER_CSV below - not the raw master.
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

# 4. (WS4) GAN Fingerprints attribution -- PARKED, not on main.
#    A team decision (PR #5) removed the GAN-fp code from main; per the supervisor, DE-FAKE
#    multi-class attribution (WS5) takes PRIORITY over GAN-Fingerprints. The full GAN-fp
#    implementation (feature + CNN paths, faithful Fridrich-Kodovsky SRM front-end, benchmark)
#    is preserved on branch `ganfp-integrated` (origin/ganfp-integrated). To resume it later:
#      git checkout ganfp-integrated
#    Do NOT run ganfp scripts from main -- they are intentionally absent here.

# 5. (WS5) DE-FAKE multi-class attribution: fine-tune head + LOGO. THIS IS THE PRIORITY.
#     Faithful 1024-dim image+text features via reused BLIP captions (defake_predictions_all.csv).
#     Runs on the aspect-preserving variant (confound-controlled geometry). Give raw vs
#     controlled runs DISTINCT caches + out_dirs.
#     CLASS SPACE: the head trains on reals + config attribution.in_set_generators UNION
#     finetune_new_classes. Any OTHER generator in the index is treated as genuinely UNSEEN:
#     not trained, only force-scored -> written to finetune_unseen_per_image.csv, and merged
#     (tagged in_set=False) into finetune_per_image.csv for the out-of-set analysis.
#   --- controlled (JPEG-aug ON) : the headline multi-class attribution result ---
$PY scripts/finetune_defake_head.py --config $CFG --index results/index_aspect.csv \
    --jpeg_aug on --out_dir results/finetune_aspect_jpegaug/ \
    --features_cache results/clip_feats_aspect_jpegaug.npz \
    --captions_csv $DS/defake_predictions_all.csv
#   --- raw baseline (JPEG-aug OFF; same geometry) for the confound comparison ---
$PY scripts/finetune_defake_head.py --config $CFG --index results/index_aspect.csv \
    --jpeg_aug off --out_dir results/finetune_aspect_raw/ \
    --features_cache results/clip_feats_aspect_raw.npz \
    --captions_csv $DS/defake_predictions_all.csv
#   --- evaluate attribution. in-set/out-of-set is taken from the finetune run's actual class
#       list (finetune_metrics.json is auto-detected next to --predictions), NOT a static list ---
$PY scripts/eval_defake_attribution.py --config $CFG \
    --predictions results/finetune_aspect_jpegaug/finetune_per_image.csv \
    --out_dir results/attr_eval_aspect/ --pred_col pred_generator
#   --- LOGO: the STRICT out-of-set test (retrains WITHOUT the target). Default targets are the
#       trained fake set; override with --targets to hold out specific generators ---
$PY scripts/leave_one_generator_out.py --config $CFG --index results/index_aspect.csv \
    --jpeg_aug on --out_dir results/logo_aspect_jpegaug/ \
    --features_cache results/clip_feats_aspect_jpegaug.npz \
    --captions_csv $DS/defake_predictions_all.csv \
    --targets "FLUX.1-schnell" "StyleGAN3-FFHQ"

# 6. (WS6) Out-of-set analysis (confidence/entropy on unseen generators). finetune_per_image.csv
#    already carries BOTH populations (in_set flag), so it alone gives the in-vs-out overlay;
#    attribution_per_image.csv (from eval) also carries in_set. (LOGO reports its own out-of-set
#    behavior in logo_summary.json.)
$PY scripts/out_of_set_analysis.py --config $CFG --out_dir results/oos_aspect/ \
    --inputs finetuned=results/finetune_aspect_jpegaug/finetune_per_image.csv \
             attr_eval=results/attr_eval_aspect/attribution_per_image.csv

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

Variant sweeps: the commands above use `index_aspect.csv` (confound-controlled). Repeat the
same steps with `index_scaled.csv` to MEASURE the aspect/format confound (scaled vs aspect
delta = how much the model leaned on distortion/format), and with `index_cropped.csv` for the
scaling-vs-cropping comparison.
