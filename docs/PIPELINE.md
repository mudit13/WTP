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

# 1b. (WS1) Diversify reals. OpenForensics reals are a TRAINED real class (Dennis's #1 steer);
#     OpenForensics-fake is out-of-set. Both are capped at sample_size 300 in config.yaml.
#
#   PRIMARY (self-contained): crop faces straight from the raw OpenForensics JSONs into
#   real/ + fake/. RUN ON THE HOST (the /vol1 source is not mounted in the container) with the
#   host python; --out_dir must be the host path the container sees as $DS/openforensics. On this
#   server that mapping is host /vol2/pitsec_sose26_topic8/sharedDockerDir/dataset == container
#   /pitsec_sose26_topic8/dataset, so:
#     # (optional) clear any prior crops in the shared dir first:
#     rm -rf /vol2/pitsec_sose26_topic8/sharedDockerDir/dataset/openforensics/*
#     python3 scripts/extract_openforensics.py \
#         --root /vol1/share/DeepFake/OpenForensics \
#         --out_dir /vol2/pitsec_sose26_topic8/sharedDockerDir/dataset/openforensics \
#         --splits Val --per_class_limit 300
#   ^ RE-EXTRACT even if you already have OF crops from before: this version additionally
#   records each crop's source photo id (source_image_id) in openforensics_metadata.csv AND a
#   dedicated openforensics_groups.csv sidecar (full_path,source_image_id) at --out_dir. That
#   sidecar is what makes every downstream split GROUP-AWARE (see step 5+ below): a real crop
#   and a fake crop cropped from the SAME source photo are now kept on the SAME side of
#   train/val/test instead of splitting purely on the crop's own full_path. This closes the
#   same-source-photo coupling leak (see docs/PROJECT_LOG.md section 14/Open items).
#
#   ALTERNATIVE (if you already have FLAT crops + a label CSV from a prior extraction): sort them
#   into real/ + fake/ instead of re-cropping. NOTE: ingest_openforensics.py does NOT produce
#   the openforensics_groups.csv sidecar (it has no access to the original annotation JSON), so
#   group-aware splitting will be a no-op (falls back to singleton groups, i.e. today's
#   pre-fix behavior) unless you also run scripts/audit_openforensics_coupling.py's ann_id/
#   image_id parsing logic to backfill one - prefer the PRIMARY path above.
#     $PY scripts/ingest_openforensics.py --crops_csv <of_metadata.csv> \
#         --crops_dir <flat crops dir> --out_root $DS/openforensics --mode symlink
#
#   Then (inside the container) rebuild the index + datasheets:
$PY scripts/build_master_index.py --config $CFG --out $DS/master_metadata.csv \
    --reconcile $DS/defake_predictions_all.csv
$PY scripts/make_datasheets.py --metadata $DS/master_metadata.csv --out results/datasheets.md

# 1c. (WS1) CONFOUND GATE for OpenForensics (colleague's crop-size concern): probe real/fake
#     separability from crop SIZE alone, restricted to OF rows. ~0.5 = OF is clean; if HIGH, the
#     bounding-box sizes leak the label -> use the aspect variant for OF before trusting results.
$PY scripts/metadata_confound_probe.py --config $CFG --metadata $DS/master_metadata.csv \
    --source_filter openforensics --out_dir results/confound_probe_of/
# After this gate passes, RE-RUN detection (step 3a) and the multi-class attribution fine-tune
# (step 5, now 7-class: +OpenForensics real) so the OF real class is reflected in the headline
# numbers, then the rigor add-ons in step 7b (which now also includes the SEPARATE same-source-
# photo coupling audit, audit_openforensics_coupling.py -- the crop-SIZE confound gate above and
# the same-source-PHOTO coupling audit below are two different risks; both must be checked).

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
#     LEGACY split (only if master was built WITHOUT DFFD, then DFFD added separately). The old
#     run_defake_dffd.py is now just run_defake_batch.py with a --dataset_filter:
#$PY scripts/run_defake_batch.py --dataset_filter dffd_ --out $DS/defake_predictions_dffd.csv
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
#     IMPORTANT (leakage fix): --mode random used to draw its OWN internal split, stratified on
#     the BINARY label; the robustness test set (make_split.py) is stratified on the 12-class
#     generator column. Same seed/test_size does NOT guarantee the same partition across two
#     different stratification columns, so without --test_index a fraction of test_index.csv's
#     rows can leak into this SVM's training set (this is exactly what inflated the old
#     robustness-clean baseline vs the from-scratch random-split number). ALWAYS pass
#     --test_index for any DCT-SVM run whose test rows will later be perturbed/scored by
#     robustness_perturb.py (i.e. any run feeding step 7b's DCT robustness table).
$PY scripts/dct_extract_features.py --index results/index_scaled.csv \
    --out results/dct_features_scaled.npz
$PY scripts/dct_extract_features.py --index results/index_scaled.csv \
    --out results/dct_features_scaled_jpegaug.npz --jpeg_aug
$PY scripts/dct_svm.py --features results/dct_features_scaled.npz \
    --out_dir results/dct_svm_scaled/ --mode random
$PY scripts/dct_svm.py --features results/dct_features_scaled.npz \
    --out_dir results/dct_svm_oos/ --mode out_of_set \
    --holdout_generators "FLUX.1-schnell" "StyleGAN3-FFHQ"
#     ^ this is a quick manual sanity check on the SCALED variant, kept without --test_index on
#     purpose (no robustness pipeline consumes it). run_experiment.py's `dct` stage ALWAYS adds
#     --test_index (it runs make_split.py first, then trains with --test_index unconditionally)
#     because its dct_svm.joblib is the SAME one the `robustness` stage reuses later - do not
#     "simplify" run_experiment.py to match this manual snippet; they intentionally differ.

# 4. (WS4) GAN Fingerprints (Yu2019-inspired) attribution -- on main.
#    Second attribution method beside DE-FAKE, targeting the GAN-specific traces CLIP misses.
#    Two paths share ONE seeded stratified split (scripts/benchmark_attribution.py):
#      Path A = residual/spectrum fingerprint features + MLP head (train_ganfp.py);
#      Path B = end-to-end CNN with a fixed Fridrich-Kodovsky SRM high-pass front-end
#               (train_ganfp_cnn.py). Yu2019-INSPIRED, not a byte-faithful port.
#    Run on the aspect variant for the confound-controlled result:
$PY scripts/train_ganfp.py --config $CFG --index results/index_aspect.csv \
    --jpeg_aug on --features_cache results/ganfp_feats_aspect.npz \
    --out_dir results/ganfp_feature_aspect/
$PY scripts/train_ganfp_cnn.py --config $CFG --index results/index_aspect.csv \
    --jpeg_aug on --device cuda --out_dir results/ganfp_cnn_aspect/
$PY scripts/benchmark_attribution.py --config $CFG --index results/index_aspect.csv \
    --out_dir results/ganfp_benchmark_aspect/ \
    --defake_csv results/finetune_aspect_jpegaug/finetune_per_image.csv
#    Full-12-class run above = each method's best-case under ITS OWN training regime (now
#    labeled per-row via classes_trained_on/n_classes_trained_on in benchmark_metrics.json).
#    For the FAIR head-to-head (same 7 classes DE-FAKE was trained on), also run:
$PY scripts/benchmark_attribution.py --config $CFG --index results/index_aspect.csv \
    --out_dir results/ganfp_benchmark_aspect_7class/ \
    --classes "London-DB" "FFHQ" "CelebA" "OpenForensics" "SD1.5" "FLUX.1-schnell" \
        "StyleGAN3-FFHQ" \
    --defake_csv results/finetune_aspect_jpegaug/finetune_per_image.csv

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
#   --- LOGO: the STRICT out-of-set test (retrains WITHOUT the target). DEFAULT targets are
#       ONLY finetune_new_classes (FLUX.1-schnell, StyleGAN3-FFHQ) -- this is "Leave-New-Class-
#       Out", not a full leave-one-generator-out sweep (see report/REPORT_OUTLINE.md section 8).
$PY scripts/leave_one_generator_out.py --config $CFG --index results/index_aspect.csv \
    --jpeg_aug on --out_dir results/logo_aspect_jpegaug/ \
    --features_cache results/clip_feats_aspect_jpegaug.npz \
    --captions_csv $DS/defake_predictions_all.csv \
    --targets "FLUX.1-schnell" "StyleGAN3-FFHQ"
#   --- Proper LOGO: hold out EVERY trained class in turn (reals included), one flag ---
$PY scripts/leave_one_generator_out.py --config $CFG --index results/index_aspect.csv \
    --jpeg_aug on --out_dir results/logo_full_aspect_jpegaug/ \
    --features_cache results/clip_feats_aspect_jpegaug.npz \
    --captions_csv $DS/defake_predictions_all.csv --all_trained_classes

# 6. (WS6) Out-of-set analysis (confidence/entropy on unseen generators). finetune_per_image.csv
#    already carries BOTH populations (in_set flag), so it alone gives the in-vs-out overlay;
#    attribution_per_image.csv (from eval) also carries in_set. (LOGO reports its own out-of-set
#    behavior in logo_summary.json.)
$PY scripts/out_of_set_analysis.py --config $CFG --out_dir results/oos_aspect/ \
    --inputs finetuned=results/finetune_aspect_jpegaug/finetune_per_image.csv \
             attr_eval=results/attr_eval_aspect/attribution_per_image.csv

# 7. (WS7) Robustness on held-out test only. Tested for ALL THREE methods (DE-FAKE detection,
#     DCT-SVM detection, DE-FAKE attribution) on the SAME perturbed set so the method comparison
#     is apples-to-apples. NOTE: adding OpenForensics changed the test split, so REGENERATE the
#     perturbations (delete results/robust + $DS/robust first) rather than reusing an old run.
#     generate writes ONLY the 8 perturbation indices (index_jpeg30.csv ... index_sharpen1.csv);
#     there is NO index_clean.csv - the CLEAN baseline is results/test_index.csv itself.
#     make_split.py NOW uses the SAME content-stable, group-aware split as every other stage
#     (defake_head.stratified_split) instead of its own ad-hoc sklearn split, so test_index.csv
#     changes composition vs. any pre-fix run - regenerate it (and everything derived from it)
#     rather than reusing an old copy.
$PY scripts/make_split.py --config $CFG --index results/index_aspect.csv \
    --train_out results/train_index.csv --test_out results/test_index.csv
$PY scripts/robustness_perturb.py --mode generate --config $CFG \
    --index results/test_index.csv --out_root $DS/robust --index_dir results/robust/

#   --- (a) DE-FAKE detection robustness ---
#   CLEAN baseline = DE-FAKE on the unperturbed test set:
WTP_MASTER_CSV=results/test_index.csv \
WTP_PRED_CSV=$DS/robust_clean_pred.csv $PY scripts/run_defake_batch.py
#   each perturbation (perturbed pred carries source_path -> the scorer joins on it):
for name in jpeg30 jpeg50 jpeg70 blur1 blur2 resize0.5 resize0.75 sharpen1; do
  WTP_MASTER_CSV=results/robust/index_${name}.csv \
  WTP_PRED_CSV=$DS/robust_${name}_pred.csv $PY scripts/run_defake_batch.py
  $PY scripts/robustness_perturb.py --mode score --clean $DS/robust_clean_pred.csv \
      --perturbed $DS/robust_${name}_pred.csv --out results/robust/${name}_drop.json
done

#   --- (b) DCT-SVM detection robustness (reuse the trained SVM from step 3b/7b) ---
#   Predict the SAME perturbed images with the fitted model (dct_svm.py --mode predict), then
#   score the flip/accuracy drop against the clean DCT predictions.
#   PREREQUISITE (leakage fix): the SVM used here (results/dct_svm_aspect/dct_svm.joblib) MUST
#   have been trained with --test_index results/test_index.csv (step 3b), otherwise this
#   "clean" baseline is partly the SVM scoring its own training rows -- see the note in step 3b.
$PY scripts/dct_extract_features.py --index results/index_aspect.csv \
    --out results/dct_features_aspect.npz --jpeg_aug
$PY scripts/dct_svm.py --features results/dct_features_aspect.npz \
    --out_dir results/dct_svm_aspect/ --mode random --test_index results/test_index.csv
$PY scripts/dct_extract_features.py --index results/test_index.csv \
    --out results/robust/dct_clean.npz --jpeg_aug
$PY scripts/dct_svm.py --mode predict --model results/dct_svm_aspect/dct_svm.joblib \
    --features results/robust/dct_clean.npz --out_dir results/robust/dct_clean/
for name in jpeg30 jpeg50 jpeg70 blur1 blur2 resize0.5 resize0.75 sharpen1; do
  $PY scripts/dct_extract_features.py --index results/robust/index_${name}.csv \
      --out results/robust/dct_${name}.npz --jpeg_aug
  $PY scripts/dct_svm.py --mode predict --model results/dct_svm_aspect/dct_svm.joblib \
      --features results/robust/dct_${name}.npz --out_dir results/robust/dct_${name}/
  $PY scripts/robustness_perturb.py --mode score \
      --clean results/robust/dct_clean/dct_per_image.csv \
      --perturbed results/robust/dct_${name}/dct_per_image.csv \
      --pred_col pred --conf_col score --out results/robust/dct_${name}_drop.json
done

#   --- (c) DE-FAKE ATTRIBUTION robustness (does perturbation change WHICH generator?) ---
#   Predict with the fine-tuned head (predict_defake_head.py), score generator-label flips.
$PY scripts/predict_defake_head.py --config $CFG \
    --head results/finetune_aspect_jpegaug/defake_head.pt --index results/test_index.csv \
    --captions_csv $DS/defake_predictions_all.csv --out results/robust/attr_clean.csv
for name in jpeg30 jpeg50 jpeg70 blur1 blur2 resize0.5 resize0.75 sharpen1; do
  $PY scripts/predict_defake_head.py --config $CFG \
      --head results/finetune_aspect_jpegaug/defake_head.pt \
      --index results/robust/index_${name}.csv \
      --captions_csv $DS/defake_predictions_all.csv --out results/robust/attr_${name}.csv
  $PY scripts/robustness_perturb.py --mode score --clean results/robust/attr_clean.csv \
      --perturbed results/robust/attr_${name}.csv --pred_col pred_generator \
      --conf_col confidence --out results/robust/attr_${name}_drop.json
done

#   (Or run all of stage 7 in one go: $PY scripts/run_experiment.py --stages robustness)

# 7b. (RIGOR) Uncertainty, threshold hygiene, and leakage audit. These are add-ons; they do
#      not change any run above, they quantify its reliability. All numpy/PIL/sklearn-only.
#   --- 95% bootstrap CIs for the headline detection + attribution numbers (per-class recall too):
$PY scripts/bootstrap_metrics.py \
    --predictions $DS/defake_predictions_aspect.csv \
    --out results/ci/defake_detection_aspect_ci.json
$PY scripts/bootstrap_metrics.py --subset in_set \
    --predictions results/attr_eval_aspect/attribution_per_image.csv \
    --out results/ci/attr_eval_aspect_ci.json
#   --- seed sweep: re-split + re-train ONLY the head over K seeds on the cached features
#       (fast, no CLIP recompute) -> mean/std/CI of in-set balanced accuracy + per-class recall:
$PY scripts/seed_sweep.py --config $CFG --index results/index_aspect.csv \
    --features_cache results/clip_feats_aspect_jpegaug.npz \
    --captions_csv $DS/defake_predictions_all.csv --jpeg_aug on \
    --n_seeds 10 --out results/ci/seed_sweep_aspect.json
#   --- paired DE-FAKE vs DCT significance (McNemar + paired AUROC bootstrap). Needs DCT
#       per-image output; this is the SAME dct_svm.py run already done in step 3b/7a with
#       --test_index (do not refit without it, or dct_per_image.csv reintroduces the leak):
$PY scripts/compare_models_significance.py \
    --defake $DS/defake_predictions_aspect.csv \
    --dct results/dct_svm_aspect/dct_per_image.csv \
    --out results/ci/defake_vs_dct_aspect.json
#   --- split-leakage audit: exact (SHA-256) + near-duplicate (dHash) checks across the
#       train/val/test partition, plus per-generator balance counts (DIAGNOSTIC). This ALSO
#       reports group_straddle (0 expected): every finetune/train_ganfp/benchmark/LOGO/
#       make_split run above already auto-loads openforensics_groups.csv and keeps each source
#       photo's crops on one split side (scripts/lib/defake_head.py stratified_split groups=);
#       group_straddle > 0 would mean that fix is not actually taking effect for this run
#       (e.g. group_map path mismatch) - investigate immediately if so:
$PY scripts/audit_split_leakage.py --config $CFG --index results/index_aspect.csv \
    --out results/leakage_audit.json
#   --- OpenForensics same-source-photo coupling, independent cross-check (the dHash audit above
#       cannot catch this coupling at all - see report/REPORT_OUTLINE.md Limitations). Re-parses
#       the ORIGINAL Val_poly.json directly (host path under /vol1/share/DeepFake/OpenForensics,
#       or wherever it was copied) rather than relying on the openforensics_groups.csv sidecar,
#       so it also validates the sidecar itself. n_real_fake_pairs_straddling_splits should now
#       be 0 (group-aware splitting is active); if it is NOT 0, the group-aware fix has a bug -
#       do not report OpenForensics numbers until it is 0:
$PY scripts/audit_openforensics_coupling.py \
    --polygon_json /vol1/share/DeepFake/OpenForensics/Val_poly.json \
    --config $CFG --index results/index_aspect.csv \
    --out results/of_coupling_audit.json
#   --- metadata-confound variant sweep (completeness): every normalized variant should be ~0.5.
for idx in scaled cropped; do
  $PY scripts/metadata_confound_probe.py --config $CFG \
      --metadata results/index_${idx}.csv --out_dir results/confound_probe_${idx}/
done
$PY scripts/metadata_confound_probe.py --config $CFG \
    --metadata results/robust/index_jpeg30.csv --out_dir results/confound_probe_jpeg30/

# 8. (WS8) Aggregate for the report
$PY scripts/aggregate_results.py --results_dir results/ --out results/REPORT_SUMMARY.md

# 9. (APPENDIX, optional) Raw-GEOMETRY out-of-set / LOGO baseline.
#    The headline out-of-set/LOGO runs above are confound-controlled (aspect geometry + JPEG-aug).
#    We already have a JPEG-aug-OFF baseline (the GAN-collapse reproduces there). This isolates the
#    remaining GEOMETRY axis for out-of-set by re-running on the SCALED (squash) index with
#    JPEG-aug OFF. ZERO new code - just different --index/--jpeg_aug + distinct caches/out_dirs.
#    Expectation: the qualitative unseen-GAN -> real collapse persists (geometry is not the driver).
$PY scripts/finetune_defake_head.py --config $CFG --index results/index_scaled.csv \
    --jpeg_aug off --out_dir results/finetune_scaled_raw/ \
    --features_cache results/clip_feats_scaled_raw.npz \
    --captions_csv $DS/defake_predictions_all.csv
$PY scripts/leave_one_generator_out.py --config $CFG --index results/index_scaled.csv \
    --jpeg_aug off --out_dir results/logo_scaled_raw/ \
    --features_cache results/clip_feats_scaled_raw.npz \
    --captions_csv $DS/defake_predictions_all.csv \
    --targets "FLUX.1-schnell" "StyleGAN3-FFHQ"
$PY scripts/out_of_set_analysis.py --config $CFG --out_dir results/oos_scaled_raw/ \
    --inputs finetuned=results/finetune_scaled_raw/finetune_per_image.csv
#    (Shortcut: $PY scripts/run_experiment.py --variant scaled --jpeg_aug off \
#        --stages attribution,oos)
```

Variant sweeps: the commands above use `index_aspect.csv` (confound-controlled). Repeat the
same steps with `index_scaled.csv` to MEASURE the aspect/format confound (scaled vs aspect
delta = how much the model leaned on distortion/format), and with `index_cropped.csv` for the
scaling-vs-cropping comparison.

## One-command runner

`scripts/run_experiment.py` wraps the stages above with consistent variant/jpeg-aug/path naming
(it just shells out to the same scripts, so behaviour matches this runbook). Preview first:

```bash
$PY scripts/run_experiment.py --dry_run                 # headline aspect + jpeg-aug plan
$PY scripts/run_experiment.py                           # run the headline path
$PY scripts/run_experiment.py --stages ganfp,robustness # add the heavy stages
$PY scripts/run_experiment.py --variant scaled --jpeg_aug off --stages attribution,oos  # raw baseline
```

Stages: `index, variants, confound, detect, dct, attribution, oos, ganfp, robustness, aggregate`.
Default = the confound-controlled headline (everything except `ganfp`/`robustness`).

## Cleanup / regenerate after a dataset change (e.g. adding OpenForensics)

Adding or recapping a dataset changes the master index, so all DERIVED artifacts must be
regenerated. The CLIP feature cache is safe on its own (its signature hashes the index content,
so a stale cache is never silently reused - `scripts/lib/features_cache.py`), but delete the
derived files anyway to reclaim disk and avoid confusion. KEEP raw datasets and model checkpoints.

```bash
# derived indices, feature caches, predictions, and result dirs (regenerated by a re-run):
rm -f  results/index_*.csv results/clip_feats_*.npz results/ganfp_feats_*.npz \
       results/dct_features_*.npz $DS/defake_predictions_*.csv
rm -rf results/finetune_* results/logo_* results/attr_eval_* results/oos_* \
       results/dct_svm_* results/ganfp_* results/defake_detection_* results/ci
# robustness perturbations are keyed to the OLD test split -> always regenerate:
rm -rf results/robust $DS/robust $DS/robust_*_pred.csv
# then rebuild from step 1 (or: $PY scripts/run_experiment.py).
```
