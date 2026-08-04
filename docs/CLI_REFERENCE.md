# Command-line reference

This is a role-based index of active scripts. Use `python <script> --help` with the appropriate configured interpreter for the complete argument list.

## Normal workflow

| Script | Role |
|---|---|
| `scripts/build_master_index.py` | Build the canonical metadata index from configured datasets. |
| `scripts/run_experiment.py` | Plan and orchestrate an immutable experiment run. This is the main execution entry point. |
| `scripts/verify_handover.py` | Verify a completed release manifest and its run artifacts. |
| `scripts/capture_env_locks.sh` | Capture exact package snapshots from the three server environments. |
| `scripts/check_docs.py` | Validate links and required active documentation. |

## Data preparation

| Script | Role |
|---|---|
| `scripts/extract_openforensics.py` | Extract balanced OpenForensics crops and grouping metadata. |
| `scripts/make_img2img_group_map.py` | Link London source images to SD1.5 img2img derivatives. |
| `scripts/prepare_variants.py` | Build normalized preprocessing variants. |
| `scripts/make_split.py` | Create stable, group-aware train and test indices. |
| `scripts/create_data_manifest.py` | Hash data and checkpoint inputs used by a run. |
| `scripts/make_datasheets.py` | Generate dataset datasheet text from metadata and the template. |

## Synthetic generation

| Script | Interpreter | Role |
|---|---|---|
| `scripts/generate_sd15_txt2img.py` | `WTP_PY_SD15` | Generate SD1.5 text-to-image data. |
| `scripts/generate_sd15_img2img.py` | `WTP_PY_SD15` | Generate SD1.5 image-to-image data. |
| `scripts/generate_flux1_txt2img.py` | `WTP_PY_FLUX1` | Generate FLUX.1-schnell data. |
| `scripts/generate_stylegan3.py` | `WTP_PY_STYLEGAN3` | Generate StyleGAN3 data through the external checkout. |

## Binary detection

| Script | Role |
|---|---|
| `scripts/dct_extract_features.py` | Extract DCT log-magnitude features. |
| `scripts/dct_svm.py` | Train and evaluate the linear DCT-SVM detector. |
| `scripts/run_defake_batch.py` | Run batch pretrained DE-FAKE binary inference. |
| `scripts/score_defake_detection.py` | Score binary DE-FAKE predictions and operating thresholds. |
| `scripts/compare_models_significance.py` | Compare aligned detector predictions statistically. |

## Attribution

| Script | Role |
|---|---|
| `scripts/finetune_defake_head.py` | Train the multi-class attribution head on frozen features. |
| `scripts/predict_defake_head.py` | Run a trained attribution head on a dataset. |
| `scripts/eval_defake_attribution.py` | Evaluate closed-set attribution. |
| `scripts/evaluate_cascade.py` | Evaluate complete detector-to-attribution performance. |
| `scripts/leave_one_generator_out.py` | Run held-out-generator training and evaluation. |
| `scripts/out_of_set_analysis.py` | Analyze unseen-source confidence, entropy, and rejection. |
| `scripts/compare_ffhq_ablation.py` | Compare matched attribution conditions with and without FFHQ. |
| `scripts/benchmark_attribution.py` | Compare attribution model families on a common split. |

## Rigor and robustness

| Script | Role |
|---|---|
| `scripts/audit_split_leakage.py` | Audit group, exact-duplicate, and perceptual-duplicate leakage. |
| `scripts/audit_openforensics_coupling.py` | Audit related OpenForensics real/fake samples across splits. |
| `scripts/bootstrap_metrics.py` | Compute confidence intervals. |
| `scripts/seed_sweep.py` | Quantify split and initialization sensitivity. |
| `scripts/metadata_confound_probe.py` | Test label predictability from metadata alone. |
| `scripts/robustness_perturb.py` | Generate and score named perturbations. |
| `scripts/aggregate_results.py` | Aggregate immutable run artifacts for reporting. |

## GAN-fingerprint research paths

| Script | Role |
|---|---|
| `scripts/train_ganfp.py` | Train the handcrafted residual and frequency-feature model. |
| `scripts/train_ganfp_cnn.py` | Train the SRM-front-end CNN model. |

These are supporting or historical comparison paths unless a release manifest explicitly includes them.

## Shared implementation

Reusable code is under `scripts/lib/` and is not normally invoked directly:

- taxonomy and schema
- CLIP and BLIP feature extraction
- attribution head and feature-cache logic
- GAN-fingerprint features and network
- image preprocessing and perturbations
- I/O, configuration, and logging
- metrics

## Archived scripts

Superseded one-off utilities are under `scripts/legacy/`. They are retained for history only and must not be added to an active runbook without review.
