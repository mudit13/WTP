# Experiment catalog

This catalog maps every reportable experiment to committed code and immutable evidence. Shell
transcripts are operational history, not the scientific record.

## Authoritative run

- Run ID: `2026-08-01_eightway_v1`
- Core model commit: `141da128aaee97607b9b05f39b49149046afc1d0`
- Index: 2,154 images (1,132 fake; 1,022 real)
- Fake attribution classes: 8
- External OOS benchmark: 300 OpenForensics-fake crops
- Headline geometry: aspect-preserving 256-pixel variant
- Training control: JPEG q30–100 on training features only
- Evidence root: `results/2026-08-01_eightway_v1/`

## Main-report experiments

| Question | Entrypoint | Primary evidence | Report role |
|---|---|---|---|
| Pretrained binary baseline | `run_defake_batch.py`, `score_defake_detection.py` | `defake_detection_aspect/` | Detection baseline |
| In-domain frequency detector | `dct_extract_features.py`, `dct_svm.py` | `dct_svm_aspect/` | Primary detection |
| Unseen manipulation detection | `dct_svm.py --mode out_of_set` | `dct_svm_aspect_oos/` | Negative generalization finding |
| Eight-way fake attribution | `finetune_defake_head.py --class_mode fake_only` | `finetune_8way_aspect_jpegaug/`, `attr_eval_8way_aspect/` | Primary attribution |
| Leave-one-generator-out | `leave_one_generator_out.py` | `logo_8way_aspect_jpegaug/` | Generator-overlap analysis |
| DCT→attribution cascade | `evaluate_cascade.py` | `cascade/` | End-to-end result |
| True external OOS attribution | `out_of_set_analysis.py` | `oos_aspect/` | Forced-label/rejection analysis |
| FFHQ sensitivity | `compare_ffhq_ablation.py` | `ffhq_ablation/` | Professor-requested diagnostic |

## Supporting experiments

| Question | Entrypoint | Evidence | Reporting |
|---|---|---|---|
| Optional merged Real class | `finetune_defake_head.py --class_mode joint` | `finetune_9way_aspect_jpegaug/` | Short sensitivity subsection |
| BLIP-caption dependence | Same head without `--captions_csv` | `finetune_8way_image_only_aspect_jpegaug/`, `ci_attr_8way_image_only.json`, `seed_sweep_8way_image_only.json` | Short ablation paragraph |
| Perturbation robustness | `run_experiment.py --stages robustness` | `robust/` | Supporting subsection / appendix |
| Confidence intervals | `bootstrap_metrics.py` | `ci_*.json` | Alongside headline metrics |
| Split/init sensitivity | `seed_sweep.py` | `seed_sweep_8way.json` | Alongside fixed-seed result |
| Paired detection comparison | `compare_models_significance.py` | `defake_vs_dct_significance.json` | Qualifies DCT-vs-DE-FAKE claim |
| Leakage audit | `audit_split_leakage.py` | `leakage_audit_8way*.json` | Integrity gate / appendix |

## Historical or non-headline work

- `docs/GANFP_HISTORICAL.md`: method-development history only; no old metric is final evidence.
- Preliminary 7-class attribution: project evolution only, not a final result.
- Img2img strength/CFG/prompt pilots: generation-method provenance, not model experiments.
- Auxiliary nine-way OOS top-1: not meaningful because OpenForensics-fake is absent from the
  output space.

## Results that must not be presented as performance

- OOS top-1 accuracy or Cohen’s kappa when the true class is absent
- `all_fakes` metrics mixing known classes with OpenForensics-fake
- Superseded DCT robustness numbers produced before the shared-test-boundary fix
- The false 102-group-straddle warning from the pre-fix audit
- Historical GAN-fp 10-class prototype metrics

## Interpretation boundaries

- DCT has significantly better balanced accuracy at the evaluated operating point; its AUROC
  advantage over pretrained DE-FAKE is not significant.
- Eight-way attribution is source classification under this dataset design, not proof of
  content-independent generator fingerprints.
- London-only sources and the fixed studio prompt make SD1.5-img2img visually narrow.
- Image-only CLIP performance shows BLIP captions are not necessary, but CLIP image embeddings
  remain semantic.
- DCT OOS failure and blur/downscale sensitivity are negative findings, not failed experiments.
- FFHQ removal demonstrates sensitivity to the FFHQ class/population, not a causal preprocessing
  mechanism.

## Reproducing the sequence

For new runs, use the default `run_experiment.py` stages, which now include mandatory rigor before
aggregation. For the completed authoritative run, the core, rigor, robustness, and image-only
logs preserve timestamps and exact command lines. `run_manifest.json` preserves the original
commit and configuration hash.
