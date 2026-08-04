# Experiment and evidence catalog

This document maps research questions to executable entry points and expected evidence. It deliberately does not contain a permanent authoritative run ID or copied metric values. The final source of truth is the completed release manifest under `releases/<release-id>/release.yaml` and the run directory it names.

## Main experiments

| Research question | Entrypoint or stage | Expected evidence | Reporting role |
|---|---|---|---|
| How does pretrained DE-FAKE perform on binary detection? | `run_defake_batch.py`, `score_defake_detection.py` | binary predictions, threshold selection, metric JSON | comparison baseline |
| Can DCT features separate configured real and generated sources? | `dct_extract_features.py`, `dct_svm.py` | feature metadata, predictions, metric JSON | primary in-domain detector |
| Does the DCT detector generalize to an unseen manipulation source? | `dct_svm.py --mode out_of_set` | OOS predictions and metrics | external generalization finding |
| Can in-set generated images be attributed among configured generators? | `finetune_defake_head.py --class_mode fake_only`, `eval_defake_attribution.py` | trained-head metadata, predictions, class metrics | primary attribution result |
| What happens when one generator is absent during training? | `leave_one_generator_out.py` | per-held-out forced-label and confidence evidence | generator-overlap analysis |
| What is complete detection-to-attribution performance? | `evaluate_cascade.py` | cascade predictions and metrics | end-to-end result |
| How does the attribution model treat an external unseen generator? | `out_of_set_analysis.py` | confidence, entropy, forced-label, rejection evidence | OOS analysis |
| How sensitive is attribution to the FFHQ real class or source population? | `compare_ffhq_ablation.py` | matched with/without-FFHQ comparison | confound diagnostic |

## Supporting experiments

| Question | Entrypoint | Evidence role |
|---|---|---|
| Does adding a merged `Real` class change behavior? | `finetune_defake_head.py --class_mode joint` | optional taxonomy sensitivity |
| Are BLIP captions necessary? | attribution head without captions | image-only ablation |
| How stable is the result across seeds? | `seed_sweep.py` | mean, standard deviation, and per-seed metrics |
| What is the uncertainty around headline metrics? | `bootstrap_metrics.py` | grouped or stratified confidence intervals |
| Are model differences significant on aligned samples? | `compare_models_significance.py` | paired comparison |
| How sensitive are models to controlled perturbations? | `robustness_perturb.py` or orchestrator robustness stage | perturbation-specific degradation |
| Can metadata alone predict labels? | `metadata_confound_probe.py` | confound evidence |
| Are split boundaries clean? | `audit_split_leakage.py`, `audit_openforensics_coupling.py` | release integrity gates |
| How do attribution model families compare? | `benchmark_attribution.py` | matched model benchmark |

## Evidence rules

A reportable result must identify:

1. run ID and release ID
2. Git commit and configuration hash
3. exact dataset/index population
4. preprocessing and augmentation condition
5. class taxonomy
6. artifact path
7. metric definition and uncertainty treatment
8. whether evaluation is closed-set, OOS, LOGO, robustness, or cascade

Do not use shell transcripts as the sole scientific record. Logs support diagnosis; generated JSON/CSV artifacts and manifests support claims.

## Results that must not be presented as ordinary performance

- top-1 accuracy for an unseen generator that is absent from the output space
- Cohen's kappa for that forced-label OOS or LOGO design
- metrics that mix in-set and OOS populations without separate reporting
- historical results produced before a split, leakage, taxonomy, or cache correction
- a source-specific ablation as proof of a causal preprocessing mechanism
- preliminary GAN-fingerprint prototype metrics as final evidence

## Historical work

Development history and superseded experiments remain available under:

- [`history/PROJECT_LOG.md`](history/PROJECT_LOG.md)
- [`history/GANFP_HISTORICAL.md`](history/GANFP_HISTORICAL.md)

They preserve reasoning and debugging context but do not override a final release manifest.

## Final release record

After the final server run, create:

```text
releases/<release-id>/release.yaml
```

That manifest should point to every required artifact and include the exact headline spot checks used by `scripts/verify_handover.py`. Report drafts should cite the release ID and artifact paths rather than treating this catalog as a metric store.
