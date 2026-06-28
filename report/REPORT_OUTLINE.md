# Scientific report outline (Topic 8)

Each section lists the experiment/script that produces its evidence so writing is a matter
of pulling in the generated CSVs/figures. Keep the GOLD framing throughout: the goal is a
scientifically reliable evaluation, not maximal accuracy.

## 1. Introduction & Problem Statement
- AI image detection (real vs fake) and attribution (which generator).
- RQ1 (detection generalization) and RQ2 (attribution, in-set vs out-of-set).
- State the approved deviations from the interim plan (see section 9).

## 2. Related Work
- Sha2023 (DE-FAKE, CLIP semantic detection + attribution) - primary method.
- Frank2020 (DCT frequency artifacts) - secondary detector (DCT linear-SVM).
- Yu2019 (GAN fingerprints, residual attribution).
- Dang2020 (DFFD dataset, attention-CNN face manipulation).

## 3. Datasets
- Real: London-DB (neutral_front only - very narrow) + DFFD FFHQ + OpenForensics reals
  (pending) - diversified per GOLD concern #1.
- Fake: SD1.5 (near in-set), FLUX.1-schnell, StyleGAN3-FFHQ, and DFFD GANs
  (PGGAN-v1/v2, StarGAN, FaceApp).
- Per-dataset datasheets with processing history (results/datasheets.md; docs/DATASHEET_TEMPLATE.md).
- Why diversity matters: avoids learning the London-DB artifact cluster.

## 4. Preprocessing Analysis (GOLD concern #2)
- Scaling vs cropping variants (scripts/prepare_variants.py).
- Report detection/attribution deltas between variants; large deltas => preprocessing
  artifacts leaking. PNG-only derived images; no stacked JPEG.

## 5. Detection: Real vs Fake (binary)
- DE-FAKE classifier: inference via the team's run_defake_batch.py; scored by
  score_defake_detection.py (overall + per-generator + per-category).
- DCT linear-SVM (dct_svm.py): random split + out-of-set holdout.
- Metrics: AUROC, AUPRC, balanced accuracy, precision, recall, macro-F1.

## 6. Attribution: Which Generator (multi-class)
- IMPORTANT: the provided DE-FAKE head is binary-only; there is no pretrained attribution.
  Attribution is produced by our fine-tuned head (section 7) and scored with
  eval_defake_attribution.py (in-set vs out-of-set, confusion matrices).
- GAN Fingerprints (`train_ganfp.py` + `run_ganfp_infer.py`; `lib/ganfp.py`): reproduced in PyTorch
  (residual/spectrum fingerprints + a small head); attribution scored with
  `eval_defake_attribution.py`, or documented reduced scope if the full run is infeasible.

## 7. Retraining / Fine-tuning (Phase E)
- Frozen CLIP + fine-tuned head adding FLUX/StyleGAN3 (finetune_defake_head.py), faithful
  1024-dim image+text features reusing existing BLIP captions.
- Leave-one-generator-out for true out-of-set behavior.

## 8. Generalization: In-set vs Out-of-set
- Leave-one-generator-out (leave_one_generator_out.py): forced-label distributions.
- Confidence/entropy/false-known-rate analysis (out_of_set_analysis.py).
- This is the project's central scientific contribution.

## 9. Robustness
- JPEG/blur/resize/sharpen on held-out test (robustness_perturb.py).
- Performance drop, confidence drop, label-flip rate.

## 10. Limitations
- London-DB resolution confound (tested, not just noted).
- Closed-set classifiers cannot reject unknown generators (forced labels).
- GAN-Fingerprints scope (weights availability / training cost).
- Small per-generator training set for fine-tuning.
- Risk of detecting preprocessing history rather than generator traces (mitigations: PNG,
  scaling/cropping ablation, datasheets).

## 11. Future Work
- SOTA open-set methods raised at interim: LIDA (low-bit-plane attribution) and OmniDFA
  (unified open-set detection + few-shot attribution).
- Feature fusion (CLIP + DCT + residual) via majority voting / ensembling.

## 12. Approved deviations from the interim presentation
- Dropped Phase A (reproduce Sha2023 on its original fake data): that data is not public;
  SD1.5 used as the supervisor-approved near in-set proxy.
- Enabled DE-FAKE retraining/fine-tuning (Phase E): explicitly endorsed in the review,
  correcting the earlier "pretrained-only" assumption.
- Reclassified DFFD as real+fake (its real subset feeds the diversified real class).

## Appendices
- All log files (logs/), generation scripts, patch notes, configs.
