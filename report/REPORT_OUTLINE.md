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
- Generator-spread caveat (state honestly): the 7 fake classes cover the two major paradigms
  (2 diffusion families + GAN architectures incl. a face-manipulation tool), but the spread is
  NARROW - all face-centric, temporally skewed, few architectures per paradigm, and each source
  is a single model/checkpoint. Do NOT claim broad generator coverage; findings generalize
  across paradigms only in the limited sense the data supports (see Limitations).
- Split integrity is audited, not assumed (audit_split_leakage.py): exact (SHA-256) + near-
  duplicate (perceptual-hash) checks across train/val/test, with attention to SD/FLUX sibling
  seeds; per-generator balance counts per split are reported.

## 4. Preprocessing Analysis (GOLD concern #2)
- Three variants (scripts/prepare_variants.py): "scaled" (squash - DISTORTS non-square
  images), "cropped" (center crop), "aspect" (resize shortest side + center crop - aspect-
  PRESERVING). PNG-only derived images; no stacked JPEG.
- Aspect-ratio caveat (supervisor, Dennis): squashing to 256x256 stretches non-square reals
  (CelebA 178x218, London-DB) but not the square 512 fakes, so a squash pipeline can REPLACE
  the format/resolution confound with an aspect-distortion confound. Confound-controlled runs
  use "aspect"; report the "scaled" vs "aspect" delta as the measurement of that risk.
- Confound-verification (to answer directly, not just assert): (i) metadata-only classifier
  (width/height/aspect/format, NO pixels) as an upper bound on how separable the confound is -
  `scripts/metadata_confound_probe.py` (run on raw master vs a normalized variant; the gap is
  the measurement); (ii) detection/attribution on "scaled" vs "aspect".
- Format/JPEG confound, MEASURED (raw vs controlled attribution on the aspect variant):
  removing the JPEG normalization changes in-set attribution by only ~1.4 pts (raw 96.2% ->
  controlled 94.8% top-1; balanced 95.9% -> 94.5%). So the fine-tuned head relies mostly on
  generator content, not compression/format artifacts - the confound is real but small.
- Geometry (aspect-distortion) confound, MEASURED (scaled/squash vs aspect on the SAME matched
  fine-tune): in-set attribution balanced acc 95.5% (scaled) vs 93.9% (aspect) - distortion buys
  only ~1.5 pts. DCT-SVM detection goes the OTHER way (scaled AUROC 0.761 vs aspect 0.777), i.e.
  distortion does NOT help and slightly hurts. Conclusion: the squash/aspect-distortion confound
  Dennis flagged is NOT meaningfully exploited by either the attribution head or DCT.
- Metadata-only confound probe, MEASURED (RandomForest on width/height/aspect/log-area/format,
  NO pixels; metadata_confound_probe.py): on the RAW master, real vs fake is separable at
  **balanced acc 0.79 / AUROC 0.89** without ever seeing a pixel - the format flags dominate
  (is_png 0.28 + is_jpeg 0.25 = 53% of importance; resolution adds the rest). After normalization
  to 256x256 PNG (aspect variant) it collapses to **exactly chance (0.50 / 0.50, all feature
  importances 0)**. The 0.89 -> 0.50 AUROC gap IS the confound, and the pipeline removes it
  entirely. Per-source detail: raw metadata mislabels FFHQ reals as fake (they are PNG like the
  fakes), while JPEG reals (CelebA/London-DB) stay real - confirming format, not content, drives
  the raw leak. This is the direct, quantified answer to the supervisor's question.

## 5. Detection: Real vs Fake (binary)
- DE-FAKE classifier: inference via run_defake_batch.py; scored by score_defake_detection.py
  (overall + per-generator + per-category + the fixed/validation-selected/oracle threshold blocks).
- DCT linear-SVM (dct_svm.py): random split + out-of-set holdout.
- Metrics: AUROC, AUPRC, balanced accuracy, precision, recall, macro-F1.
- Result so far (pretrained DE-FAKE, balanced 722 real / 724 fake): AUROC 0.713, balanced acc
  0.591, fake recall 0.80, real specificity 0.378 (CelebA 27.5% / London-DB 12.7% / FFHQ 57.3%);
  StyleGAN3 is the fake blind spot (46%). AUROC is stable vs the 202-real baseline (0.710),
  so the low specificity is systematic (domain shift to real faces), not a London-DB artifact.
  Key framing: ranking is moderate (AUROC ~0.71) but the default 0.5 threshold is fake-biased.
- Confound-controlled DE-FAKE (SAME 722/724 set, aspect variant vs raw originals - apples-to-
  apples): AUROC 0.713 -> **0.674** (-0.04), balanced acc 0.591 -> 0.560 (0.641 at Youden's J),
  real specificity 0.378 -> 0.292. So the raw format/geometry confound gave DE-FAKE only ~4
  AUROC points; removing it does NOT rescue the weak face detection. StyleGAN3 stays the blind
  spot (0.46 -> 0.51). This is the measured, honest answer for the detector: confound present
  but small.
- Cross-method on the CONTROLLED (aspect) set: **DCT-SVM beats DE-FAKE** for binary detection
  (AUROC 0.777 / balanced 0.703 vs 0.674 / 0.560). Frequency artifacts generalize better than
  CLIP-semantic features on normalized data - a concrete argument for the fusion/future-work
  section.
- DCT confound delta (clean, matched 1446 set, scaled vs aspect): AUROC 0.761 -> 0.777, balanced
  0.697 -> 0.703. So for DCT the aspect-distortion confound is negligible (if anything the
  controlled variant is slightly BETTER) - DCT is reading genuine frequency artifacts, not
  distortion.
- Out-of-set generalization fails for BOTH methods: DCT-SVM on held-out generators (FLUX +
  StyleGAN3, matched 1446 set) collapses to ~chance (balanced 0.54, AUROC 0.62), mirroring the
  DE-FAKE attribution collapse - neither method transfers to unseen generators.
- Threshold hygiene (score_defake_detection.py -> overall.thresholds): report threshold-free
  AUROC as the primary detection number, and the `validation_selected` operating point
  (threshold chosen on a seeded stratified val holdout, metrics on disjoint test rows) as the
  reportable balanced accuracy. The Youden's-J "best" number is kept ONLY as a labeled
  `oracle_upper_bound` (fit on all rows -> optimistic, non-achievable); never quote it as the result.
- Uncertainty: attach 95% stratified-bootstrap CIs to AUROC / balanced accuracy / per-generator
  detection rate (bootstrap_metrics.py). With ~22 fake images/class the CIs are wide; report
  them so the DE-FAKE-vs-DCT gap is read against its uncertainty.
- Model comparison is PAIRED on the shared test paths (compare_models_significance.py): McNemar
  exact test on discordant pairs + a paired bootstrap of the AUROC difference, so "DCT-SVM beats
  DE-FAKE" is stated with a p-value / CI on the difference, not two independent point estimates.

## 6. Attribution: Which Generator (multi-class)
- IMPORTANT: the provided DE-FAKE head is binary-only; there is no pretrained attribution.
  Attribution is produced by our fine-tuned head (section 7) and scored with
  eval_defake_attribution.py (in-set vs out-of-set, confusion matrices). This is the primary
  (and current) attribution method.
- GAN-Fingerprints (Yu2019) is PARKED as an optional second method (deprioritized per the
  supervisor: DE-FAKE multi-class first). A PyTorch re-implementation exists on the
  `ganfp-integrated` branch; only add it back if time allows.
- Results (fine-tuned head, controlled/JPEG-normalized aspect variant; 6 classes = 3 reals +
  SD1.5/FLUX/StyleGAN3): in-set test top-1 94.8% / balanced 94.5% (n=210). Per-class recall:
  FLUX 100%, SD1.5 100%, FFHQ 96.7%, London-DB 95%, CelebA 93.8%, StyleGAN3 81.8% (weakest -
  all 4 of its errors -> FFHQ). Fake-only in-set attribution (eval): balanced 93.9% (n=66).
  Caveat: small per-fake-class support (~22 test each) -> report recalls with that uncertainty.
- Uncertainty is quantified, not just noted: bootstrap_metrics.py gives 95% CIs on top-1 /
  macro-F1 / balanced accuracy and on each per-class recall; seed_sweep.py re-splits + re-trains
  the head over 10 seeds and reports mean/std/CI (so e.g. "StyleGAN3 recall 0.82" is quoted with
  its across-seed spread, not as a single fragile point estimate).

## 7. Retraining / Fine-tuning (Phase E)
- Frozen CLIP + fine-tuned head adding FLUX/StyleGAN3 (finetune_defake_head.py), faithful
  1024-dim image+text features reusing existing BLIP captions.
- Leave-one-generator-out for true out-of-set behavior.

## 8. Generalization: In-set vs Out-of-set
- Leave-one-generator-out (leave_one_generator_out.py): forced-label distributions.
- Confidence/entropy/false-known-rate analysis (out_of_set_analysis.py).
- This is the project's central scientific contribution.
- Out-of-set force-scoring (4 unseen DFFD GANs: FaceApp/PGGAN-v1/PGGAN-v2/StarGAN, n=400):
  top-1 = 0 BY CONSTRUCTION (the true class is absent from the label space; explain this so it
  is not read as a bug). The informative signal is the forced distribution + confidence:
  ~98% (393/400) of unseen-GAN images are attributed to a REAL class (CelebA/FFHQ) at mean
  confidence 0.82; false-known rate 0.96 @0.5, 0.76 @0.7, 0.44 @0.9. So unseen GAN fakes would
  largely pass as authentic. Entropy DOES separate populations (in-set 0.19 vs out-of-set 0.47),
  so an entropy/confidence rejection rule could recover some unseen fakes - only a partial fix
  (44% remain confident at 0.9).
- LOGO (retrain WITHOUT the target) exposes a family asymmetry - THE key finding:
  * Unseen DIFFUSION (FLUX held out, n=108): forced to the other diffusion model SD1.5 81.5% of
    the time; ~94% land on a FAKE class -> detection survives, misattribution is family-
    consistent (mean conf 0.79, FKR 0.85 @0.5).
  * Unseen GAN (StyleGAN3 held out, n=108): forced to FFHQ 85% (97% to real classes overall) ->
    detection FAILS; StyleGAN3 collapses onto its FFHQ training source (mean conf 0.76, FKR 0.88).
  * ROBUSTNESS: the same asymmetry reproduces on the raw (JPEG-aug OFF) baseline - StyleGAN3->FFHQ
    102/108 (FKR 0.95), FLUX->SD1.5 81/108 (FKR 0.87). So the GAN-collapse is not an artifact of
    the JPEG normalization.
- Unifying mechanism: face GANs trained on real face datasets (StyleGAN3<-FFHQ; PGGAN/StarGAN/
  FaceApp on face data) collapse onto the real manifold, whereas diffusion generalizes within
  its family. This is consistent across THREE independent measurements: binary detection
  (StyleGAN3 46% fake recall), in-set attribution (StyleGAN3->FFHQ errors), and out-of-set/LOGO
  (unseen GANs -> real). That triangulation is the report's strongest claim.

## 9. Robustness
- JPEG/blur/resize/sharpen on the held-out test split (n=290, aspect variant; robustness_perturb.py).
  Metrics: accuracy drop (clean-perturbed), prob_fake drop, per-image label-flip rate.
- Headline: aggregate accuracy is STABLE under every perturbation - clean 0.555, perturbed range
  0.524-0.600 (max |drop| 0.045). No catastrophic degradation. BUT the clean detector is already
  weak/fake-biased (~chance), so aggregate stability is NOT strong evidence of robustness.
- The per-image view contradicts the aggregate: predictions are NOT stable. JPEG q30 flips
  **33.4%** of labels and sharpening **30.3%** (jpeg50 25.2%, blur2 18.6%, others 11-14%). The
  aggregate looks flat only because the flips are roughly symmetric, not because decisions hold.
- Direction: high-frequency edits (JPEG q30, sharpen) DROP prob_fake the most (~0.21) and, because
  the baseline over-calls faces "fake" (real specificity ~0.29, section 5), this actually RAISES
  accuracy slightly (jpeg30 0.60, sharpen 0.60). Low-pass edits (blur, resize) nudge prob_fake up
  a little and barely move accuracy. So robustness is entangled with the threshold/fake-bias
  problem: perturbations that push toward "real" happen to help a fake-biased detector.
- Takeaway for the report: DE-FAKE's aggregate score is perturbation-insensitive but its
  individual decisions are volatile (up to 1/3 flip under mild JPEG), which is the honest robustness
  characterization - stable-looking metric, unstable predictions.

## 10. Limitations
- Format/resolution confound: fakes are PNG/512, reals mix JPEG + varied sizes. Controlled by
  uniform PNG + JPEG augmentation + aspect-preserving resize; still a boundary of the work.
- Aspect-ratio distortion (supervisor-flagged): naive squashing distorts non-square reals only;
  we mitigate with the "aspect" variant and report the scaled-vs-aspect delta.
- Confound exploitation QUANTIFIED (fully): metadata-only upper bound = AUROC 0.89 raw -> 0.50
  normalized (format flags dominate); format axis on the actual detector (raw vs controlled
  DE-FAKE) = ~4 AUROC points (0.713->0.674); geometry axis (scaled vs aspect) = ~1.5 pts for
  attribution and slightly NEGATIVE for DCT. Net: the confound is strongly present in the RAW
  metadata (0.89) but the normalized pipeline erases it (0.50), and the residual effect on the
  real models is small - removing it does not rescue detection.
- London-DB resolution confound (tested, not just noted).
- Closed-set classifiers cannot reject unknown generators (forced labels). QUANTIFIED: ~98% of
  unseen-GAN images are confidently assigned a REAL class (false-known rate 0.96 @0.5); an
  entropy-based rejection is only a partial mitigation (44% still confident @0.9).
- GAN-Fingerprints attribution is out of scope for the current report (parked on the
  `ganfp-integrated` branch); DE-FAKE multi-class attribution is the method of record.
- Small per-generator training set for fine-tuning (~22 test images/fake class). All headline
  numbers are therefore reported with 95% bootstrap CIs + a 10-seed sweep; treat point estimates
  as indicative, not precise.
- Narrow generator spread: 7 face-centric generators (2 diffusion families + a few GAN
  architectures + one face-manipulation tool), each a single checkpoint, temporally skewed. The
  paradigm-level claims (diffusion generalizes within family; face GANs collapse onto the real
  manifold) are supported, but broad cross-generator coverage is NOT claimed.
- Threshold dependence: DE-FAKE's default 0.5 is miscalibrated (fake-biased) on faces; results
  are reported at a validation-selected operating point plus threshold-free AUROC, with the
  Youden's-J number labeled as a non-achievable oracle upper bound only.
- Split leakage: audited via exact + perceptual-hash duplicate checks across splits
  (audit_split_leakage.py); report the audit result. Note we do NOT use identity-group splitting
  (no identity labels; reals drawn from large pools), which the audit substitutes for.
- OpenForensics (real+fake in one image, a strong confound control) planned, pending the JSON
  upload + a face-extraction step.

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
