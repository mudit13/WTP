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
- Real: London-DB (neutral_front only - very narrow) + DFFD FFHQ + CelebA + OpenForensics reals
  (in-the-wild faces, capped 300) - diversified per GOLD concern #1 (OpenForensics reals added on
  Dennis's strongest steer; they become a TRAINED real class).
- Fake: SD1.5 (near in-set), FLUX.1-schnell, StyleGAN3-FFHQ, and DFFD GANs
  (PGGAN-v1/v2, StarGAN, FaceApp). OpenForensics-fake is added as an OUT-OF-SET (unseen
  manipulation) fake, cropped from the same source photos as the OF reals.
- Per-dataset datasheets with processing history (results/datasheets.md; docs/DATASHEET_TEMPLATE.md).
- Why diversity matters: avoids learning the London-DB artifact cluster.
- Generator-spread caveat (state honestly): the 7 fake classes cover the two major paradigms
  (2 diffusion families + GAN architectures incl. a face-manipulation tool), but the spread is
  NARROW - all face-centric, temporally skewed, few architectures per paradigm, and each source
  is a single model/checkpoint. Do NOT claim broad generator coverage; findings generalize
  across paradigms only in the limited sense the data supports (see Limitations).
- Split integrity is audited, not assumed (audit_split_leakage.py): exact (SHA-256) + near-
  duplicate (perceptual-hash) checks across train/val/test, with attention to SD/FLUX sibling
  seeds; per-generator balance counts per split are reported. A second, dHash-blind audit
  (audit_openforensics_coupling.py) specifically targets OpenForensics same-source-photo
  real/fake coupling (see Limitations) that the perceptual-hash check cannot see.

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
- **DCT-SVM train/test-boundary fix (BLOCKING, resolved in code -- re-run needed).**
  `dct_svm.py --mode random` used to draw its OWN internal split, stratified on the BINARY
  real/fake label; the robustness test set (`make_split.py` -> `test_index.csv`) is stratified
  on the 12-class GENERATOR column. Same seed + same test_size do NOT guarantee the same
  partition when the stratification column differs, so a fraction of `test_index.csv`'s rows
  were inside the SVM's own training set -- `robustness_perturb.py`'s "clean" DCT baseline was
  partly measuring the SVM on (perturbed) training data. `dct_svm.py` now accepts
  `--test_index results/test_index.csv`, which makes its train/test boundary IDENTICAL to the
  shared split instead of re-deriving one. Every DCT number that feeds the robustness table
  (§9) must be regenerated with this flag before being reported; the old 0.907-clean / 0.666
  from-scratch discrepancy was exactly this leak and should disappear (both numbers should
  converge to the SAME split-consistent baseline).
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
- OpenForensics crop-size confound, MEASURED (metadata-only probe on 600 OF crops,
  metadata_confound_probe.py --source_filter openforensics): RAW OF is real/fake separable from
  crop GEOMETRY alone at balanced acc 0.608 / AUROC 0.634 (aspect + area, NOT format - is_png/
  is_jpeg importance 0, since OF is uniformly JPEG q95). After aspect normalization it drops to
  exactly chance (0.500 / 0.500). So OF is used ONLY on the aspect variant; raw-geometry OF numbers
  are not reported. Same measure-then-remove logic as the format confound, on the size axis.
- OpenForensics same-photo confound benchmark (pending run): OF real and fake faces are cropped
  from the SAME source photos through one JPEG q95 pipeline, so within OF there is no format/source
  gap between classes. Report detection on the OF subset as the cleanest confound-controlled test,
  but only after the OF-only metadata probe (metadata_confound_probe.py --source_filter
  openforensics) confirms crop SIZE does not leak the label (~0.5); if it does, use the aspect
  variant.

## 6. Attribution: Which Generator (multi-class)
- IMPORTANT: the provided DE-FAKE head is binary-only; there is no pretrained attribution.
  Attribution is produced by our fine-tuned head (section 7) and scored with
  eval_defake_attribution.py (in-set vs out-of-set, confusion matrices). This is the primary
  (and current) attribution method.
- GAN-Fingerprints (Yu2019-inspired) is the SECOND attribution method (on main): a PyTorch
  re-implementation with two paths (residual/spectrum features + MLP; and an end-to-end CNN with
  a fixed Fridrich-Kodovsky SRM front-end), benchmarked head-to-head against DE-FAKE on one
  shared split (benchmark_attribution.py). Honestly "Yu2019-inspired", not a byte-faithful port.
  DE-FAKE multi-class remains the primary attribution method; GAN-fp targets the GAN-specific
  traces CLIP misses.
- **Benchmark comparison is NOT apples-to-apples by default (BLOCKING, resolved -- re-run
  needed).** GAN-fp (Path A/B) trains on all 12 classes present in the index (incl.
  PGGAN-v1/v2, StarGAN, FaceApp, OpenForensics-fake); the DE-FAKE head trains on only its own
  7 classes (4 reals + SD1.5/FLUX/StyleGAN3, see §7). When `benchmark_attribution.py` ingests a
  DE-FAKE per-image CSV via `--defake_csv`, DE-FAKE structurally CANNOT output the right label
  for the 5 classes outside its training set -- part of any GAN-fp-vs-DE-FAKE gap in the
  comparison table is that structural disadvantage, not necessarily worse learning. Fix (both
  landed in `benchmark_attribution.py`): (1) every comparison row now carries a
  `classes_trained_on` / `n_classes_trained_on` field (DE-FAKE's is read back from its own
  `finetune_metrics.json`), so the table is self-documenting instead of implying equivalence;
  (2) re-run restricted to the SAME 7 classes for a genuinely fair number:
  `benchmark_attribution.py --classes "London-DB" "FFHQ" "CelebA" "OpenForensics" "SD1.5"
  "FLUX.1-schnell" "StyleGAN3-FFHQ"`. Report BOTH the full-12-class number (each method's
  best-case under its own regime) and the matched-7-class number (the fair head-to-head) side
  by side, labeled accordingly.
- Results (fine-tuned head, controlled/JPEG-normalized aspect variant; 6 classes = 3 reals +
  SD1.5/FLUX/StyleGAN3). NOTE: adding OpenForensics as a 4th real class makes this a 7-class run;
  the numbers below predate that and must be regenerated after the OF re-run. In-set test top-1
  94.8% / balanced 94.5% (n=210). Per-class recall:
  FLUX 100%, SD1.5 100%, FFHQ 96.7%, London-DB 95%, CelebA 93.8%, StyleGAN3 81.8% (weakest -
  all 4 of its errors -> FFHQ). Fake-only in-set attribution (eval): balanced 93.9% (n=66).
  Caveat: small per-fake-class support (~22 test each) -> report recalls with that uncertainty.
- Uncertainty is quantified, not just noted: bootstrap_metrics.py gives 95% CIs on top-1 /
  macro-F1 / balanced accuracy and on each per-class recall; seed_sweep.py re-splits + re-trains
  the head over 10 seeds and reports mean/std/CI (so e.g. "StyleGAN3 recall 0.82" is quoted with
  its across-seed spread, not as a single fragile point estimate).
- **Single source of truth for Path A (GAN-fp feature+MLP) numbers (BLOCKING -- reconcile
  before quoting either).** `train_ganfp.py` (standalone) and `benchmark_attribution.py`
  (Path A inside the head-to-head) use the IDENTICAL split mechanism (`defake_head.
  stratified_split`, content-stable hash on `full_path`, same config seed) and an
  index-content-hashed feature-cache signature that refuses to silently reuse a cache built
  from a different index/config -- so a standalone run and a benchmark run on the SAME
  `--index`/`--classes`/`--features_cache` MUST produce the same number. A large gap (e.g. a
  standalone top1=0.378/balAcc=0.414 vs a benchmark top1=0.756/balAcc=0.713) is not explainable
  by hyperparameters alone at 40 epochs each; it is almost certainly a STALE standalone number
  from an earlier, smaller run (e.g. the 200-image toy set or an index predating a class-set
  change -- see the toy-vs-real gap already documented in `report/GANFP_REPORT.md` section 5-6,
  which is the same class of artifact). Before quoting a standalone Path A number anywhere in
  the report: either (a) re-run `train_ganfp.py` with the EXACT `--index`, `--classes`, and a
  FRESH (`--recompute_features`) cache used by the current benchmark run and confirm the numbers
  match within run-to-run variance, or (b) drop the standalone appendix number entirely and cite
  `benchmark_metrics.json`'s `path_a` block as the only reported Path A result (simplest and
  removes the duplicate-reporting risk going forward).

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
  is not read as a bug). **Footnote to attach to every out-of-set top-1 cell in this section's
  tables: "top-1 = 0.000 is definitional -- the head's output space contains no unseen class
  label, so top-1 cannot be nonzero by construction. The meaningful metric is the false-known
  rate below, not this cell."** The informative signal is the forced distribution + confidence:
  ~98% (393/400) of unseen-GAN images are attributed to a REAL class (CelebA/FFHQ) at mean
  confidence 0.82; false-known rate 0.96 @0.5, 0.76 @0.7, 0.44 @0.9. So unseen GAN fakes would
  largely pass as authentic. Entropy DOES separate populations (in-set 0.19 vs out-of-set 0.47),
  so an entropy/confidence rejection rule could recover some unseen fakes - only a partial fix
  (44% remain confident at 0.9).
- **"Leave-One-Generator-Out" naming caveat (reframed).** `leave_one_generator_out.py`'s DEFAULT
  run only holds out FLUX.1-schnell and StyleGAN3-FFHQ - the two `finetune_new_classes`, i.e.
  generators the regular head IS otherwise trained on. That default is more accurately
  **"Leave-New-Class-Out (FLUX, StyleGAN3)"**, not a full leave-one-generator-out sweep: it
  tests whether the two most-recently-added classes could be dropped and the model still
  recognise them, which is a narrower (and more favorable-looking) claim than "the model
  generalizes when any one of its trained classes is removed." A proper LOGO holds out EVERY
  trained class in turn, INCLUDING a real class (e.g. CelebA), to test whether the model
  confuses one real source with another when it's absent. `--all_trained_classes` (added to the
  script) runs that full sweep in one command; report BOTH: the narrow FLUX/StyleGAN3 numbers
  under the corrected heading below, and the full-sweep false-known rate averaged across all 7
  trained classes as the actual generalization number.
- LOGO (retrain WITHOUT the target) exposes a family asymmetry - THE key finding. Below is the
  Leave-New-Class-Out (FLUX, StyleGAN3) result; §8b (pending the `--all_trained_classes` re-run)
  reports the full sweep including a held-out real class:
  * Unseen DIFFUSION (FLUX held out, n=108): forced to the other diffusion model SD1.5 81.5% of
    the time; ~94% land on a FAKE class -> detection survives, misattribution is family-
    consistent (mean conf 0.79, FKR 0.85 @0.5).
  * Unseen GAN (StyleGAN3 held out, n=108): forced to FFHQ 85% (97% to real classes overall) ->
    detection FAILS; StyleGAN3 collapses onto its FFHQ training source (mean conf 0.76, FKR 0.88).
  * ROBUSTNESS: the same asymmetry reproduces on the raw (JPEG-aug OFF) baseline - StyleGAN3->FFHQ
    102/108 (FKR 0.95), FLUX->SD1.5 81/108 (FKR 0.87). So the GAN-collapse is not an artifact of
    the JPEG normalization. (Appendix, optional: a raw-GEOMETRY LOGO on the scaled/squash index
    with JPEG-aug off isolates the remaining geometry axis - PIPELINE step 9; the qualitative
    collapse is expected to persist since in-set attribution showed geometry buys only ~1.5 pts.)
- **§8b (pending re-run): full LOGO sweep, `--all_trained_classes`.** Holds out each of the 7
  trained classes in turn (4 reals + SD1.5/FLUX/StyleGAN3), including at least one held-out
  REAL class (e.g. CelebA -> does the model confuse it with London-DB/FFHQ/OpenForensics when
  CelebA is absent from training?). Report the false-known rate averaged over all 7 targets,
  not just the two diffusion/GAN targets above, as the headline generalization number.
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

### 9.2 DCT-SVM detection robustness (re-run required before quoting)
- Same perturbation set, scored via `dct_svm.py --mode predict` against the fitted SVM. MUST be
  fit with `--test_index results/test_index.csv` (see §5's DCT train/test-boundary fix) so the
  "clean" baseline here is evaluated on genuinely held-out rows, not partly on the SVM's own
  training data. Any DCT-SVM clean/perturbed numbers generated BEFORE this fix (e.g. a clean
  balanced accuracy far above the random-split §5 number, such as an 0.907-vs-0.666-style gap
  between the robustness-clean number and the from-scratch random-split number) are invalid and
  must be regenerated.
- AUROC is now reported alongside balanced accuracy for every perturbation (`robustness_perturb.py
  --mode score` computes `auroc_clean`/`auroc_perturbed`/`auroc_drop` whenever a numeric
  `--conf_col` is available, e.g. the SVM's decision-function `score` column), so the
  robustness table shows whether a perturbation degrades RANKING quality, not just the
  balanced accuracy at a fixed operating point. **Fixed a follow-on gap in that same change:**
  the accuracy/AUROC block was gated on a ground-truth `label_col` lookup that only recognized
  DE-FAKE's `label` ("real"/"fake" string) column; `dct_svm.py`'s `dct_per_image.csv` has no
  `label` column at all (only a numeric `y_true`), so `label_col` silently resolved to `None`
  for every DCT drop JSON - which would have shipped with only `n`/`label_flip_rate` and NONE
  of `accuracy_clean`/`accuracy_perturbed`/`performance_drop`/`auroc_*`, exactly the fields
  §9.2's DCT rows need, while DE-FAKE's JSONs got the full block. `score()` now also recognizes
  a `y_true`/`y_true_clean` numeric column as ground truth. Covered by
  `tests/test_robustness_perturb.py` (one case per schema).

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
- GAN-Fingerprints (Yu2019-inspired) is included as a SECOND attribution method (on main),
  benchmarked against DE-FAKE; it is a re-implementation of the method, not a byte-faithful port.
  DE-FAKE multi-class attribution remains the method of record.
- Small per-generator training set for fine-tuning (~22 test images/fake class). All headline
  numbers are therefore reported with 95% bootstrap CIs + a 10-seed sweep; treat point estimates
  as indicative, not precise.
- **Statistical significance (BLOCKING, code ready, run pending).** `bootstrap_metrics.py` and
  `compare_models_significance.py` exist and are wired into the runbook (PIPELINE.md step 7b)
  but have not been executed. With ~20-22 test images per fake class, a single flipped
  prediction moves a per-class recall by ~5 points, so headline deltas in §8/§9 could be within
  sampling noise. Before presenting any cross-method comparison ("DCT-SVM beats DE-FAKE",
  "GAN-fp CNN beats Path A", etc.) as a finding, run both scripts and quote the 95% CI /
  McNemar p-value alongside the point estimate, not the point estimate alone.
- **Hyperparameter search (documented, not exhaustive).** GAN-fp CNN channels ([16,32,64] vs
  [32,64,128] vs [48,96,192]) WERE swept informally on the 1626-image set ([16,32,64] won,
  val_top1=0.736; the larger configs overfit) -- this is a real, if informal, grid search and
  is now stated as such rather than left as a config comment. The DE-FAKE MLP head
  (hidden=256, dropout=0.3, lr=1e-3) and the DCT-SVM regularization (C=1.0) were NOT
  cross-validated; both are fixed at defaults from the source papers (DE-FAKE / Frank2020).
  Treat their reported numbers as indicative of the method, not as the best achievable
  performance -- a supervisor should not read "DCT-SVM AUROC 0.777" as "the best a linear-SVM
  detector can do on this data."
- **CLIP+BLIP semantic confound (deferred, not measured).** The fine-tuned attribution head's
  1024-dim input is 512-dim CLIP image features concatenated with 512-dim BLIP CAPTION
  features. BLIP captions describe image CONTENT (scene, subject, composition), not generation
  artifacts; real and AI-generated faces differ in subject matter and setting independently of
  any forensic trace, so part of the head's signal could be "what is this a picture of" rather
  than "what generated this." No ablation isolating visual-only CLIP (512-dim) from image+text
  CLIP (1024-dim) has been run; it would need a feature re-extraction + head retrain, which is
  deferred to future work given project scope. State this explicitly rather than implying the
  1024-dim head is purely reading forensic artifacts.
- Narrow generator spread: 7 face-centric generators (2 diffusion families + a few GAN
  architectures + one face-manipulation tool), each a single checkpoint, temporally skewed. The
  paradigm-level claims (diffusion generalizes within family; face GANs collapse onto the real
  manifold) are supported, but broad cross-generator coverage is NOT claimed.
- Threshold dependence: DE-FAKE's default 0.5 is miscalibrated (fake-biased) on faces; results
  are reported at a validation-selected operating point plus threshold-free AUROC, with the
  Youden's-J number labeled as a non-achievable oracle upper bound only.
- **Split leakage (BLOCKING, code ready, run pending).** `audit_split_leakage.py` (exact
  SHA-256 + near-duplicate dHash checks across train/val/test, plus per-generator/per-source
  balance counts) exists but has not been executed. Known near-duplicate risk vectors to check
  specifically once run: CelebA (202k pool, 320 sampled, seeded -> expected low risk but
  confirm), FFHQ vs StyleGAN3-FFHQ (StyleGAN3 was TRAINED on FFHQ, so its outputs may
  perceptually resemble specific FFHQ training images even though they are technically distinct
  files), and OpenForensics same-source-photo pairs (see below -- a DIFFERENT mechanism the
  dHash check cannot catch). Run before the supervisor meeting either way: a clean result is
  reportable as a positive finding, a dirty one is a critical finding that must be reported, not
  a footnote. We do NOT use identity-group splitting (no identity labels; reals drawn from large
  pools), which this audit substitutes for.
- **OpenForensics same-source-photo coupling (RESOLVED: group-aware split, re-extraction +
  re-run pending).** OpenForensics scene photos contain BOTH genuine and manipulated face
  annotations; our extractor (`extract_openforensics.py`, run with `--splits Val
  --per_class_limit 300 --seed 42` -- Val split ONLY, none of Train/Test-Dev/Test-Challenge)
  used to crop and name each face by ANNOTATION id, dropping the source IMAGE id, so a real
  crop and a fake crop from the SAME source photo (shared camera/lighting/JPEG history) could
  land on opposite sides of the train/test split -- a leak the dHash near-duplicate audit
  cannot catch, because a real face region and a fake face region from one photo are different
  pixels under perceptual hashing even though they share acquisition statistics. This is
  SEPARATE from the crop-size confound already measured and controlled (§4/§5, OF raw balanced
  accuracy 0.608 -> chance after aspect normalization). Chosen fix (the scientifically cleanest
  option, implemented in code): `extract_openforensics.py` now records each crop's source
  `image_id` (both in `openforensics_metadata.csv` and a dedicated `openforensics_groups.csv`
  sidecar); `scripts/lib/defake_head.py`'s `stratified_split` gained a `groups=` argument so
  every crop sharing a source photo is kept on the SAME split side (backward-compatible: any
  row without a group sidecar entry falls back to its own path as a singleton group, i.e.
  splits exactly as before); every split-consuming script (`finetune_defake_head.py`,
  `train_ganfp.py`, `benchmark_attribution.py`, `leave_one_generator_out.py`, `make_split.py`)
  auto-loads the sidecar. `audit_split_leakage.py` now also reports `group_straddle` (expected
  0) as a live regression check, and `audit_openforensics_coupling.py` independently re-derives
  the coupling from the raw polygon JSON to cross-validate the sidecar. **Remaining work (needs
  the GPU server): re-run `extract_openforensics.py` to regenerate the crops with the sidecar,
  then rebuild the master index and every downstream stage** (docs/PIPELINE.md step 1b) --
  every prior OpenForensics-inclusive number was computed on the OLD, non-group-aware split and
  must be regenerated before being reported as final.
  **Deeper fix (section 17, found by empirically comparing splits, not just inspection):** the
  group/singleton decision was initially based on counting a group's members WITHIN the array a
  given caller passed in, not on the group id itself. Because `finetune_defake_head.py` restricts
  to trained classes before splitting (dropping the out-of-set `OpenForensics-fake` sibling of a
  coupled pair), an `OpenForensics` real row could look like a singleton there but a 2-member
  group in `make_split.py`'s unrestricted call - the two functions could (and did, verified with
  a synthetic check) disagree on that row's split side, reopening the exact leak the group-aware
  fix was meant to close, specifically for OpenForensics reals paired with an out-of-set fake.
  Fixed: grouped-ness is now an identity check (`groups[i] != keys[i]`, i.e. "does this row have
  an explicit sidecar-assigned id") rather than a co-occurrence count, so a row's bucket depends
  only on `(group_id, seed)` and is identical across every caller regardless of how that caller
  filtered its population first. Two regression tests added
  (`tests/test_defake_head.py::test_group_membership_is_id_based_not_call_population_based` and
  `::test_group_decision_matches_across_differently_filtered_calls`).
  **Two real infrastructure bugs found and fixed on the actual first server run (sections
  21-22), plus a THIRD apparent failure that turned out not to be a bug at all (section 23):**
  (1) `extract_openforensics.py` (host, required for `/vol1`) recorded the sidecar's `full_path`
  using the HOST's absolute path prefix, while `build_master_index.py` (container) records the
  SAME physical files under the CONTAINER's prefix - fixed via `--record_prefix`. (2)
  `prepare_variants.py` (which produces `index_aspect.csv`, the index every stage actually
  trains on) rewrites `full_path` to a NEW derived variant file, keeping the original extraction
  path only in `source_path`; the sidecar was written against that original path, so a variant
  index's `full_path` could never match it regardless of prefix - fixed by resolving each row's
  group-map lookup key via `source_path` first (`io_utils.apply_group_map_with_lookup`). (3)
  After both fixes, `audit_openforensics_coupling.py` STILL reported the coupled photos as
  "100% straddling" - traced (not re-guessed) to `OpenForensics-fake` being a permanently
  out-of-set generator BY DESIGN (never in `in_set_generators`/`finetune_new_classes`, so it
  never enters the train/val/test split algorithm at all); ANY coupled pair will show a
  different split label for its real vs. fake member almost by definition, independent of
  whether grouping works. The metric that actually matters - of the 12 coupled photos, how many
  have their REAL sibling specifically in `train` (the model's weights were actually fit on it,
  vs. `val`/`test` which are not leakage either) - is **10/12, i.e. 10 of 300 (3.3%) of the
  out-of-set `OpenForensics-fake` evaluation crops have a same-source-photo real crop in
  training.** The model never saw those exact fake images, only a different face crop from the
  same photograph.
  **Decision: document, not further re-engineer.** Given the small absolute magnitude (3.3%)
  and the narrow leakage mechanism, this is reported as a measured, bounded limitation rather
  than triggering another pipeline change/re-run. State this explicitly in the report:
  *"12 of the 600 sampled OpenForensics crops (300 real + 300 fake) share a source photograph
  with a crop of the opposite label. Because `OpenForensics-fake` is deliberately held fully
  out-of-set (never trained on), 10 of these 12 coupled photos have their real sibling in the
  training set - meaning 10 of the 300 (3.3%) out-of-set OpenForensics-fake evaluation images
  have a different face crop from the same source photograph in training. This narrows, but
  does not eliminate, the 'genuinely unseen manipulation type' claim for those 10 images; the
  group-aware split fix (verified working correctly for classes that ARE part of the
  train/val/test split) cannot address this specific case because the fake side is never part
  of that split to begin with."* `audit_openforensics_coupling.py`'s output JSON now reports
  `n_real_fake_pairs_train_fit_leak` (10 here) as the headline metric instead of the broad,
  misleading `n_real_fake_pairs_straddling_splits` (12 here, ~100% whenever one label is
  permanently out-of-set, regardless of whether grouping works).
- OpenForensics wiring: reals are added as a TRAINED real class to diversify the narrow real class
  (Dennis's #1 steer), OF-fake is kept out-of-set (unseen manipulation), and the same-photo pairs
  are a strong within-dataset confound control. Consequence to state: OF reals are therefore NOT a
  held-out real distribution (that role is given up for diversification); OF-fake remains the
  held-out unseen-fake probe. Guarded by an OF-only crop-size confound check before use.
- **Datasheets incomplete (BLOCKING for an MSc submission).** `results/datasheets.md`
  (`make_datasheets.py`) auto-fills count/resolution/format per dataset but every "Provenance
  (manual)" field (generation/sensor pipeline, source resize/crop, JPEG quality, alignment
  method, license) is still a TODO placeholder for every dataset. At minimum, fill in: the SD1.5
  prompt set / sampler / steps used for `sd15_txt2img`, the StyleGAN3 checkpoint + truncation
  parameter used for `stylegan3`, the FLUX.1-schnell prompt set, and license/access notes for
  CelebA / FFHQ(DFFD) / London-DB / OpenForensics (already partly captured in `docs/PROJECT_LOG.md`
  section 8 for DFFD -- port that into the datasheet, then fill in the rest).
- **No SOTA baseline run (acknowledged explicitly, not just by omission).** We did not run
  Wang2020 (CNNDetect) or Ojha2023 (UniversalFakeDetect) on our split. Be ready to say why:
  DCT-SVM (Frank2020) and the GAN-fp CNN (Yu2019-inspired, SRM front-end per Fridrich-Kodovsky
  2012) are our from-scratch equivalents of that same forensic/frequency family, not a
  substitute for running the original released code -- reproducing a third-party repo's exact
  training recipe on our data was out of scope for the time available, not an oversight.

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
