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
  HONEST current status: the JPEG-format axis is now measured; the scaled-vs-aspect geometry
  ablation and the metadata-only classifier are still pending.

## 5. Detection: Real vs Fake (binary)
- DE-FAKE classifier: inference via run_defake_batch.py; scored by score_defake_detection.py
  (overall + per-generator + per-category + best-threshold).
- DCT linear-SVM (dct_svm.py): random split + out-of-set holdout.
- Metrics: AUROC, AUPRC, balanced accuracy, precision, recall, macro-F1.
- Result so far (pretrained DE-FAKE, balanced 722 real / 724 fake): AUROC 0.713, balanced acc
  0.591, fake recall 0.80, real specificity 0.378 (CelebA 27.5% / London-DB 12.7% / FFHQ 57.3%);
  StyleGAN3 is the fake blind spot (46%). AUROC is stable vs the 202-real baseline (0.710),
  so the low specificity is systematic (domain shift to real faces), not a London-DB artifact.
  Key framing: ranking is moderate (AUROC ~0.71) but the default 0.5 threshold is fake-biased.

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
- Unifying mechanism: face GANs trained on real face datasets (StyleGAN3<-FFHQ; PGGAN/StarGAN/
  FaceApp on face data) collapse onto the real manifold, whereas diffusion generalizes within
  its family. This is consistent across THREE independent measurements: binary detection
  (StyleGAN3 46% fake recall), in-set attribution (StyleGAN3->FFHQ errors), and out-of-set/LOGO
  (unseen GANs -> real). That triangulation is the report's strongest claim.

## 9. Robustness
- JPEG/blur/resize/sharpen on held-out test (robustness_perturb.py).
- Performance drop, confidence drop, label-flip rate.

## 10. Limitations
- Format/resolution confound: fakes are PNG/512, reals mix JPEG + varied sizes. Controlled by
  uniform PNG + JPEG augmentation + aspect-preserving resize; still a boundary of the work.
- Aspect-ratio distortion (supervisor-flagged): naive squashing distorts non-square reals only;
  we mitigate with the "aspect" variant and report the scaled-vs-aspect delta.
- Confound exploitation not fully quantified: we control for it, but the metadata-only /
  scaled-vs-aspect ablations that would MEASURE how much the model used it are pending.
- London-DB resolution confound (tested, not just noted).
- Closed-set classifiers cannot reject unknown generators (forced labels). QUANTIFIED: ~98% of
  unseen-GAN images are confidently assigned a REAL class (false-known rate 0.96 @0.5); an
  entropy-based rejection is only a partial mitigation (44% still confident @0.9).
- GAN-Fingerprints attribution is out of scope for the current report (parked on the
  `ganfp-integrated` branch); DE-FAKE multi-class attribution is the method of record.
- Small per-generator training set for fine-tuning.
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
