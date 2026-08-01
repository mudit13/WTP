# Authoritative Results and Discussion draft

Evidence source: run `2026-08-01_eightway_v1`, core commit `141da128`, aspect-preserving
256-pixel variant, training-only JPEG q30–100 augmentation.

Population: 2,154 images (1,132 fake, 1,022 real), eight trained fake-generator classes,
173 known-fake test images, and 300 OpenForensics-fake crops used only as an external challenge.

This is report-ready technical content, not a final formatted chapter. Citations should be
inserted using `CITATIONS.md`.

## 1. Binary detection

The pretrained binary DE-FAKE checkpoint was evaluated as an external baseline, whereas the
DCT linear SVM was fitted in-domain. Consequently, their headline numbers describe different
training regimes and should not be interpreted as a pure architectural comparison.

On the 435 images shared by both methods, DCT achieved balanced accuracy 0.608 and AUROC 0.661,
compared with 0.519 and 0.620 for pretrained DE-FAKE. The paired balanced-accuracy difference
was +0.090 in favour of DCT (95% bootstrap interval +0.030 to +0.147, p=0.003). The AUROC
difference was not significant (DCT advantage 0.042; interval −0.027 to +0.110, p=0.244), and
McNemar’s test did not reach 0.05 (p=0.078). Thus DCT performed better at the evaluated operating
point, but the evidence does not establish a significant ranking advantage.

The external OpenForensics-fake challenge revealed a stronger limitation. With all 300
OpenForensics-fake crops excluded from training, DCT obtained balanced accuracy 0.416,
AUROC 0.418 and Cohen’s κ −0.147 on n=504. Fake recall was 0.17. This is a systematic
out-of-domain failure rather than an uninformative chance result: known synthetic-source
frequency cues do not transfer to this manipulated in-the-wild population and are partly ranked
in the wrong direction.

## 2. Primary eight-way attribution

The fine-tuned DE-FAKE head classified the eight known fake sources with fixed-seed top-1
accuracy 0.873, balanced accuracy 0.864, macro-F1 0.866 and κ 0.854 on n=173. Bootstrap
intervals were 0.827–0.919 for top-1, 0.815–0.913 for balanced accuracy and 0.802–0.907 for κ.

Per-class recall showed a structured pattern:

| Class | Recall | Main error |
|---|---:|---|
| FLUX.1-schnell | 1.00 | — |
| SD1.5 img2img | 1.00 | — |
| StarGAN | 1.00 | — |
| SD1.5 txt2img | 0.955 | FaceApp |
| StyleGAN3-FFHQ | 0.909 | FaceApp |
| FaceApp | 0.75 | PGGAN-v1, StyleGAN3 |
| PGGAN-v2 | 0.70 | PGGAN-v1 |
| PGGAN-v1 | 0.60 | PGGAN-v2 |

The ten-seed sweep substantially qualifies the fixed-seed headline. Mean top-1 was
0.814 ± 0.039, balanced accuracy 0.809 ± 0.039 and κ 0.788 ± 0.045. Seed 42 produced the
maximum top-1 among the ten seeds. The report should therefore present the pre-specified
fixed-seed result together with seed sensitivity, rather than treating 0.873 as a stable
population estimate.

## 3. Auxiliary merged-Real classification

The optional nine-way head added one source-balanced `real` label. It obtained top-1 0.720,
balanced accuracy 0.772 and κ 0.682 on n=236. Real-class recall was only 0.429: real images were
often assigned to FaceApp (0.190), StyleGAN3-FFHQ (0.190) and SD1.5-img2img (0.095). This
experiment demonstrates the difficulty of joint authenticity/source classification and is kept
as sensitivity analysis rather than replacing the fake-only result.

## 4. Image-only ablation

Removing BLIP caption features reduced fixed-seed top-1 only from 0.873 to 0.861 and κ from
0.854 to 0.841; bootstrap intervals strongly overlapped. Both evaluations used the same
173 test paths.

Across ten seeds, image-only top-1 averaged 0.830 ± 0.038, slightly above the image+caption
mean of 0.814 ± 0.039. BLIP captions are therefore not necessary for the observed attribution
performance. This does not make the representation content-independent: CLIP image embeddings
remain semantic and can encode source-domain appearance.

## 5. End-to-end DCT→DE-FAKE cascade

DCT detected 119 of 173 known fake test images (68.8%). Conditional on detection, attribution
remained strong: top-1 0.866, κ 0.843, with top-1 interval 0.807–0.916. Counting undetected fakes
as end-to-end failures reduced top-1 to 0.595 and κ to 0.557 (top-1 interval 0.532–0.659).
Detection, not attribution, is therefore the dominant pipeline bottleneck.

| Generator | Detection recall | Conditional attribution | End-to-end recall |
|---|---:|---:|---:|
| FLUX.1-schnell | 0.909 | 1.000 | 0.909 |
| StyleGAN3-FFHQ | 0.818 | 1.000 | 0.818 |
| SD1.5 txt2img | 0.818 | 0.944 | 0.773 |
| SD1.5 img2img | 0.741 | 0.900 | 0.667 |
| PGGAN-v1 | 0.800 | 0.750 | 0.600 |
| FaceApp | 0.550 | 0.636 | 0.350 |
| PGGAN-v2 | 0.600 | 0.583 | 0.350 |
| StarGAN | 0.200 | 1.000 | 0.200 |

StarGAN is the clearest detection-limited class: attribution was perfect for the four detected
examples, but DCT missed 16 of 20. FaceApp and PGGAN-v2 were weak at both stages.

Among 197 real test images, 65 were incorrectly passed as fake (33.0%). Their forced attribution
labels were dominated by FaceApp (24), StyleGAN3-FFHQ (19) and PGGAN-v1 (10).

## 6. Leave-one-generator-out generalization

Each LOGO fold omitted one fake generator entirely, trained on the remaining seven and
force-scored all held-out images. Ordinary top-1 is undefined because the true label is absent.

| Held-out generator | Dominant forced label | Share | Mean confidence |
|---|---|---:|---:|
| PGGAN-v1 | PGGAN-v2 | 96% | 0.902 |
| PGGAN-v2 | PGGAN-v1 | 94% | 0.887 |
| StyleGAN3-FFHQ | FaceApp | 92% | 0.782 |
| SD1.5 img2img | SD1.5 | 90% | 0.773 |
| SD1.5 txt2img | SD1.5 img2img | 57% | 0.643 |
| FLUX.1-schnell | SD1.5 | 58% | 0.646 |
| StarGAN | FaceApp | 62% | 0.684 |
| FaceApp | StyleGAN3-FFHQ | 49% | 0.548 |

The reciprocal PGGAN and SD1.5 mappings indicate family/mode overlap. StyleGAN3 and StarGAN
collapse toward FaceApp when their own labels are unavailable. High false-known rates for
PGGAN-v1/v2 show that closed-set confidence is not a reliable unknown-generator detector.

## 7. OpenForensics-fake external OOS

The primary eight-way head force-scored 300 OpenForensics-fake crops. Mean confidence fell from
0.797 in-set to 0.563 OOS, while mean entropy rose from 0.542 to 1.108. Nevertheless, 62.0% of
OOS images exceeded confidence 0.50, 24.3% exceeded 0.70 and 1.3% exceeded 0.90. Forced labels
were dominated by FaceApp (63.3%), followed by StarGAN and StyleGAN3 (approximately 12% each).

The nine-way model reduced false-known rates (39.3% at 0.50; 8.3% at 0.70), but did not solve
unknown-class recognition. OOS top-1 and κ are omitted because OpenForensics-fake is absent from
the output space.

In the cascade subset, DCT detected only 11 of 65 OpenForensics-fake images (recall 0.169).
Detected crops were force-attributed mostly to FaceApp. This combines the detector’s OOS failure
with the attributor’s closed-set limitation.

## 8. FFHQ removal diagnostic

The source-specific attribution head with FFHQ obtained fake-only top-1 0.798. Removing the FFHQ
class increased top-1 to 0.867 (+6.9 points), balanced accuracy by 7.3 points and κ by 7.7 points.
StyleGAN3 recall increased from 0.818 to 0.955, while StyleGAN3→real assignments fell from
18.2% (four images assigned to FFHQ) to zero. Fifteen paired fake test images were correct only
without FFHQ versus three only with it.

This establishes sensitivity to the FFHQ label/training population. It does not prove that
alignment, padding, resampling or another FFHQ preprocessing step caused the original errors.

## 9. Robustness

DCT robustness was transformation-specific rather than generally weak:

| Condition | Balanced accuracy | κ | AUROC |
|---|---:|---:|---:|
| Clean | 0.608 | 0.212 | 0.661 |
| JPEG q30 | 0.584 | 0.163 | 0.615 |
| JPEG q50 | 0.601 | 0.196 | 0.640 |
| JPEG q70 | 0.614 | 0.223 | 0.658 |
| Blur σ1 | 0.517 | 0.031 | 0.626 |
| Blur σ2 | 0.508 | 0.016 | 0.535 |
| Resize ×0.5 | 0.527 | 0.050 | 0.633 |
| Resize ×0.75 | 0.582 | 0.154 | 0.675 |
| Sharpen | 0.609 | 0.222 | 0.662 |

Blur and downscaling remove or alter the high-frequency evidence used by DCT. JPEG and sharpening
preserved ranking better, although individual predictions still changed. Attribution flip rates
of 21–41% were computed over the full 435-row pool (including real and OOS rows) and should be
treated as exploratory stability evidence, not primary eight-way accuracy.

## 10. Split integrity

The corrected audit found zero explicit source-group straddles and zero exact cross-split
duplicates. It reported 427 dHash-near pairs. Most arose from homogeneous frontal img2img,
prompt siblings or aligned-face composition; only one cross-generator pair had Hamming distance
≤3. dHash findings are therefore diagnostic rather than evidence of duplicate leakage.

## 11. Discussion synthesis

The study supports four primary conclusions:

1. **Known-source attribution is feasible in-domain.** The eight-way head performs strongly, but
   fixed-seed performance is optimistic relative to the seed mean.
2. **The sequential system is detection-limited.** High conditional attribution does not
   translate to high end-to-end recall, especially for StarGAN, FaceApp and PGGAN-v2.
3. **Generator families overlap under class omission.** PGGAN siblings, SD1.5 modes and
   StyleGAN3/FaceApp form strong LOGO associations.
4. **Unknown manipulations are not reliably rejected.** DCT performs below chance on the
   OpenForensics challenge and the closed-set head remains confidently wrong on many OOS crops.

The results should be described as source classification under the tested dataset design, not
proof of intrinsic content-independent generator fingerprints. Each class originates from a
different face/content pool, and SD1.5-img2img is especially narrow because it combines
London-only identities with one fixed studio prompt.

## 12. Claims to avoid

- Do not state that DCT has significantly higher AUROC; only the balanced-accuracy difference is
  significant.
- Do not interpret LOGO or OOS top-1/κ as ordinary accuracy.
- Do not use mixed `all_fakes` metrics as headline results.
- Do not claim FFHQ preprocessing caused StyleGAN3 errors.
- Do not present 0.873 without the ten-seed mean and variability.
- Do not present old GAN-fp or pre-fix robustness metrics.

## 13. Figure and table plan

### Main report

1. Updated DCT→DE-FAKE pipeline diagram.
2. Eight-way in-set confusion matrix:
   `attr_eval_8way_aspect/cm_in_set.png`.
3. Detection comparison table with paired significance.
4. Cascade per-generator table.
5. LOGO dominant forced-label table.
6. OpenForensics in-set/OOS confidence histogram:
   `oos_aspect/confidence_eight_way.png`.

### Appendix

- Nine-way confusion matrix.
- FFHQ with/without confusion matrices.
- Full LOGO distributions.
- Full robustness table and all drop JSONs.
- Image-only confusion matrix and seed sweep.
- Split audit and dHash diagnostic.
- SD1.5 img2img manifest/prompt/eye-artifact documentation.
- GAN-fp method history without prototype metrics.
