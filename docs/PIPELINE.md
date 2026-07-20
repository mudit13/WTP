# Professor-aligned server runbook

This is the only active command runbook. Run commands inside the container from
`/pitsec_sose26_topic8`.

## 1. Environment

```bash
cd /pitsec_sose26_topic8
export $(grep -v '^#' configs/paths.env | xargs)
PY=$WTP_PY_DEFAKE
CFG=configs/config.yaml
RUN=2026-07-20_eightway_v1
```

Never use bare `python`. Do not start an authoritative run from uncommitted code.

## 2. Preflight

Confirm the London source count, OpenForensics sidecar, and cached SD1.5 revision:

```bash
$PY - <<'PY'
from pathlib import Path

root = Path("/pitsec_sose26_topic8")
london = root / "dataset/londondb/neutral_front/neutral_front"
images = [p for p in london.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
print("London images:", len(images))

of_map = root / "dataset/openforensics/openforensics_groups.csv"
print("OpenForensics group map:", of_map, "exists=", of_map.exists())

ref = root / "models/models--runwayml--stable-diffusion-v1-5/refs/main"
print("Cached SD1.5 revision:", ref.read_text().strip() if ref.exists() else "<not found>")
PY
```

Stop if London images are missing, the OpenForensics group map is absent, or no exact SD1.5
revision can be identified.

## 3. SD1.5 img2img pilot

The primary strength is pre-registered as 0.6. The pilot is a go/no-go feasibility check, not
an accuracy-driven hyperparameter search. It uses train-hashed identities only and writes to
directories that the master index never scans.

```bash
export SD15_REV=<EXACT_CACHED_REVISION>

nohup sh -c '
  for strength in 0.4 0.6 0.8; do
    $WTP_PY_DEFAKE scripts/generate_sd15_img2img.py \
      --purpose pilot --identity_partition train \
      --max_sources 12 --num_images 12 \
      --strength $strength --revision $SD15_REV \
      --output_root $WTP_ROOT/dataset/sd15_img2img_pilot/strength_${strength}
  done
' > logs/img2img_pilot.out 2>&1 &
```

Review the 36 pilot images using a predeclared feasibility criterion: recognizable frontal face,
no gross generation failure, and meaningful but identity-preserving transformation. Keep 0.6
unless it fails that criterion. Record any deviation and its rationale before authoritative
generation.

## 4. Authoritative img2img generation and grouping

Use a clean canonical output directory. The script refuses an unpinned model revision, a
non-`all` identity partition, or a manifest mismatch.

```bash
nohup $PY scripts/generate_sd15_img2img.py \
  --purpose authoritative --identity_partition all \
  --strength 0.6 --revision $SD15_REV \
  > logs/img2img_authoritative.out 2>&1 &
```

After `img2img_authoritative.out` reports 108 total outputs:

```bash
$PY scripts/make_img2img_group_map.py
```

Validate generation before training:

```bash
$PY - <<'PY'
import json
import pandas as pd
from pathlib import Path

root = Path("/pitsec_sose26_topic8/dataset/sd15_img2img")
meta = pd.read_csv(root / "metadata.csv")
groups = pd.read_csv(root / "londondb_img2img_groups.csv")
manifest = json.loads((root / "generation_manifest.json").read_text())

assert len(meta) == 108, len(meta)
assert meta["output_path"].nunique() == 108
assert manifest["purpose"] == "authoritative"
assert manifest["identity_partition"] == "all"
assert float(manifest["strength"]) == 0.6
assert manifest["revision"] not in {"main", "<default-repository-revision>"}
assert set(meta["output_path"]).issubset(set(groups["full_path"]))
assert set(meta["source_image"]).issubset(set(groups["full_path"]))
print("img2img validation passed:", len(meta), "outputs,", groups["source_image_id"].nunique(), "identities")
PY
```

## 5. Inspect the experiment plan

```bash
$PY scripts/run_experiment.py --dry_run --run_id $RUN
```

The plan must print:

- Eight fake classes
- Auxiliary joint model with merged Real
- OpenForensics-fake excluded from DCT training
- Group-aware OpenForensics challenge
- Eight-fold LOGO
- DCT-to-DE-FAKE cascade
- 22 planned steps

## 6. Authoritative experiment

```bash
nohup $PY scripts/run_experiment.py \
  --run_id $RUN \
  --variant aspect --jpeg_aug on \
  --stages index,variants,confound,detect,dct,attribution,cascade,oos,aggregate \
  > logs/${RUN}.out 2>&1 &
```

Do not reuse another run ID. Use `--resume` only to continue this exact run with an unchanged
config hash.

## 7. Required post-run rigor

Run these only after the authoritative pipeline succeeds:

```bash
nohup sh -c '
  $WTP_PY_DEFAKE scripts/audit_split_leakage.py \
    --config configs/config.yaml \
    --index results/'"$RUN"'/index_aspect.csv \
    --class_mode fake_only \
    --out results/'"$RUN"'/leakage_audit_8way.json &&
  $WTP_PY_DEFAKE scripts/bootstrap_metrics.py \
    --predictions results/'"$RUN"'/attr_eval_8way_aspect/attribution_per_image.csv \
    --subset in_set \
    --out results/'"$RUN"'/ci_attr_8way.json &&
  $WTP_PY_DEFAKE scripts/seed_sweep.py \
    --config configs/config.yaml \
    --index results/'"$RUN"'/index_aspect.csv \
    --class_mode fake_only --jpeg_aug on --n_seeds 10 \
    --features_cache results/'"$RUN"'/clip_feats_aspect_clean.npz \
    --captions_csv $WTP_ROOT/dataset/defake_predictions_'"$RUN"'_aspect.csv \
    --out results/'"$RUN"'/seed_sweep_8way.json
' > logs/${RUN}_rigor.out 2>&1 &
```

Required gates:

- No missing attribution class
- No OOS/training overlap
- No explicit group straddling
- No exact cross-split duplicate
- Every class has train/validation/test support
- Bootstrap and seed-sweep uncertainty accompany headline metrics

## 8. Optional appendix stages

GAN-fp and robustness are not required for the core professor-facing result:

```bash
nohup $PY scripts/run_experiment.py \
  --run_id $RUN --resume \
  --stages robustness,ganfp,aggregate \
  > logs/${RUN}_appendix.out 2>&1 &
```

## 9. Evidence locations

```text
results/<run_id>/run_manifest.json
results/<run_id>/REPORT_SUMMARY.md
results/<run_id>/dct_svm_aspect/
results/<run_id>/finetune_8way_aspect_jpegaug/
results/<run_id>/finetune_9way_aspect_jpegaug/
results/<run_id>/logo_8way_aspect_jpegaug/
results/<run_id>/cascade/
results/<run_id>/oos_aspect/
results/<run_id>/ci_attr_8way.json
results/<run_id>/seed_sweep_8way.json
```

Never copy metrics from the superseded 7-class study into the final report.
