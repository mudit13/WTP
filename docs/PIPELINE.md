# Professor-aligned server runbook

This is the only active command runbook. Run commands inside the container from
`/pitsec_sose26_topic8`.

## 1. Environment

```bash
cd /pitsec_sose26_topic8
export $(grep -v '^#' configs/paths.env | xargs)
PY=$WTP_PY_DEFAKE
CFG=configs/config.yaml
RUN=2026-08-01_eightway_v1
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
- Matched source-specific attribution with and without FFHQ
- DCT-to-DE-FAKE cascade
- Cohen's kappa in binary, multi-class, and cascade metric outputs
- Mandatory leakage gates, bootstrap CIs, paired significance, and ten-seed sensitivity
- 36 planned steps

## 6. Authoritative experiment

```bash
nohup $PY scripts/run_experiment.py \
  --run_id $RUN \
  --variant aspect --jpeg_aug on \
  --stages index,variants,confound,detect,dct,attribution,ffhq_ablation,cascade,oos,rigor,aggregate \
  > logs/${RUN}.out 2>&1 &
```

Do not reuse another run ID. Use `--resume` only to continue this exact run with an unchanged
config hash.

## 7. Mandatory rigor stage

`rigor` is part of the default orchestrator and runs before final aggregation. It:

- Hard-fails on exact cross-split duplicates or explicit group straddles
- Retains near-dHash matches as a diagnostic rather than a hard failure
- Bootstraps DE-FAKE, DCT, 8-way, 9-way, and cascade metrics
- Runs the paired DCT/DE-FAKE significance test
- Runs the ten-seed primary-attribution sensitivity analysis

For an older run that already completed model stages, execute rigor without retraining:

```bash
$PY scripts/run_experiment.py \
  --run_id $RUN --resume \
  --stages rigor,aggregate
```

Required gates:

- No missing attribution class
- No OOS/training overlap
- No explicit group straddling
- No exact cross-split duplicate
- Every class has train/validation/test support
- Bootstrap kappa intervals accompany DCT, DE-FAKE, attribution, and cascade metrics
- Seed-sweep uncertainty accompanies primary attribution metrics

## 8. Optional appendix stages

Robustness is not required for the core professor-facing result:

```bash
nohup $PY scripts/run_experiment.py \
  --run_id $RUN --resume \
  --stages robustness,aggregate \
  > logs/${RUN}_appendix.out 2>&1 &
```

## 9. Evidence locations

```text
results/<run_id>/run_manifest.json
results/<run_id>/REPORT_SUMMARY.md
results/<run_id>/dct_svm_aspect/
results/<run_id>/finetune_8way_aspect_jpegaug/
results/<run_id>/finetune_9way_aspect_jpegaug/
results/<run_id>/ffhq_ablation/
results/<run_id>/logo_8way_aspect_jpegaug/
results/<run_id>/cascade/
results/<run_id>/oos_aspect/
results/<run_id>/ci_attr_8way.json
results/<run_id>/ci_defake_detection.json
results/<run_id>/ci_dct_detection.json
results/<run_id>/ci_cascade_conditional.json
results/<run_id>/ci_cascade_end_to_end.json
results/<run_id>/seed_sweep_8way.json
results/<run_id>/defake_vs_dct_significance.json
results/<run_id>/leakage_audit_8way.json
```

Never copy metrics from the superseded 7-class study into the final report.

## 10. Handover verification

Read-only; run after `rigor`/`aggregate` complete and before handing the run to a supervisor:

```bash
$PY scripts/verify_handover.py --results_dir results/$RUN/ --run_id $RUN
```

Fails loudly on missing artifacts, non-zero leakage-audit gates, headline numbers that do not
match the known authoritative values, or a stale flat `REPORT_SUMMARY.md` sitting next to the
run directory. See `HANDOVER.md` for the full recipient checklist.
