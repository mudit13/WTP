# Handover

Single entrypoint for a new recipient (supervisor, examiner, or teammate) of this repository.
Read this file first; it links to everything else you need.

This project targets the **existing** Magdeburg institutional GPU container. There is no
Dockerfile here on purpose: the container, its CUDA stack, and the licensed datasets/weights are
already provisioned server-side and must stay mounted external assets, not baked into an image.

## 1. Container entry and mount mapping

```bash
# Connect EduVPN first, then:
ssh pitsec_sose26_topic8@gensynth.cs.uni-magdeburg.de
sudo pitsec_sose26_topic8.docker PITSEC26   # enter the container
```

| Host | Container |
|---|---|
| `/vol2/pitsec_sose26_topic8/sharedDockerDir/` | `/pitsec_sose26_topic8/` |

The repo root **is** `/pitsec_sose26_topic8` (= `sharedDockerDir`): tracked code
(`scripts/`, `configs/`, `docs/`, `De-Fake-patched/`, ...) lives there alongside large data,
model weights, and venvs, which are git-ignored and stay in place across `git pull`. Full detail:
`docs/SERVER_WORKFLOW.md`.

## 2. Required external mounts

Nothing here is redistributed in this git repository; these must already be mounted / reachable
from inside the container for a run to work:

| Path (container) | Contents | Source |
|---|---|---|
| `$WTP_DFFD_DIR` (`/share/DeepFake/DFFD_Images`) | FFHQ/PGGAN-v1/PGGAN-v2/StarGAN/FaceApp subsets | DFFD dataset (licensed; supervisor-provided share) |
| `$WTP_ROOT/dataset/openforensics/{real,fake}` | OpenForensics face crops + `openforensics_groups.csv` | OpenForensics dataset (see `docs/DATA_PROVENANCE.md`) |
| `$WTP_ROOT/dataset/londondb/neutral_front/neutral_front` | Face Research Lab London Set | Face Research Lab (licensed academic use) |
| `$WTP_ROOT/models/clip_linear.pt`, `finetune_clip.pt` | Pretrained DE-FAKE binary checkpoints | Supervisor-provided |

See `configs/paths.example.env` for every path variable and `docs/DATA_PROVENANCE.md` for full
per-dataset provenance (licenses, generation parameters, known unknowns).

## 3. Environment activation

```bash
cd /pitsec_sose26_topic8
cp configs/paths.example.env configs/paths.env   # first time only; defaults already match
export $(grep -v '^#' configs/paths.env | xargs)
```

Three generation venvs already exist on the server, one per generator; DE-FAKE inference and
every script in this repo run inside `venv_sd15` (`$WTP_PY_DEFAKE`). Never use bare `python`.
Full detail and exact recreate-from-scratch commands: `docs/ENVIRONMENTS.md`. Committed
`requirements-sd15.lock` / `requirements-flux1.lock` / `requirements-stylegan3.lock` pin the
exact package versions each venv should have (generate/refresh with
`bash scripts/capture_env_locks.sh` if a venv drifts).

## 4. Checkpoint placement

```
$WTP_ROOT/models/clip_linear.pt     # pretrained DE-FAKE binary head (real/fake only)
$WTP_ROOT/models/finetune_clip.pt   # pretrained DE-FAKE CLIP backbone
```

Both are supervisor-provided and already present on the server; this repo does not ship or
redistribute them. The fine-tuned multi-class attribution head(s) this project trains
(`defake_head.pt` per run) are written under `results/<run_id>/finetune_*_*/` and are evidence,
not checkpoints to hand out separately.

## 5. Authoritative run and evidence location

- Run ID: `2026-08-01_eightway_v1`
- Evidence root (server, authoritative): `results/2026-08-01_eightway_v1/`
- Same run, extracted locally from the evidence archive (if you received one instead of server
  access): `results/2026-08-01_eightway_v1_evidence/results/2026-08-01_eightway_v1/`
- What it is, headline numbers, and interpretation boundaries: `docs/EXPERIMENT_CATALOG.md`,
  `report/AUTHORITATIVE_RESULTS_DRAFT.md`
- Full document map and authority order when documents disagree: `docs/README.md`

## 6. Verification commands

Run these, in order, before trusting or re-citing any number from the run above. Steps 1-2 work
anywhere (no GPU / dataset access needed); steps 3-6 must run inside the container, against the
real mounts, on `venv_sd15` (`$WTP_PY_DEFAKE`).

```bash
# 1. CPU-only regression tests + Python 3.9 syntax check (no GPU / dataset access needed)
python -m pytest -q
python -m compileall -q scripts tests

# 2. Confirm the orchestrator's default plan matches docs/PIPELINE.md
$WTP_PY_DEFAKE scripts/run_experiment.py --dry_run --run_id 2026-08-01_eightway_v1

# 3. Import/install smoke check for each generation venv against its committed lock file
#    (repeat per venv: venv_sd15 / venv_flux1 / venv_stylegan3)
diff <(venv_sd15/bin/pip freeze) requirements-sd15.lock   # expect no unreviewed drift

# 4. DE-FAKE inference smoke test, specifically to confirm the BLIP med_config portability
#    fix (De-Fake-patched/blipmodels/blip.py) still resolves correctly from this container
$WTP_PY_DEFAKE scripts/run_defake_batch.py --test

# 5. Regenerate data/checkpoint manifests against the exact mounted files, then re-verify them
#    byte-for-byte (--verify_mounts makes an unreachable/missing mount a hard failure here,
#    since we're running ON the server where the mount must be present)
$WTP_PY_DEFAKE scripts/create_data_manifest.py --mode data --config configs/config.yaml \
    --index results/2026-08-01_eightway_v1/index_aspect.csv \
    --out results/2026-08-01_eightway_v1/data_manifest.csv
$WTP_PY_DEFAKE scripts/create_data_manifest.py --mode checkpoints \
    --out results/2026-08-01_eightway_v1/checkpoint_manifest.csv

# 6. Read-only completeness + headline-number + leakage-gate + manifest check for the
#    authoritative run
$WTP_PY_DEFAKE scripts/verify_handover.py \
    --results_dir results/2026-08-01_eightway_v1/ --run_id 2026-08-01_eightway_v1 \
    --require_data_manifest --require_checkpoint_manifest --verify_mounts
```

`verify_handover.py` fails loudly (non-zero exit) on missing artifacts, a non-zero leakage-audit
gate, a headline number outside tolerance of the known authoritative value, a stale flat
`REPORT_SUMMARY.md` sitting next to the run directory, or (with the flags above) a manifested
dataset/checkpoint file that is missing or does not match its recorded SHA-256.

## 7. Packaging for handover

Only create the tag below once the working tree is clean (`git status` empty) and every command
in section 6 has passed. Two artifacts leave the Magdeburg environment; the dataset/model mounts
never do:

```bash
# 1. Tag the source at the exact commit the authoritative run and its manifests were verified
#    against
git tag -a handover-2026-08-01_eightway_v1 -m "Handover: repo state verified against 2026-08-01_eightway_v1"
git push origin handover-2026-08-01_eightway_v1   # only if/when told to push

# 2. Build a lightweight evidence archive: everything verify_handover.py checked, nothing
#    that duplicates licensed images or feature caches (no .npz/.pt/.joblib)
tar -czf 2026-08-01_eightway_v1_evidence.tar.gz \
    --exclude='*.npz' --exclude='*.pt' --exclude='*.joblib' \
    results/2026-08-01_eightway_v1/
```

Hand over: (a) the git repository at the `handover-2026-08-01_eightway_v1` tag, and (b) the
`2026-08-01_eightway_v1_evidence.tar.gz` archive. The supervisor's own server access covers the
mounted datasets/checkpoints; re-verify the archive's manifests against that mount with
`verify_handover.py --verify_mounts` (section 6, step 6) rather than re-sending the data.

## 8. What must not be redistributed

- DFFD, OpenForensics, and Face Research Lab London Set images/crops (licensed academic data).
- `clip_linear.pt` / `finetune_clip.pt` (supervisor-provided checkpoints).
- Any file under `results/<run_id>/` that contains raw image paths pointing at the above - the
  evidence archive shared for reporting is JSON/CSV/PNG/manifests only (no `.npz`/`.pt`/`.joblib`
  feature caches or model weights).

Only the git repository (code, configs, docs) and the lightweight report evidence archive
(metrics/CIs/manifests, no NPZ/PT/joblib) are meant to leave the Magdeburg environment.
