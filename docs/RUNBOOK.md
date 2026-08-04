# Setup and run guide

From fresh checkout to verified supervisor handover. Complete the setup sections once per server, then follow the run sequence for each experiment.

## 1. Access and security

Server access details are not committed. Obtain through the institution's approved private channel:

- VPN or network-access instructions
- SSH username and hostname
- container-entry command, if required
- dataset and checkpoint permissions

Never commit credentials, tokens, private hostnames, container identifiers, or shell history containing secrets.

## 2. Filesystem layout

```text
<WTP_ROOT>/
├── configs/
├── scripts/
├── De-Fake-patched/
├── docs/
├── dataset/              # external or generated, not committed
├── models/               # external checkpoints and caches, not committed
├── results/              # generated runs, not committed
├── logs/                 # generated logs, not committed
├── venv_sd15/            # server environment, not committed
├── venv_flux1/           # server environment, not committed
├── venv_stylegan3/       # server environment, not committed
└── stylegan3/            # external source checkout, not committed
```

`De-Fake-patched/` stays at the repository root. Do not relocate it.

## 3. Environment variables

Create the local environment file:

```bash
cp configs/paths.example.env configs/paths.env
```

Edit every required value in `configs/paths.env`, then load it:

```bash
set -a
. configs/paths.env
set +a
```

Do not use `export $(grep ... | xargs)`. It breaks values with spaces or shell metacharacters.

Confirm the main values:

```bash
printf 'WTP_ROOT=%s\n' "$WTP_ROOT"
printf 'WTP_PY_DEFAKE=%s\n' "$WTP_PY_DEFAKE"
printf 'WTP_PY_FLUX1=%s\n' "$WTP_PY_FLUX1"
printf 'WTP_PY_STYLEGAN3=%s\n' "$WTP_PY_STYLEGAN3"
```

### Required variables

| Variable | Role |
|---|---|
| `WTP_ROOT` | Absolute repository and project root. |
| `WTP_DFFD_DIR` | Absolute DFFD image root. |
| `WTP_PY_DEFAKE` | Python interpreter for analysis, DCT, DE-FAKE, attribution, and SD1.5. |
| `WTP_PY_SD15` | Compatibility alias for SD1.5 generation. Normally identical to `WTP_PY_DEFAKE`. |
| `WTP_PY_FLUX1` | Interpreter for FLUX.1 generation. |
| `WTP_PY_STYLEGAN3` | Interpreter for StyleGAN3 generation. |
| `WTP_DEFAKE_CLIP_LINEAR` | Pretrained binary DE-FAKE head checkpoint. |
| `WTP_DEFAKE_FINETUNE_CLIP` | Pretrained DE-FAKE CLIP backbone checkpoint. |
| `WTP_STYLEGAN3_REPO` | External StyleGAN3 checkout. |

### Optional variables

| Variable | Role |
|---|---|
| `WTP_BLIP_URL` | Override the BLIP checkpoint URL. |
| `WTP_MODEL_CACHE` | Override the shared generator model cache. |
| `WTP_SD15_OUTPUT_DIR` | Override SD1.5 text-to-image output. |
| `WTP_FLUX1_OUTPUT_DIR` | Override FLUX.1 output. |
| `WTP_STYLEGAN3_OUTPUT_DIR` | Override StyleGAN3 output. |

## 4. Environments and dependencies

Three separate environments cover the GPU-generation stacks, which have conflicting dependencies.

| Environment | Interpreter variable | Used for |
|---|---|---|
| SD1.5 and DE-FAKE | `WTP_PY_DEFAKE`, alias `WTP_PY_SD15` | analysis, indexing, DCT, attribution, SD1.5 generation |
| FLUX.1 | `WTP_PY_FLUX1` | FLUX.1-schnell generation only |
| StyleGAN3 | `WTP_PY_STYLEGAN3` | StyleGAN3 generation only |

Always use the explicit interpreter variable. Do not rely on whichever `python` is active.

Dependency files:

```text
requirements/base.txt        Shared analysis dependencies
requirements/defake.txt      DE-FAKE support dependencies
requirements/dev.txt         Test and development dependencies
requirements/locks/*.txt     Exact server package snapshots
```

The shared files document direct dependencies for CPU checks. They are not a complete CUDA recreation recipe. Capture exact server state after the final run:

```bash
bash scripts/capture_env_locks.sh
```

## 5. Required external assets

Confirm before running:

- DFFD data is readable at `WTP_DFFD_DIR`
- OpenForensics real and fake crops are present
- London Set source images for SD1.5 img2img are present
- `WTP_DEFAKE_CLIP_LINEAR` and `WTP_DEFAKE_FINETUNE_CLIP` exist and are readable
- BLIP can resolve or download its checkpoint
- External StyleGAN3 checkout and weight file are present when regeneration is required

See [`DATA_PROVENANCE.md`](DATA_PROVENANCE.md) for provenance and redistribution constraints.

## 6. Common setup failures

### A path appears literally as `${WTP_ROOT}`

`configs/paths.env` needs concrete absolute paths. The loader does not do nested shell expansion inside file values.

### A script imports the wrong DE-FAKE package

Run from the repository root with `WTP_ROOT` loaded. Confirm `De-Fake-patched/` is at that root.

### A generator works but analysis does not

Check that the command used `WTP_PY_DEFAKE`, not the generator-specific interpreter.

### Lock files differ

Do not overwrite automatically. Determine whether the server environment changed intentionally before committing a lock-file update.

### A dataset is missing or unreadable

Resolve the mount or permission problem, then rerun preflight. Do not substitute a different dataset path.

---

## Run sequence

Complete setup above once, then follow this sequence for each experiment run.

## 7. Choose immutable identifiers

Use a new run ID. Do not reuse an existing results directory.

```bash
export RUN_ID="$(date +%Y%m%d)"
export RELEASE_ID="$RUN_ID"
```

Confirm the branch, commit, and clean working tree:

```bash
git branch --show-current
git rev-parse HEAD
git status --short
```

Stop if `git status --short` prints tracked changes.

## 8. Load environment and preflight

```bash
cd <repository-root>
set -a
. configs/paths.env
set +a

printf 'root: %s\n' "$WTP_ROOT"
sha256sum configs/config.yaml
```

Run repository checks:

```bash
python3 scripts/check_docs.py
python3 -m compileall -q scripts tests
python3 -m pytest -q
```

Verify interpreters and assets:

```bash
for py in "$WTP_PY_DEFAKE" "$WTP_PY_FLUX1" "$WTP_PY_STYLEGAN3"; do
  test -x "$py" || { echo "Missing interpreter: $py" >&2; exit 1; }
done

test -d "$WTP_ROOT/De-Fake-patched" || exit 1
test -d "$WTP_DFFD_DIR" || exit 1
test -f "$WTP_DEFAKE_CLIP_LINEAR" || exit 1
test -f "$WTP_DEFAKE_FINETUNE_CLIP" || exit 1
```

Run the DE-FAKE smoke test and review the planned pipeline:

```bash
"$WTP_PY_DEFAKE" scripts/run_defake_batch.py --test
"$WTP_PY_DEFAKE" scripts/run_experiment.py \
  --dry_run \
  --run_id "$RUN_ID"
```

Verify the run directory does not already exist:

```bash
test ! -e "results/$RUN_ID" || {
  echo "Run directory already exists: results/$RUN_ID" >&2
  exit 1
}
```

Resolve every failed check, broken link, or missing asset before continuing.

## 9. Regenerate synthetic data only when required

Skip when reusing previously verified generated datasets. Regeneration changes the data and requires new manifests and a new run ID.

### SD1.5 text-to-image

```bash
"$WTP_PY_SD15" scripts/generate_sd15_txt2img.py
```

### FLUX.1-schnell

```bash
"$WTP_PY_FLUX1" scripts/generate_flux1_txt2img.py
```

### StyleGAN3

```bash
"$WTP_PY_STYLEGAN3" scripts/generate_stylegan3.py
```

### SD1.5 image-to-image

Review the full generation command before running:

```bash
"$WTP_PY_SD15" scripts/generate_sd15_img2img.py --help
```

After generation, rebuild the group map:

```bash
"$WTP_PY_DEFAKE" scripts/make_img2img_group_map.py --help
```

Do not proceed with missing group relationships.

## 10. Run the experiment

Use the orchestrator. It creates an immutable run directory and records the configuration.

```bash
"$WTP_PY_DEFAKE" scripts/run_experiment.py \
  --run_id "$RUN_ID"
```

The default run covers the core pipeline and mandatory rigor checks. Use `--stages` only when running a documented subset or resuming a failed run. Do not manually copy outputs from an older run.

## 11. Optional report-scope stages

Examples: perturbation robustness, image-only attribution ablation, additional seed sweeps. Check supported stages with:

```bash
"$WTP_PY_DEFAKE" scripts/run_experiment.py --help
```

Every supplemental artifact must go under `results/$RUN_ID/`.

## 12. Manifests and environment locks

Generate data and checkpoint manifests from the final run's canonical index:

```bash
"$WTP_PY_DEFAKE" scripts/create_data_manifest.py \
  --mode data \
  --config configs/config.yaml \
  --index "results/$RUN_ID/index_aspect.csv" \
  --out "results/$RUN_ID/data_manifest.csv"

"$WTP_PY_DEFAKE" scripts/create_data_manifest.py \
  --mode checkpoints \
  --out "results/$RUN_ID/checkpoint_manifest.csv"
```

Capture exact package snapshots:

```bash
bash scripts/capture_env_locks.sh
git diff -- requirements/locks/
```

Unexpected drift must be explained before sign-off. Commit approved lock changes with or just before the release manifest.

## 13. Create the release manifest

```bash
mkdir -p "releases/$RELEASE_ID"
cp releases/release.template.yaml "releases/$RELEASE_ID/release.yaml"
```

Fill from generated artifacts — not from memory or report drafts. The manifest must name the run directory, git commit, required artifacts, headline values, and manifest verification requirements. Do not leave placeholder values.

## 14. Verify the release

```bash
"$WTP_PY_DEFAKE" scripts/verify_handover.py \
  --release "releases/$RELEASE_ID/release.yaml" \
  --verify_mounts
```

A non-zero exit is a release blocker. Rerun repository checks after creating the release file:

```bash
python3 scripts/check_docs.py
python3 -m compileall -q scripts tests
python3 -m pytest -q
```

## 15. Report material

Update report drafts from verified run artifacts. Do not type values from memory. Every reported value must have a named artifact path, metric definition, class scope, and uncertainty estimate. Use [`EXPERIMENTS.md`](EXPERIMENTS.md) as the map between report questions and evidence.

## 16. Commit, tag, and package

Commit the release record and approved lock files:

```bash
git add releases/ requirements/locks/ report/ docs/
git commit -m "Finalize verified supervisor handover $RELEASE_ID"
```

If this commit changes only documentation, locks, and release records, rerun the verifier. If it changes executable code, configuration, datasets, or checkpoints, the previous run no longer corresponds to the commit and must be repeated.

Create the evidence archive, excluding licensed images and model weights:

```bash
tar -czf "${RELEASE_ID}_evidence.tar.gz" \
  --exclude='*.npz' \
  --exclude='*.pt' \
  --exclude='*.pth' \
  --exclude='*.joblib' \
  --exclude='*.png' \
  --exclude='*.jpg' \
  --exclude='*.jpeg' \
  "results/$RUN_ID"
```

Review the archive before transfer:

```bash
tar -tzf "${RELEASE_ID}_evidence.tar.gz" | less
```

Tag only after the verifier passes and the working tree is clean:

```bash
git tag -a "handover-$RELEASE_ID" -m "Verified supervisor handover: $RELEASE_ID"
git push origin HEAD
git push origin "handover-$RELEASE_ID"
```

## 17. Resume and failure rules

- Never delete or overwrite evidence from a partially completed run to reuse its ID.
- Resume only through orchestrator-supported stage selection; document the resumed stages.
- A code, config, taxonomy, split, data, or checkpoint change requires a new run ID.
- A documentation-only correction may retain the run, but the release manifest still identifies the code commit that produced the evidence.
- Preserve failed-run logs until the final run is accepted.
