# WTP Topic 8: AI Image Detection and Attribution

This project is a research pipeline for detecting AI-generated images and attributing generated images to their source model. It combines two stages:

1. **Detection:** classify an image as real or generated using DCT features and a linear SVM, with pretrained DE-FAKE as a comparison baseline.
2. **Attribution:** classify detected generated images among the configured generator classes using a fine-tuned DE-FAKE feature head.

The repository contains code, configuration, tests, documentation, and report material. Datasets, model checkpoints, generated images, virtual environments, and large run artifacts are external assets and are not committed.

## Start here

Use this reading order:

1. [`docs/RUNBOOK.md`](docs/RUNBOOK.md) for machine setup, environments, the final-run sequence, and verification.
2. [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for the scientific design and safeguards.
3. [`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md) for dataset origins, licensing, transformations, and limitations.
4. [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for the experiment-to-evidence map.

Historical notes are under `docs/history/` and are not operational instructions.

## Repository layout

```text
configs/                 Scientific configuration and local path template
scripts/                 Pipeline entry points and shared implementation
scripts/lib/             Shared schemas, metrics, features, image operations, and I/O
De-Fake-patched/         Patched DE-FAKE and BLIP code used by this project
requirements/            Shared dependencies and captured server environment locks
docs/                    Setup, runbook, methodology, provenance, and experiment map
templates/               Generated-document templates
releases/                Release-manifest template and final handover records
tests/                   CPU-safe regression tests
report/                  Report drafting material
results/ and logs/        Generated outputs, ignored except for directory placeholders
```

`De-Fake-patched/` intentionally remains at its current path because active imports and server workflows depend on it.

## Quick setup

From the repository root:

```bash
cp configs/paths.example.env configs/paths.env
# Edit configs/paths.env for the server or workstation.
set -a
. configs/paths.env
set +a

python3 scripts/check_docs.py
python3 -m compileall -q scripts tests
python3 -m pytest -q
```

Use the configured interpreter for GPU or DE-FAKE work rather than bare `python`:

```bash
"$WTP_PY_DEFAKE" scripts/run_experiment.py \
  --dry_run \
  --run_id "$(date +%Y%m%d)"
```

See [`docs/RUNBOOK.md`](docs/RUNBOOK.md) for the full setup and run sequence.

## Main operational entry points

| Task | Command |
|---|---|
| Build or refresh the canonical metadata index | `scripts/build_master_index.py` |
| Review the complete experiment plan without running it | `scripts/run_experiment.py --dry_run` |
| Run the pipeline | `scripts/run_experiment.py` |
| Verify final handover evidence | `scripts/verify_handover.py --release <release.yaml>` |
| Capture exact server environments | `scripts/capture_env_locks.sh` |
| Validate internal documentation links | `scripts/check_docs.py` |

The complete command index is in [`docs/CLI_REFERENCE.md`](docs/CLI_REFERENCE.md).

## Configuration model

The project deliberately uses two configuration layers:

- `configs/config.yaml` contains scientific settings such as datasets, labels, preprocessing, split rules, augmentation, taxonomy, seeds, and evaluation settings.
- `configs/paths.env` contains machine-specific paths and interpreter locations. It is created from `configs/paths.example.env` and is ignored by Git.

Do not move scientific choices into environment variables. Do not put credentials, tokens, usernames, or host-specific access commands into committed files.

## Reproducibility

A final result is identified by all of the following together:

- Git commit
- `configs/config.yaml` hash
- run ID
- data and checkpoint manifests
- exact environment locks
- release manifest under `releases/<release-id>/release.yaml`

The fixed default seed makes repeated runs substantially more stable, but GPU training can still show small numerical or initialization variation. The project therefore includes grouped bootstrap confidence intervals and a multi-seed sensitivity analysis. Treat the release manifest and its verified run directory as the authoritative evidence, not a number copied into prose.

## Supervisor handover

The handover consists of:

- the Git repository at a reviewed commit or annotated tag
- one completed release manifest at `releases/<release-id>/release.yaml`
- a lightweight evidence archive (metric JSONs, logs, manifests — no licensed images or model weights)
- server access details supplied privately through the institution's approved channel

The handover is ready when `scripts/verify_handover.py --release ... --verify_mounts` exits without errors and the evidence archive has been inspected. See [`docs/RUNBOOK.md`](docs/RUNBOOK.md) for the complete sequence.

Authoritative claims come from the completed release manifest and its verified run directory. Report drafts do not override generated artifacts.

## Contributing

Use `snake_case` for functions and variables, `PascalCase` for classes. Generator and dataset names must match `configs/config.yaml` exactly — no second spellings.

Before committing:

```bash
python3 scripts/check_docs.py
python3 -m compileall -q scripts tests
python3 -m pytest -q
```

Executable changes that affect a final result require a new server run and release record.
