# Documentation map

Use these documents according to their role. Only `PIPELINE.md` is an executable runbook.

## Handover entry point

- `../HANDOVER.md` - single entrypoint for a new recipient: environment, mounts, checkpoints,
  the authoritative run location, and verification commands.

## Immutable evidence and authoritative science docs

These take priority over any narrative document below when reporting a number:

- `results/<run_id>/run_manifest.json` and the rest of `results/<run_id>/` - the immutable
  evidence for one run. The authoritative run is `2026-08-01_eightway_v1`
  (see `docs/EXPERIMENT_CATALOG.md`). Verify with `scripts/verify_handover.py` before citing.
- `docs/DATA_PROVENANCE.md` - filled authoritative dataset provenance and known unknowns.
- `docs/EXPERIMENT_CATALOG.md` - reportable experiments mapped to scripts and evidence.
- `report/AUTHORITATIVE_RESULTS_DRAFT.md` - report-ready results/discussion with evidence
  citations.

## Active documents

- `IMPLEMENTATION_GUIDE.md` - short onboarding guide and current project design.
- `PIPELINE.md` - authoritative server commands for the professor-aligned experiment.
- `REVIEW_SAFEGUARDS.md` - scientific constraints inherited from earlier reviews.
- `SERVER_WORKFLOW.md` - connection, host/container paths, interpreter rules, security note.
- `ENVIRONMENTS.md` - Python environments, dependency boundaries, and lock files.
- `CODE_STYLE.md` - naming, docstring, and comment conventions.
- `DATASHEET_TEMPLATE.md` - fill-in-the-blank template consumed by `make_datasheets.py`
  (superseded for REPORTING by `DATA_PROVENANCE.md` above).
- `../report/REPORT_OUTLINE.md` - current report structure and required evidence.
- `../CITATIONS.md` - methods, datasets, checkpoints, and licenses.
- `../scripts/legacy/README.md` - archived one-off utilities and what replaced them.

## Historical-only documents

- `GANFP_HISTORICAL.md` - method-development history; no active metrics.
- `PROJECT_LOG.md` - chronological decision and debugging history; carries a "historical record
  only" banner. Do not cite it as the current state of anything - use the docs above instead.

## Authority order

When documents disagree:

1. Immutable evidence under `results/<run_id>/` for the run being cited
2. `docs/DATA_PROVENANCE.md`, `docs/EXPERIMENT_CATALOG.md`, `report/AUTHORITATIVE_RESULTS_DRAFT.md`
3. Latest professor feedback and `configs/config.yaml`
4. `docs/PIPELINE.md`
5. Current implementation and tests
6. `docs/REVIEW_SAFEGUARDS.md`
7. Historical entries in `docs/PROJECT_LOG.md`

Generated files under `results/<run_id>/` are evidence for one immutable run. Files directly
under `results/` (not inside a `run_id` subdirectory) are not authoritative -
`aggregate_results.py` and `verify_handover.py` both refuse to treat them as such.
