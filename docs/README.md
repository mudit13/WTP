# Documentation map

Use these documents according to their role. Only `PIPELINE.md` is an executable runbook.

## Active documents

- `IMPLEMENTATION_GUIDE.md` - short onboarding guide and current project design.
- `PIPELINE.md` - authoritative server commands for the professor-aligned experiment.
- `REVIEW_SAFEGUARDS.md` - scientific constraints inherited from earlier reviews.
- `SERVER_WORKFLOW.md` - connection, host/container paths, and interpreter rules.
- `ENVIRONMENTS.md` - Python environments and dependency boundaries.
- `DATASHEET_TEMPLATE.md` - required dataset provenance fields.
- `PROJECT_LOG.md` - chronological decision and debugging history; not a runbook.
- `../report/REPORT_OUTLINE.md` - current report structure and required evidence.
- `../CITATIONS.md` - methods, datasets, checkpoints, and licenses.

## Authority order

When documents disagree:

1. Latest professor feedback and `configs/config.yaml`
2. `docs/PIPELINE.md`
3. Current implementation and tests
4. `docs/REVIEW_SAFEGUARDS.md`
5. Historical entries in `docs/PROJECT_LOG.md`

Generated files under `results/<run_id>/` are evidence for one immutable run. Files directly
under `results/` from the superseded 7-class study are not authoritative.
